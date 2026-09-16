"""Optional model adapters. All probability columns are explicitly H, D, A."""
from copy import deepcopy
from dataclasses import dataclass
import importlib
import random

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def dependency(module, package):
    try:
        return importlib.import_module(module)
    except (ImportError, OSError) as exc:
        raise RuntimeError(f"Cannot load {package}. Install requirements-tfm.txt; original error: {exc}") from exc


def probabilities(p, classes=(0, 1, 2)):
    p = np.asarray(p, dtype=np.float64)
    classes = np.asarray(classes)
    if p.ndim != 2 or p.shape[1] != 3 or set(classes.tolist()) != {0, 1, 2}:
        raise ValueError("Model must return probabilities for all three classes H=0,D=1,A=2")
    p = p[:, [int(np.flatnonzero(classes == k)[0]) for k in range(3)]]
    if not np.isfinite(p).all() or (p < 0).any() or (p.sum(axis=1) <= 0).any():
        raise ValueError("Invalid model probabilities")
    p = np.clip(p, 1e-12, 1)
    return p / p.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class ModelConfig:
    seed: int = 42
    device: str = "cpu"
    epochs: int = 100
    patience: int = 15
    batch_size: int = 256
    learning_rate: float = 0.0003
    tabpfn_estimators: int = 4
    tabpfn_version: str = "v2"
    tabpfn_fit_mode: str = "fit_preprocessors"
    checkpoint: str | None = None


class TabularModel:
    def __init__(self, name, config=ModelConfig()):
        if name not in {"lightgbm", "tabpfn", "tabnet", "ft_transformer"}:
            raise ValueError(f"Unknown model: {name}")
        if min(config.epochs, config.patience, config.batch_size, config.tabpfn_estimators) < 1:
            raise ValueError("Epochs, patience, batch size and ensemble size must be positive")
        if config.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        self.name, self.config = name, config

    def fit(self, X, y, X_stop, y_stop):
        if set(np.unique(y)) != {0, 1, 2}:
            raise ValueError("Training partition must contain H, D and A")
        cfg = self.config
        self.feature_names_ = list(X.columns)
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler_ = StandardScaler()
        train = self.imputer_.fit_transform(X.replace([np.inf, -np.inf], np.nan))
        stop = self.imputer_.transform(X_stop.replace([np.inf, -np.inf], np.nan))
        # Identical train-only preprocessing for all arms.
        train = self.scaler_.fit_transform(train).astype(np.float32)
        stop = self.scaler_.transform(stop).astype(np.float32)
        y, y_stop = np.array(y, dtype=np.int64, copy=True), np.array(y_stop, dtype=np.int64, copy=True)
        self.best_epoch_ = None
        if self.name == "lightgbm":
            lgb = dependency("lightgbm", "lightgbm")
            self.model_ = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.01,
                num_leaves=15, max_depth=3, subsample=0.7, colsample_bytree=0.7,
                random_state=cfg.seed, verbosity=-1, n_jobs=1)
            self.model_.fit(train, y)
        elif self.name == "tabpfn":
            tabpfn = dependency("tabpfn", "tabpfn")
            kwargs = dict(device=cfg.device, n_estimators=cfg.tabpfn_estimators,
                          random_state=cfg.seed, fit_mode=cfg.tabpfn_fit_mode)
            if cfg.checkpoint:
                self.model_ = tabpfn.TabPFNClassifier(model_path=cfg.checkpoint, **kwargs)
            else:
                constants = dependency("tabpfn.constants", "tabpfn")
                version = {"v2": constants.ModelVersion.V2, "v2.5": constants.ModelVersion.V2_5}[cfg.tabpfn_version]
                self.model_ = tabpfn.TabPFNClassifier.create_default_for_version(version, **kwargs)
            # In-context learning; neither the stopping nor calibration/test
            # labels are passed into the foundation model's context.
            self.model_.fit(train, y)
        elif self.name == "tabnet":
            torch = dependency("torch", "torch")
            torch.manual_seed(cfg.seed)
            TabNet = dependency("pytorch_tabnet.tab_model", "pytorch-tabnet").TabNetClassifier
            self.model_ = TabNet(n_d=16, n_a=16, n_steps=3, seed=cfg.seed,
                device_name=cfg.device, verbose=0, optimizer_params={"lr": 0.02})
            # TabNet's batch normalization needs at least two rows per batch.
            batch = min(cfg.batch_size, len(train))
            if batch < 2:
                raise ValueError("TabNet requires batch_size >= 2")
            self.model_.fit(train, y, eval_set=[(stop, y_stop)], eval_metric=["logloss"],
                max_epochs=cfg.epochs, patience=cfg.patience, batch_size=batch,
                virtual_batch_size=batch, num_workers=0, drop_last=True)
            self.best_epoch_ = int(self.model_.best_epoch) + 1
        else:
            self._fit_transformer(train, y, stop, y_stop)
        return self

    def _fit_transformer(self, train, y, stop, y_stop):
        torch = dependency("torch", "torch")
        FTTransformer = dependency("rtdl_revisiting_models", "rtdl-revisiting-models").FTTransformer
        cfg = self.config
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.device_ = torch.device(cfg.device)
        self.model_ = FTTransformer(n_cont_features=train.shape[1], cat_cardinalities=[],
            d_out=3, n_blocks=2, d_block=64, attention_n_heads=8,
            attention_dropout=0.2, ffn_d_hidden=None, ffn_d_hidden_multiplier=4 / 3,
            ffn_dropout=0.1, residual_dropout=0.0).to(self.device_)
        optimizer = torch.optim.AdamW(self.model_.make_parameter_groups(),
            lr=cfg.learning_rate, weight_decay=1e-5)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            torch.from_numpy(train), torch.from_numpy(y)), batch_size=cfg.batch_size,
            shuffle=True, generator=torch.Generator().manual_seed(cfg.seed))
        best_loss, best_state, bad_epochs = float("inf"), None, 0
        for epoch in range(cfg.epochs):
            self.model_.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(self.model_(xb.to(self.device_), None), yb.to(self.device_))
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite FT-Transformer training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
            from sklearn.metrics import log_loss
            val_loss = log_loss(y_stop, self._transformer_predict(stop), labels=[0, 1, 2])
            if val_loss < best_loss - 1e-6:
                best_loss, best_state, bad_epochs = val_loss, deepcopy(self.model_.state_dict()), 0
                self.best_epoch_ = epoch + 1
            else:
                bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break
        self.model_.load_state_dict(best_state)
        self.model_.eval()

    def _transformer_predict(self, X):
        torch = dependency("torch", "torch")
        self.model_.eval()
        batches = []
        with torch.no_grad():
            for start in range(0, len(X), self.config.batch_size):
                batch = torch.from_numpy(X[start:start + self.config.batch_size]).to(self.device_)
                batches.append(self.model_(batch, None).softmax(-1).cpu().numpy())
        return probabilities(np.concatenate(batches))

    def predict_proba(self, X):
        if list(X.columns) != self.feature_names_:
            raise ValueError("Feature names/order differ from fitted schema")
        if len(X) == 0:
            return np.empty((0, 3))
        X = self.scaler_.transform(self.imputer_.transform(X.replace([np.inf, -np.inf], np.nan))).astype(np.float32)
        if self.name == "ft_transformer":
            return self._transformer_predict(X)
        # Limit query batch memory, retaining the same fitted context.
        batches = [self.model_.predict_proba(X[i:i + self.config.batch_size])
                   for i in range(0, len(X), self.config.batch_size)]
        return probabilities(np.concatenate(batches), self.model_.classes_)
