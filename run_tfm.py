"""CLI: python run_tfm.py --help"""
import argparse
from pathlib import Path

from tfm.experiment import run
from tfm.models import ModelConfig


def main():
    parser = argparse.ArgumentParser(description="Chronological SRP tabular foundation/deep model comparison")
    parser.add_argument("--source", type=Path, default=Path(__file__).with_name("matches_multiseason.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/tfm"))
    parser.add_argument("--models", nargs="+", choices=["lightgbm", "tabpfn", "tabnet", "ft_transformer"], default=["lightgbm", "tabpfn", "tabnet", "ft_transformer"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--test-seasons", nargs="+")
    parser.add_argument("--market", choices=["prematch", "closing"], default="prematch")
    parser.add_argument("--feature-set", choices=["baseline", "market_blind"], default="baseline")
    parser.add_argument("--max-train-rows", type=int, default=1000, help="Recent training rows, same cap for every model; 0=uncapped")
    parser.add_argument("--min-season-matches", type=int, default=1000)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--tabpfn-estimators", type=int, default=4)
    parser.add_argument("--tabpfn-version", choices=["v2", "v2.5"], default="v2")
    parser.add_argument("--tabpfn-fit-mode", choices=["fit_preprocessors", "fit_with_cache"],
                        default="fit_preprocessors", help="Cache attention state for repeated queries; uses more memory")
    parser.add_argument("--checkpoint", help="Optional local TabPFN checkpoint instead of the named version")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0, .01, .02, .05, .1])
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if any(m != "lightgbm" for m in args.models):
        import torch
        torch.set_num_threads(args.threads)
    config = ModelConfig(device=args.device, epochs=args.epochs, patience=args.patience,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        tabpfn_estimators=args.tabpfn_estimators, tabpfn_version=args.tabpfn_version,
        tabpfn_fit_mode=args.tabpfn_fit_mode,
        checkpoint=args.checkpoint)
    metrics = run(args.source, args.output, models=args.models, seeds=args.seeds,
        market=args.market, feature_set=args.feature_set, max_train_rows=args.max_train_rows,
        min_season_matches=args.min_season_matches, test_seasons=args.test_seasons,
        model_config=config, thresholds=args.thresholds, n_boot=args.bootstrap)
    print(metrics.to_string(index=False))
    print(f"Saved experiment to {args.output.resolve()}")


if __name__ == "__main__":
    main()
