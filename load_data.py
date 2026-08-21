"""Loader for the football-data.co.uk 2025/26 dataset.

`load(source)` accepts either shape the project has used:

  * a directory of per-division CSVs (``csv_out/E0.csv``, ``csv_out/D1.csv``, ...)
    as downloaded from football-data.co.uk, or
  * the single concatenated export ``all-euro-data-2025-2026.csv``, which is those
    same 22 divisions stacked into one semicolon-delimited file with a repeated
    header row and a blank line between each block.

If the named directory is absent it falls back to the concatenated file, so the
same call works on either machine.

Returns a DataFrame with the raw columns plus ``League`` (EDA.py needs ``League``;
the raw column is ``Div``). ``Date`` is returned already parsed to datetime64 —
EDA.py's own ``pd.to_datetime`` call then becomes a harmless no-op, which avoids
pandas falling back to per-element dateutil parsing on ambiguous d/m vs m/d dates.
"""
import glob
import io
import os

import pandas as pd

CONCATENATED = "all-euro-data-2025-2026.csv"

# Data sits next to this module. Resolving against it means the loader works no
# matter what directory the script was launched from (VS Code's run button and a
# plain `python EDA.py` do not always agree on the working directory).
_HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve(path):
    """Return `path` if it exists as given, otherwise the copy beside this module."""
    if os.path.exists(path):
        return path
    return os.path.join(_HERE, path)

# Ordered by how football-data writes dates; the first format that parses every
# value wins. Day-first is tried first because the raw site files use it, while
# the concatenated export has been through Excel and comes out month-first.
_DATE_FORMATS = ("%d/%m/%Y", "%d/%m/%y", "%m/%d/%Y", "%m/%d/%y")


def _parse_dates(s):
    """Parse a date column by finding the one format that fits the whole column.

    Trying formats rather than letting pandas infer avoids silently swapping day
    and month on ambiguous values like 1/12/26.
    """
    s = s.astype(str).str.strip()
    valid = s.ne("") & s.ne("nan")
    best, best_hits = None, -1
    for fmt in _DATE_FORMATS:
        hits = pd.to_datetime(s.where(valid), format=fmt, errors="coerce").notna().sum()
        if hits > best_hits:
            best, best_hits = fmt, hits
        if hits == valid.sum():          # exact fit, no need to keep looking
            break
    return pd.to_datetime(s.where(valid), format=best, errors="coerce")


def _read_concatenated(path):
    """Split the stacked export on its blank lines and read each block separately."""
    with open(path, "r", encoding="utf-8-sig") as fh:
        text = fh.read()

    blocks, current = [], []
    for line in text.splitlines():
        if not line.strip():
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current))

    frames = [
        pd.read_csv(io.StringIO(b), sep=";", decimal=",", low_memory=False)
        for b in blocks
    ]
    return pd.concat(frames, ignore_index=True)


def _read_directory(path):
    """Read every per-division CSV in a directory, keeping the filename as Div."""
    files = sorted(glob.glob(os.path.join(path, "*.csv")))
    if not files:
        raise FileNotFoundError(f"no CSV files found in {path!r}")

    frames = []
    for f in files:
        # football-data ships comma-delimited files; sep=None sniffs it and still
        # copes if a division has been re-exported with semicolons.
        part = pd.read_csv(f, sep=None, engine="python", low_memory=False)
        if "Div" not in part.columns:
            part["Div"] = os.path.splitext(os.path.basename(f))[0]
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def load(source="csv_out"):
    source, fallback = _resolve(source), _resolve(CONCATENATED)
    if os.path.isdir(source):
        df = _read_directory(source)
    elif os.path.isfile(source):
        df = _read_concatenated(source)
    elif os.path.isfile(fallback):
        df = _read_concatenated(fallback)
    else:
        raise FileNotFoundError(
            f"neither {source!r} nor {fallback!r} exists (cwd is {os.getcwd()!r})"
        )

    df = df[df["Div"].notna()].copy()
    df["League"] = df["Div"]
    df["Date"] = _parse_dates(df["Date"])
    return df.reset_index(drop=True)
