# Validates and cleans raw Motley Fool earnings transcripts, then saves Parquet.
import sys
import pandas as pd
from pathlib import Path

# Make `src` importable when the script is run directly from the repo root or
# from any working directory by inserting the project root onto sys.path.
_PROJECT_ROOT = Path(__file__).parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.utils import parse_dates

DATA_PATH = Path(__file__).parents[2] / "data" / "raw" / "motley-fool-data.pkl"
OUT_PATH = Path(__file__).parents[2] / "data" / "raw" / "transcripts.parquet"

REQUIRED_COLUMNS = [
    "date", "date_parsed", "call_hour", "after_close", "return_start_date",
    "is_international_hours", "ticker", "exchange", "exchange_clean",
    "q", "transcript", "transcript_len",
]


def load_raw() -> pd.DataFrame:
    """Load raw Motley Fool transcript pickle and apply list-date fix."""
    df = pd.read_pickle(DATA_PATH)
    df = fix_list_dates(df)
    return df


def fix_list_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    379 rows have ``date`` stored as a list, e.g.
    ['Brunswick (BC 0.66%) Q4 2018 ', 'Jan. 31, 2019, 10:00 a.m. ET'].
    The real date string is always the second element.
    """
    mask = df["date"].apply(lambda x: isinstance(x, list))
    n = mask.sum()
    print(f"[fix_list_dates] Fixing {n} list-type date rows")

    df = df.copy()
    df.loc[mask, "date"] = df.loc[mask, "date"].apply(
        lambda x: x[1].strip() if isinstance(x, list) and len(x) > 1 else None
    )

    still_bad = df["date"].apply(lambda x: isinstance(x, list)).sum()
    if still_bad:
        raise RuntimeError(f"{still_bad} list-type dates remain after fix — check raw data")
    return df


def clean_exchange(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive ``exchange_clean`` by splitting on ':' and stripping a leading '('.

    31 rows contain values like '(NASDAQ' or '(NYSE' with a spurious parenthesis.
    """
    df = df.copy()
    df["exchange_clean"] = (
        df["exchange"].str.split(":").str[0].str.strip().str.lstrip("(")
    )
    dirty = (df["exchange_clean"] != df["exchange"].str.split(":").str[0].str.strip()).sum()
    print(f"[clean_exchange] Stripped leading '(' from {dirty} rows")
    return df


def drop_bad_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where ``date_parsed`` is NaT (null or unparseable date strings).

    Logs each dropped ticker so no rows are silently discarded.
    """
    bad = df[df["date_parsed"].isna()]
    if bad.empty:
        print("[drop_bad_dates] No null/unparseable dates found")
        return df

    print(f"[drop_bad_dates] Dropping {len(bad)} rows with null/unparseable dates:")
    for _, row in bad.iterrows():
        print(f"  ticker={row['ticker']}  raw_date={row['date']!r}")

    return df[df["date_parsed"].notna()].copy()


def drop_duplicate_calls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate (ticker, date_parsed) pairs.

    For each duplicate group, inspect whether transcripts differ in length.
    If they appear to be the same call at different lengths, keep the longer one.
    Logs every dropped row.
    """
    dupes = df[df.duplicated(subset=["ticker", "date_parsed"], keep=False)]
    n_groups = dupes.groupby(["ticker", "date_parsed"]).ngroups
    print(f"[drop_duplicate_calls] Found {len(dupes)} rows in {n_groups} duplicate groups")

    if dupes.empty:
        return df

    # Sample a few groups to characterise duplicates
    sample_groups = list(dupes.groupby(["ticker", "date_parsed"]))[:3]
    print("[drop_duplicate_calls] Sample duplicate groups (ticker, date_parsed, transcript_len):")
    for (ticker, dt), grp in sample_groups:
        lens = grp["transcript_len"].tolist()
        print(f"  {ticker} {dt}  lengths={lens}")

    # Keep the row with the longest transcript in each (ticker, date_parsed) group
    df_sorted = df.sort_values("transcript_len", ascending=False)
    df_deduped = df_sorted.drop_duplicates(subset=["ticker", "date_parsed"], keep="first")
    dropped = len(df) - len(df_deduped)
    print(f"[drop_duplicate_calls] Dropped {dropped} shorter duplicate rows")
    return df_deduped.copy()


def add_transcript_len(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``transcript_len`` column (character count of the transcript text)."""
    df = df.copy()
    df["transcript_len"] = df["transcript"].str.len()
    return df


def inspect(df: pd.DataFrame) -> None:
    """
    Full validation audit of the raw transcript dataframe.
    Prints shape, dtypes, nulls, date range, duplicates,
    transcript length distribution, exchange breakdown,
    temporal distribution, and after-hours split.
    """
    print("\n=== Checking for list-type cells ===")
    bad_ticker = df[df["ticker"].apply(lambda x: isinstance(x, list))]
    bad_date = df[df["date"].apply(lambda x: isinstance(x, list))]
    print(f"List-type tickers: {len(bad_ticker)}")
    print(f"List-type dates:   {len(bad_date)}")

    print("\n=== Shape ===")
    print(f"Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")

    print("\n=== Columns & dtypes ===")
    print(df.dtypes.to_string())

    print("\n=== Null counts ===")
    print(df.isnull().sum().to_string())

    print("\n=== Duplicates (ticker, date) ===")
    n_dups = df.duplicated(subset=["ticker", "date"]).sum()
    print(f"  Duplicate (ticker, date) pairs: {n_dups}")

    print("\n=== Transcript length distribution ===")
    tl = df["transcript"].str.len()
    print(tl.describe().to_string())
    print(f"  Transcripts under 500 chars: {(tl < 500).sum()}")

    print("\n=== Exchange breakdown (raw) ===")
    exch_raw = df["exchange"].str.split(":").str[0].str.strip()
    print(exch_raw.value_counts().to_string())

    print("\n=== Calls per year ===")
    tmp = parse_dates(df)
    unparseable = tmp["date_parsed"].isna().sum()
    print(f"  Unparseable dates after ET strip: {unparseable}")
    print(tmp.groupby(tmp["date_parsed"].dt.year).size().to_string())

    print("\n=== After-hours split ===")
    print(tmp["after_close"].value_counts().to_string())
    print("\n  Call hour distribution:")
    print(tmp["call_hour"].value_counts().sort_index().to_string())

    intl = (tmp["call_hour"] < 7).sum()
    print(f"\n  International-hours calls (0-6 ET): {intl}")

    print("\n=== Sample rows (non-transcript columns) ===")
    preview_cols = [c for c in df.columns if c != "transcript"]
    print(df[preview_cols].head(5).to_string())


def save_parquet(df: pd.DataFrame, path: Path = OUT_PATH) -> None:
    """Write the cleaned dataframe to Parquet, keeping only required columns."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns before save: {missing}")

    out = df[REQUIRED_COLUMNS].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    print(f"[save_parquet] Wrote {len(out):,} rows → {path}")


def build_clean() -> pd.DataFrame:
    """
    Full cleaning pipeline:
      1. Load raw pickle (includes list-date fix)
      2. Add transcript_len
      3. Parse dates (adds date_parsed, call_hour, after_close, etc.)
      4. Drop rows with null/unparseable dates
      5. Drop duplicate (ticker, date_parsed) pairs — keep longest transcript
      6. Clean exchange values
    Returns the cleaned DataFrame ready for Parquet.
    """
    print("=== ES-02: transcript ingestion & cleaning ===")
    df = load_raw()
    print(f"Loaded {len(df):,} rows")

    df = add_transcript_len(df)
    df = parse_dates(df)

    intl_count = df["is_international_hours"].sum()
    print(f"[build_clean] International-hours calls (0-6 ET): {intl_count}")

    df = drop_bad_dates(df)
    df = drop_duplicate_calls(df)
    df = clean_exchange(df)

    print(f"=== Final row count: {len(df):,} ===")
    return df


if __name__ == "__main__":
    df = build_clean()
    save_parquet(df)
