# Purpose of this file is to view the raw transcript data in detail, identify 
# any issues, and apply fixes before saving a cleaned version for analysis.
import pandas as pd
from pathlib import Path
import pytz

DATA_PATH = Path(__file__).parents[2] / "data" / "raw" / "motley-fool-data.pkl"
MARKET_CLOSE_HOUR = 16  # 4pm ET


def load_raw() -> pd.DataFrame:
    """Load raw Motley Fool transcript pickle."""
    df = pd.read_pickle(DATA_PATH)
    df = fix_list_dates(df)
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the date column, stripping unrecognized 'ET' suffix and
    localizing to America/New_York to handle EST/EDT automatically.
    Adds: date_parsed, call_hour, after_close, return_start_date.
    """
    cleaned = df["date"].str.replace(r"\s*ET\s*$", "", regex=True)
    parsed = pd.to_datetime(cleaned, errors="coerce")
    parsed = parsed.dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="NaT"
    )

    df = df.copy()
    df["date_parsed"]      = parsed
    df["call_hour"]        = parsed.dt.hour
    df["after_close"]      = parsed.dt.hour >= MARKET_CLOSE_HOUR
    df["return_start_date"] = parsed.dt.normalize()
    df.loc[df["after_close"], "return_start_date"] = (
        df.loc[df["after_close"], "date_parsed"] + pd.Timedelta(days=1)
    ).dt.normalize()

    return df

def fix_list_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    379 rows have date stored as a list e.g.
    ['Brunswick (BC 0.66%) Q4 2018 ', 'Jan. 31, 2019, 10:00 a.m. ET']
    The actual date string is always the second element.
    """
    mask = df['date'].apply(lambda x: isinstance(x, list))
    n = mask.sum()

    # inspect a few before fixing
    print(f"  Fixing {n} list-type date rows")
    print("  Sample raw values:")
    for val in df.loc[mask, 'date'].head(3):
        print(f"    {val}")

    df = df.copy()
    df.loc[mask, 'date'] = df.loc[mask, 'date'].apply(
        lambda x: x[1].strip() if isinstance(x, list) and len(x) > 1 else None
    )

    # verify the fix
    still_bad = df['date'].apply(lambda x: isinstance(x, list)).sum()
    print(f"  List-type dates remaining after fix: {still_bad}")

    return df

def inspect(df: pd.DataFrame) -> None:
    """
    Full validation audit of the raw transcript dataframe.
    Prints shape, dtypes, nulls, date range, duplicates,
    transcript length distribution, exchange breakdown,
    temporal distribution, and after-hours split.
    """
    # add this temporarily at the top of inspect(), before the duplicates check
    print("\n=== Checking for list-type cells ===")
    bad_ticker = df[df['ticker'].apply(lambda x: isinstance(x, list))]
    bad_date   = df[df['date'].apply(lambda x: isinstance(x, list))]
    print(f"List-type tickers: {len(bad_ticker)}")
    print(f"List-type dates:   {len(bad_date)}")
    if len(bad_ticker) > 0:
        print(bad_ticker[['ticker', 'date', 'exchange', 'q']].head())
    if len(bad_date) > 0:
        print(bad_date[['ticker', 'date', 'exchange', 'q']].head())
    print("=== Shape ===")
    print(f"Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")

    print("\n=== Columns & dtypes ===")
    print(df.dtypes.to_string())

    print("\n=== Null counts ===")
    print(df.isnull().sum().to_string())

    print("\n=== Duplicates ===")
    n_dups = df.duplicated(subset=["ticker", "date"]).sum()
    print(f"  Duplicate (ticker, date) pairs: {n_dups}")

    print("\n=== Transcript length distribution ===")
    df["transcript_len"] = df["transcript"].str.len()
    print(df["transcript_len"].describe().to_string())
    short = df[df["transcript_len"] < 500]
    print(f"  Transcripts under 500 chars: {len(short)}")

    print("\n=== Exchange breakdown ===")
    df["exchange_clean"] = df["exchange"].str.split(":").str[0].str.strip()
    print(df["exchange_clean"].value_counts().to_string())

    print("\n=== Calls per year ===")
    df = parse_dates(df)
    unparseable = df["date_parsed"].isna().sum()
    print(f"  Unparseable dates after ET strip: {unparseable}")
    print(df.groupby(df["date_parsed"].dt.year).size().to_string())

    print("\n=== After-hours split ===")
    print(df["after_close"].value_counts().to_string())
    print("\n  Call hour distribution:")
    print(df["call_hour"].value_counts().sort_index().to_string())

    print("\n=== Sample rows (non-transcript columns) ===")
    preview_cols = [c for c in df.columns if c != "transcript"]
    print(df[preview_cols].head(5).to_string())

    print("\n=== 3 raw transcript samples ===")
    for i in [0, 100, 1000]:
        print(f"\n--- ROW {i} ({df['ticker'].iloc[i]}, {df['date'].iloc[i]}) ---")
        print(df["transcript"].iloc[i][:1000])
        print("...")


if __name__ == "__main__":
    df = load_raw()
    inspect(df)