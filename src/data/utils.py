"""Shared data utilities reused across ES-02, ES-03, and beyond."""
import pandas as pd

MARKET_CLOSE_HOUR = 16  # 4 PM ET


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the raw ``date`` string column into structured datetime columns.

    Strips the trailing 'ET' timezone suffix (which pandas cannot parse),
    localizes to America/New_York so EST/EDT offsets are handled automatically,
    then derives:
      - ``date_parsed``         — tz-aware datetime in ET
      - ``call_hour``           — integer hour (0-23) in ET
      - ``after_close``         — True when call_hour >= 16
      - ``return_start_date``   — date the return window opens; next calendar
                                  day for after-close calls, same day otherwise
      - ``is_international_hours`` — True when call_hour < 7 (likely foreign HQ)
    """
    cleaned = df["date"].str.replace(r"\s*ET\s*$", "", regex=True)
    parsed = pd.to_datetime(cleaned, format="mixed", errors="coerce")
    parsed = parsed.dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="NaT"
    )

    df = df.copy()
    df["date_parsed"] = parsed
    df["call_hour"] = parsed.dt.hour
    df["after_close"] = parsed.dt.hour >= MARKET_CLOSE_HOUR
    df["is_international_hours"] = parsed.dt.hour < 7

    df["return_start_date"] = parsed.dt.normalize()
    df.loc[df["after_close"], "return_start_date"] = (
        df.loc[df["after_close"], "date_parsed"] + pd.Timedelta(days=1)
    ).dt.normalize()

    return df
