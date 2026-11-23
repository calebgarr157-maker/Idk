"""Market data and analyst price targets from Yahoo Finance (via yfinance)."""
import math

import pandas as pd
import yfinance as yf


def fetch_closes(tickers: list[str], days: int) -> pd.DataFrame:
    """Adjusted daily closes, one column per ticker, oldest first."""
    period = f"{int(days * 1.6) + 30}d"  # calendar days covering `days` trading days
    df = yf.download(tickers, period=period, interval="1d", auto_adjust=True,
                     progress=False, threads=True)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0])
    df = df[tickers].dropna(how="all").ffill()
    missing = [t for t in tickers if df[t].isna().all()]
    if missing:
        raise RuntimeError(f"no price history for {missing}")
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df.dropna()


def _num(x):
    try:
        x = float(x)
        return x if math.isfinite(x) and x > 0 else None
    except (TypeError, ValueError):
        return None


def fetch_targets(tickers: list[str]) -> pd.DataFrame:
    """Consensus analyst price targets and shares outstanding per ticker."""
    rows = []
    for t in tickers:
        info = yf.Ticker(t).info or {}
        rows.append({
            "ticker": t,
            "target_mean": _num(info.get("targetMeanPrice")),
            "target_median": _num(info.get("targetMedianPrice")),
            "target_high": _num(info.get("targetHighPrice")),
            "target_low": _num(info.get("targetLowPrice")),
            "n_analysts": int(info.get("numberOfAnalystOpinions") or 0),
            "shares_outstanding": _num(info.get("sharesOutstanding")),
        })
    return pd.DataFrame(rows).set_index("ticker")
