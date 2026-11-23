"""Market data and analyst price targets from Yahoo Finance (via yfinance)."""
import datetime as dt
import math
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf


def fetch_prices(tickers: list[str], days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adjusted daily (closes, opens), one column per ticker, oldest first."""
    period = f"{int(days * 1.6) + 30}d"  # calendar days covering `days` trading days
    raw = yf.download(tickers, period=period, interval="1d", auto_adjust=True,
                      progress=False, threads=True)
    out = []
    for field in ("Close", "Open"):
        df = raw[field]
        if isinstance(df, pd.Series):
            df = df.to_frame(tickers[0])
        df = df[tickers].dropna(how="all").ffill()
        missing = [t for t in tickers if df[t].isna().all()]
        if missing:
            raise RuntimeError(f"no price history for {missing}")
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        out.append(df.dropna())
    closes, opens = out
    # Before 4:15pm ET, today's bar is still moving: leave it out so a mid-day run uses yesterday's close.
    now = dt.datetime.now(ZoneInfo("America/New_York"))
    if now.time() < dt.time(16, 15) and len(closes) and closes.index[-1].date() == now.date():
        closes = closes.iloc[:-1]
    return closes, opens


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
