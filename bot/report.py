"""Weekly paper-book vs SPY comparison, written to logs/weekly.csv and reports/weekly/<date>.md."""
import numpy as np
import pandas as pd

from .common import LOGS, REPORTS, load_config, upsert_row


def _stats(nav: pd.Series) -> dict:
    r = nav.pct_change().dropna()
    dd = (nav / nav.cummax() - 1).min()
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(252) if len(r) > 1 and r.std(ddof=1) > 0 else float("nan")
    return {"total": nav.iloc[-1] / nav.iloc[0] - 1, "sharpe": sharpe, "max_dd": dd,
            "vol": r.std(ddof=1) * np.sqrt(252) if len(r) > 1 else float("nan")}


def weekly_report(as_of: str | None = None) -> str | None:
    path = LOGS / "nav.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").set_index("date")
    if as_of:
        df = df[df.index <= pd.Timestamp(as_of)]
    if len(df) < 2:
        return None
    end = df.index[-1]
    week = df[df.index > end - pd.Timedelta(days=7)]
    prior = df[df.index <= end - pd.Timedelta(days=7)]
    base = prior.iloc[-1] if len(prior) else df.iloc[0]
    wk_book = df["nav"].iloc[-1] / base["nav"] - 1
    wk_spy = df["benchmark"].iloc[-1] / base["benchmark"] - 1
    book, spy = _stats(df["nav"]), _stats(df["benchmark"])
    bench = load_config()["benchmark"]

    upsert_row(LOGS / "weekly.csv", {
        "date": end.date(), "week_book": round(wk_book, 6), "week_spy": round(wk_spy, 6),
        "week_excess": round(wk_book - wk_spy, 6), "total_book": round(book["total"], 6),
        "total_spy": round(spy["total"], 6), "sharpe_book": round(book["sharpe"], 3),
        "sharpe_spy": round(spy["sharpe"], 3), "maxdd_book": round(book["max_dd"], 4),
        "maxdd_spy": round(spy["max_dd"], 4), "nav": round(df["nav"].iloc[-1], 2),
    })

    dec = pd.read_csv(LOGS / "decisions.csv", parse_dates=["date"]) if (LOGS / "decisions.csv").exists() else pd.DataFrame()
    wk_dec = dec[dec["date"] > end - pd.Timedelta(days=7)] if len(dec) else dec
    lines = [
        f"# Paper book vs {bench}: week ending {end.date()}",
        "",
        f"Tracking since {df.index[0].date()} ({len(df)} trading days). NAV ${df['nav'].iloc[-1]:,.0f}.",
        "",
        f"| | Book | {bench} |",
        "|---|---:|---:|",
        f"| This week | {wk_book:+.2%} | {wk_spy:+.2%} |",
        f"| Since start | {book['total']:+.2%} | {spy['total']:+.2%} |",
        f"| Sharpe (ann.) | {book['sharpe']:.2f} | {spy['sharpe']:.2f} |",
        f"| Volatility (ann.) | {book['vol']:.1%} | {spy['vol']:.1%} |",
        f"| Max drawdown | {book['max_dd']:.2%} | {spy['max_dd']:.2%} |",
        "",
        f"Excess return this week: {wk_book - wk_spy:+.2%}.",
        "",
        "## Decisions this week",
        "",
    ]
    if len(wk_dec):
        for _, r in wk_dec.iterrows():
            lines.append(f"- {r['date'].date()}: **{r['decision']}** ({r['decider']}): {r['reason']}")
    else:
        lines.append("- none logged")
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"{end.date()}.md"
    out.write_text("\n".join(lines) + "\n")
    return str(out)


if __name__ == "__main__":
    print(weekly_report() or "not enough NAV history yet")
