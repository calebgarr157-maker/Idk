"""Run once per trading day after the close.

  python -m bot.run_daily            # normal daily run (skips if today is already logged)
  python -m bot.run_daily --force    # re-run today (replaces nothing in the book if it already traded)
  python -m bot.run_daily --dry-run  # compute and print, write nothing
  python -m bot.run_daily --dry-run --show-context   # print the full proposal for a decider
  python -m bot.run_daily --decision hold --reason "..."   # apply a decision made outside the script
"""
import argparse
import datetime as dt
import json

import pandas as pd

from . import data, model
from .common import LOGS, append_rows, load_config, read_rows, upsert_row
from .decide import claude_decision
from .paper import Book
from .report import weekly_report


def target_changes(targets: pd.DataFrame) -> dict:
    """Which tickers' consensus target moved since the last logged run."""
    last = {}
    for r in read_rows(LOGS / "targets.csv"):
        last[r["ticker"]] = r
    out = {}
    for t, row in targets.iterrows():
        prev = last.get(t)
        if prev is None or not prev["target_mean"]:
            out[t] = "new"
            continue
        old = float(prev["target_mean"])
        if pd.notna(row["target_mean"]) and abs(row["target_mean"] / old - 1) > 0.001:
            out[t] = f"{old:.2f}->{row['target_mean']:.2f}"
        elif int(prev["n_analysts"]) != row["n_analysts"]:
            out[t] = f"analysts {prev['n_analysts']}->{row['n_analysts']}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--show-context", action="store_true", help="print the full proposal sent to the decider")
    ap.add_argument("--decision", choices=["execute", "hold"],
                    help="use this decision instead of calling the API (e.g. made by a Claude session)")
    ap.add_argument("--reason", default="", help="reason for --decision")
    ap.add_argument("--decider", default="claude session", help="who made --decision")
    args = ap.parse_args()
    if args.decision and not args.reason:
        ap.error("--decision needs --reason")
    cfg = load_config()
    tickers, bench = cfg["tickers"], cfg["benchmark"]

    closes = data.fetch_closes(tickers + [bench], cfg["cov_lookback_days"] + cfg["sharpe_window_days"] + 10)
    date = str(closes.index[-1].date())
    if not args.force and not args.dry_run and any(r["date"] == date for r in read_rows(LOGS / "decisions.csv")):
        print(f"{date} already logged; use --force to re-run")
        return

    targets = data.fetch_targets(tickers)
    changes = target_changes(targets)
    px = closes.iloc[-1]
    rets = model.log_returns(closes[tickers])

    bl = model.black_litterman(rets, px[tickers], targets, cfg)
    candidate = model.apply_caps(bl["w_raw"], cfg["max_single_short"], cfg["max_single_long"], cfg.get("min_net_exposure"))

    book = Book.load(cfg["start_capital"])
    seed_trades = []
    if not book.start_date:
        # Like the notebook (fin_weights_list starts at 1/9 each), the book starts equal-weight,
        # so the first Sharpe gate compares the model against a real portfolio rather than cash.
        book.start_date, book.benchmark_start = date, float(px[bench])
        seed = pd.Series(1 / len(tickers), index=tickers)
        seed_trades = book.rebalance(seed, px, cfg["cost_bps"], date)
        book.start_nav = book.nav(px)
    borrow = 0.0
    if book.last_date and book.last_date < date:
        days = (dt.date.fromisoformat(date) - dt.date.fromisoformat(book.last_date)).days
        borrow = book.accrue_borrow(px, days, cfg["short_borrow_bps_annual"])
    book_w = book.weights(px, tickers)
    nav = book.nav(px)

    gate = model.sharpe_gate(rets, candidate, book_w, cfg["sharpe_window_days"])
    turnover = float((candidate - book_w).abs().sum())

    nav_hist = pd.DataFrame(read_rows(LOGS / "nav.csv"))
    peak = max([nav] + ([float(x) for x in nav_hist["nav"]] if len(nav_hist) else []))
    ctx = {
        "date": date,
        "per_ticker": [
            {
                "ticker": t,
                "price": round(float(px[t]), 2),
                "target_mean": targets.loc[t, "target_mean"],
                "target_low": targets.loc[t, "target_low"],
                "target_high": targets.loc[t, "target_high"],
                "n_analysts": int(targets.loc[t, "n_analysts"]),
                "upside_to_target": round(float(targets.loc[t, "target_mean"] / px[t] - 1), 4) if pd.notna(targets.loc[t, "target_mean"]) else None,
                "target_change_since_last_run": changes.get(t),
                "current_weight": round(float(book_w[t]), 4),
                "proposed_weight": round(float(candidate[t]), 4),
                "uncapped_model_weight": round(float(bl["w_raw"][t]), 4),
            }
            for t in tickers
        ],
        "gate": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in gate.items()},
        "sharpe_window_days": cfg["sharpe_window_days"],
        "turnover": round(turnover, 4),
        "est_trading_cost_usd": round(turnover * nav * cfg["cost_bps"] / 1e4, 2),
        "exposure": {
            "current_net": round(float(book_w.sum()), 4), "current_gross": round(float(book_w.abs().sum()), 4),
            "proposed_net": round(float(candidate.sum()), 4), "proposed_gross": round(float(candidate.abs().sum()), 4),
            "uncapped_net": round(float(bl["w_raw"].sum()), 4),
            "caps": {"max_single_short": cfg["max_single_short"], "max_single_long": cfg["max_single_long"],
                     "min_net_exposure": cfg.get("min_net_exposure")},
        },
        "book": {
            "nav": round(nav, 2), "start_date": book.start_date, "start_nav": book.start_nav,
            "return_since_start": round(nav / book.start_nav - 1, 4),
            "benchmark_return_since_start": round(float(px[bench]) / book.benchmark_start - 1, 4),
            "drawdown_from_peak": round(nav / peak - 1, 4),
        },
        "recent_decisions": [
            {k: r[k] for k in ("date", "decision", "reason")} for r in read_rows(LOGS / "decisions.csv")[-5:]
        ],
    }

    if not gate["passed"]:
        dec = {"decision": "hold", "decider": "gate",
               "reason": f"Sharpe gate failed: candidate {gate['candidate_sharpe']:.2f} vs book {gate['book_sharpe']:.2f}"}
    elif turnover < cfg["min_turnover"]:
        dec = {"decision": "hold", "decider": "gate", "reason": f"turnover {turnover:.1%} below minimum {cfg['min_turnover']:.0%}"}
    elif args.decision:
        dec = {"decision": args.decision, "reason": args.reason.strip(), "decider": args.decider}
    else:
        dec = claude_decision(ctx, cfg)

    needs_decision = gate["passed"] and turnover >= cfg["min_turnover"]
    if args.show_context:
        print("DECISION NEEDED" if needs_decision else "NO DECISION NEEDED (gate holds)")
        print(json.dumps(ctx, indent=2, default=float))
    print(json.dumps({"date": date, "gate": ctx["gate"], "turnover": ctx["turnover"],
                      "exposure": ctx["exposure"], **dec}, indent=2))
    print(pd.DataFrame(ctx["per_ticker"]).set_index("ticker")[
        ["price", "target_mean", "upside_to_target", "n_analysts", "current_weight", "proposed_weight", "uncapped_model_weight"]
    ].to_string())
    if args.dry_run:
        return

    # 1. log every target
    append_rows(LOGS / "targets.csv", [
        {"date": date, "ticker": t, "price": round(float(px[t]), 4),
         "target_mean": targets.loc[t, "target_mean"], "target_median": targets.loc[t, "target_median"],
         "target_high": targets.loc[t, "target_high"], "target_low": targets.loc[t, "target_low"],
         "n_analysts": int(targets.loc[t, "n_analysts"]), "view_daily": round(float(bl["view"][t]), 6),
         "changed": changes.get(t, "")}
        for t in tickers
    ])
    # 2. trade if Claude (or the rule) said execute
    trades = book.rebalance(candidate, px, cfg["cost_bps"], date) if dec["decision"] == "execute" else []
    append_rows(LOGS / "trades.csv", seed_trades + trades)
    # 3. log the decision
    append_rows(LOGS / "decisions.csv", [{
        "date": date, "decision": dec["decision"], "decider": dec["decider"], "reason": dec["reason"],
        "gate_passed": gate["passed"], "candidate_sharpe": round(gate["candidate_sharpe"], 4),
        "book_sharpe": round(gate["book_sharpe"], 4), "turnover": round(turnover, 4),
        "proposed_net": round(float(candidate.sum()), 4), "uncapped_net": round(float(bl["w_raw"].sum()), 4),
        "targets_changed": ";".join(f"{k}:{v}" for k, v in changes.items()),
        "proposed_weights": json.dumps({t: round(float(w), 4) for t, w in candidate.items()}),
    }])
    # 4. mark the book
    book.last_date = date
    nav = book.nav(px)
    w_after = book.weights(px, tickers)
    upsert_row(LOGS / "nav.csv", {
        "date": date, "nav": round(nav, 2), "cash": round(book.cash, 2), "benchmark": round(float(px[bench]), 4),
        "net": round(float(w_after.sum()), 4), "gross": round(float(w_after.abs().sum()), 4),
        "borrow_fee": round(borrow, 2),
    })
    book.save()

    # 5. weekly comparison with SPY on Fridays (or if the last one is a week old)
    last_week = read_rows(LOGS / "weekly.csv")
    due = dt.date.fromisoformat(date).weekday() == 4 or (
        last_week and (dt.date.fromisoformat(date) - dt.date.fromisoformat(last_week[-1]["date"])).days >= 7)
    if due:
        print("weekly report:", weekly_report())


if __name__ == "__main__":
    main()
