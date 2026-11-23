import numpy as np
import pandas as pd

from bot import model
from bot.decide import claude_decision, rule_decision
from bot.paper import Book

T = list("ABCDEFGHI")
CFG = {"tau": 0.05, "risk_aversion": 2.4, "target_horizon_days": 252, "cov_lookback_days": 252,
       "scale_confidence_by_dispersion": True, "decider_model": "claude-fable-5-2"}


def fake_rets(seed=0, n=300):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0.0003, 0.015, (n, len(T))), columns=T)


def test_short_cap_limits_each_short():
    # The kind of raw output the user saw: heavily net short.
    raw = pd.Series([-0.30, -0.25, -0.15, -0.10, -0.05, -0.02, -0.01, -0.01, 0.11], index=T)
    assert raw.sum() < -0.75
    capped = model.apply_caps(raw, max_short=0.10, max_long=0.30)
    assert capped.min() >= -0.10 - 1e-12
    assert capped.max() <= 0.30
    assert capped.sum() > raw.sum()  # less net short; the clipped part sits in cash


def test_optional_net_floor():
    raw = pd.Series([-0.3] * 8 + [0.05], index=T)
    capped = model.apply_caps(raw, 0.10, 0.30)
    assert abs(capped.sum() - (-0.75)) < 1e-9  # per-name cap alone still allows -75% net
    floored = model.apply_caps(raw, 0.10, 0.30, min_net=-0.2)
    assert abs(floored.sum() - (-0.2)) < 1e-9
    assert floored.min() >= -0.10 and floored["I"] == 0.05


def test_no_views_gives_market_weights():
    rets = fake_rets()
    px = pd.Series(100.0, index=T)
    shares = pd.Series(np.arange(1, 10) * 1e9, index=T)
    targets = pd.DataFrame({"target_mean": np.nan, "target_high": np.nan, "target_low": np.nan,
                            "n_analysts": 0, "shares_outstanding": shares}, index=T)
    bl = model.black_litterman(rets, px, targets, CFG)
    w_mkt = shares / shares.sum()
    assert np.allclose(bl["w_raw"].to_numpy(), (w_mkt / w_mkt.abs().sum()).to_numpy(), atol=1e-6)


def test_bullish_target_raises_weight():
    rets = fake_rets()
    px = pd.Series(100.0, index=T)
    base = pd.DataFrame({"target_mean": 105.0, "target_high": 120.0, "target_low": 90.0,
                         "n_analysts": 20, "shares_outstanding": 1e9}, index=T)
    bull = base.copy()
    bull.loc["C", "target_mean"] = 160.0
    w0 = model.black_litterman(rets, px, base, CFG)["w_raw"]
    w1 = model.black_litterman(rets, px, bull, CFG)["w_raw"]
    assert w1["C"] > w0["C"]


def test_sharpe_gate_prefers_better_book():
    rets = pd.DataFrame({"A": np.r_[[0.01, -0.005] * 40], "B": np.r_[[-0.01, 0.004] * 40]})
    good, bad = pd.Series({"A": 1.0, "B": 0.0}), pd.Series({"A": 0.0, "B": 1.0})
    assert model.sharpe_gate(rets, good, bad, 63)["passed"]
    assert not model.sharpe_gate(rets, bad, good, 63)["passed"]


def test_paper_book_short_accounting():
    px = pd.Series({"A": 100.0, "B": 50.0})
    b = Book(cash=10_000, start_nav=10_000)
    b.rebalance(pd.Series({"A": 0.5, "B": -0.1}), px, cost_bps=0, date="d")
    assert abs(b.nav(px) - 10_000) < 1e-6
    assert b.positions["B"] < 0
    px2 = pd.Series({"A": 100.0, "B": 25.0})  # short halves -> gain 0.1 * 10k * 0.5
    assert abs(b.nav(px2) - 10_500) < 1e-6


def test_decision_falls_back_to_rule_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ctx = {"gate": {"candidate_sharpe": 1.0, "book_sharpe": 0.2}, "turnover": 0.3}
    d = claude_decision(ctx, CFG)
    assert d["decider"] == "rule" and d["decision"] == "execute"
    assert rule_decision({"gate": {"candidate_sharpe": 0.25, "book_sharpe": 0.2}, "turnover": 0.3})["decision"] == "hold"


def test_weekly_report_vs_benchmark(tmp_path, monkeypatch):
    from bot import report
    logs, reps = tmp_path / "logs", tmp_path / "reports"
    logs.mkdir()
    monkeypatch.setattr(report, "LOGS", logs)
    monkeypatch.setattr(report, "REPORTS", reps)
    dates = pd.bdate_range("2026-11-02", periods=10)
    pd.DataFrame({"date": dates.date, "nav": np.linspace(100_000, 102_000, 10),
                  "benchmark": np.linspace(600, 606, 10)}).to_csv(logs / "nav.csv", index=False)
    out = report.weekly_report()
    text = open(out).read()
    assert "This week" in text and "SPY" in text
    row = pd.read_csv(logs / "weekly.csv").iloc[-1]
    assert row["total_book"] > row["total_spy"] > 0


def test_execute_fills_at_next_open(tmp_path, monkeypatch):
    import sys
    from bot import paper, run_daily

    tick = ["AAA", "BBB"]
    days = pd.bdate_range("2025-06-02", periods=330)
    rng = np.random.default_rng(1)
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, (len(days), 3)), axis=0)),
                         index=days, columns=tick + ["SPY"])
    open_ = close.shift(1).fillna(close) * 1.001  # opens differ from the prior close
    tg = pd.DataFrame({"target_mean": [150.0, 90.0], "target_median": [150.0, 90.0], "target_high": [170.0, 110.0],
                       "target_low": [120.0, 70.0], "n_analysts": [20, 20], "shares_outstanding": [1e9, 1e9]}, index=tick)
    cfg = {**run_daily.load_config(), "tickers": tick, "min_turnover": 0.0}
    monkeypatch.setattr(run_daily, "load_config", lambda: cfg)
    monkeypatch.setattr(run_daily, "LOGS", tmp_path / "logs")
    monkeypatch.setattr(paper, "BOOK_PATH", tmp_path / "book.json")
    monkeypatch.setattr(paper, "STATE", tmp_path)
    monkeypatch.setattr(run_daily.model, "sharpe_gate", lambda *a: {"candidate_sharpe": 2.0, "book_sharpe": 0.0, "passed": True})
    monkeypatch.setattr(run_daily.data, "fetch_targets", lambda t: tg)

    def run(upto, *argv):
        monkeypatch.setattr(run_daily.data, "fetch_prices", lambda t, d: (close.loc[:upto], open_.loc[:upto]))
        monkeypatch.setattr(sys, "argv", ["run_daily", *argv])
        run_daily.main()

    d1, d2 = days[-2], days[-1]
    run(d1, "--decision", "execute", "--reason", "test")
    book = paper.Book.load(0)
    assert book.pending and book.pending["decided"] == str(d1.date())
    trades_day1 = pd.read_csv(tmp_path / "logs" / "trades.csv")
    assert set(trades_day1["date"]) == {str(d1.date())}  # only the equal-weight seed so far

    run(d2, "--decision", "hold", "--reason", "test")
    book = paper.Book.load(0)
    assert book.pending is None
    trades = pd.read_csv(tmp_path / "logs" / "trades.csv")
    filled = trades[trades["date"] == str(d2.date())]
    assert len(filled) == 2
    assert np.allclose(filled.set_index("ticker")["price"], open_.loc[d2, tick].round(4))
