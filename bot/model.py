"""Black-Litterman with analyst price targets as views, the Case 3 Sharpe gate, and position caps.

Mapping from the original notebook (case3/case3_local.ipynb):
  * analyst predicted price paths  -> Yahoo consensus 12-month price target
  * view = next-day predicted log return  -> log(target / price) / target_horizon_days
  * Omega = tau * P Sigma P'  (their conf_mat)  -> same, optionally scaled per name by
    analyst dispersion ((high - low) / mean), so names analysts disagree on get less weight
  * market weights from shares outstanding x price  -> same, with Yahoo shares outstanding
  * Sharpe gate: a candidate only replaces the book when it has the higher Sharpe
    -> same idea, measured on trailing realized daily returns of both weight vectors
"""
import numpy as np
import pandas as pd


def log_returns(closes: pd.DataFrame) -> pd.DataFrame:
    return np.log(closes / closes.shift(1)).dropna()


def black_litterman(rets: pd.DataFrame, price: pd.Series, targets: pd.DataFrame, cfg: dict) -> dict:
    tickers = list(rets.columns)
    n = len(tickers)
    S = rets.tail(cfg["cov_lookback_days"]).cov().to_numpy()
    tau, delta = cfg["tau"], cfg["risk_aversion"]

    shares = targets["shares_outstanding"].reindex(tickers).astype(float)
    mcap = (shares * price.reindex(tickers)).to_numpy()
    if np.isnan(mcap).any():  # no share count -> fall back to equal weight for the prior
        w_mkt = np.repeat(1 / n, n)
    else:
        w_mkt = mcap / mcap.sum()
    pi = delta * S @ w_mkt  # implied equilibrium daily returns

    # Views: one absolute view per covered name.
    t = targets.reindex(tickers)
    covered = [i for i, tk in enumerate(tickers) if pd.notna(t.loc[tk, "target_mean"]) and t.loc[tk, "n_analysts"] > 0]
    Q_all = np.log(t["target_mean"].to_numpy(dtype=float) / price.reindex(tickers).to_numpy()) / cfg["target_horizon_days"]

    ts_inv = np.linalg.pinv(tau * S)
    if covered:
        P = np.eye(n)[covered]
        Q = Q_all[covered]
        omega = tau * P @ S @ P.T
        if cfg.get("scale_confidence_by_dispersion", True):
            disp = ((t["target_high"] - t["target_low"]) / t["target_mean"]).to_numpy(dtype=float)[covered]
            med = np.nanmedian(disp) if np.isfinite(disp).any() else 1.0
            rel = np.clip(np.nan_to_num(disp / med, nan=1.0), 0.5, 3.0)
            D = np.diag(rel)
            omega = D @ omega @ D
        om_inv = np.linalg.pinv(omega)
        M = np.linalg.pinv(ts_inv + P.T @ om_inv @ P)
        mu = M @ (ts_inv @ pi + P.T @ om_inv @ Q)
    else:
        mu = pi

    w_raw = np.linalg.pinv(delta * S) @ mu
    gross = np.abs(w_raw).sum()
    w_raw = w_raw / gross if gross > 0 else np.repeat(1 / n, n)  # scale to 100% gross

    return {
        "tickers": tickers,
        "w_mkt": pd.Series(w_mkt, tickers),
        "pi": pd.Series(pi, tickers),
        "view": pd.Series(Q_all, tickers),
        "mu": pd.Series(mu, tickers),
        "w_raw": pd.Series(w_raw, tickers),
    }


def apply_caps(w: pd.Series, max_short: float, max_long: float, min_net: float | None = None) -> pd.Series:
    """Clip each name to [-max_short, +max_long]. Whatever is clipped off stays in cash.

    If min_net is set (e.g. -0.2) and the book is still more net short than that,
    all shorts are scaled down together until net exposure equals min_net.
    """
    w = w.clip(lower=-abs(max_short), upper=abs(max_long))
    if min_net is not None and w.sum() < min_net:
        shorts = w[w < 0].sum()
        need = min_net - w[w > 0].sum()  # the short total we can afford
        if shorts < 0:
            w = w.where(w >= 0, w * max(need, shorts) / shorts if need < 0 else 0.0)
    return w


def trailing_sharpe(rets: pd.DataFrame, w: pd.Series, window: int) -> float:
    """Annualized Sharpe of holding weights `w` (rest in cash at 0%) over the last `window` days."""
    r = rets.tail(window).to_numpy() @ w.reindex(rets.columns).fillna(0).to_numpy()
    sd = r.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return 0.0
    return float(r.mean() / sd * np.sqrt(252))


def sharpe_gate(rets: pd.DataFrame, candidate: pd.Series, book: pd.Series, window: int) -> dict:
    sc = trailing_sharpe(rets, candidate, window)
    sb = trailing_sharpe(rets, book, window)
    return {"candidate_sharpe": sc, "book_sharpe": sb, "passed": sc > sb}
