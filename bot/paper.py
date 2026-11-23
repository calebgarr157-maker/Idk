"""A simple paper-trading ledger: fractional shares, shorts allowed, fills at the day's close."""
import json
from dataclasses import asdict, dataclass, field

import pandas as pd

from .common import STATE

BOOK_PATH = STATE / "book.json"


@dataclass
class Book:
    cash: float
    positions: dict = field(default_factory=dict)  # ticker -> shares (negative = short)
    start_date: str = ""
    start_nav: float = 0.0
    benchmark_start: float = 0.0
    last_date: str = ""

    @classmethod
    def load(cls, start_capital: float) -> "Book":
        if BOOK_PATH.exists():
            return cls(**json.loads(BOOK_PATH.read_text()))
        return cls(cash=start_capital, start_nav=start_capital)

    def save(self) -> None:
        STATE.mkdir(parents=True, exist_ok=True)
        BOOK_PATH.write_text(json.dumps(asdict(self), indent=2) + "\n")

    def nav(self, px: pd.Series) -> float:
        return self.cash + sum(sh * float(px[t]) for t, sh in self.positions.items())

    def weights(self, px: pd.Series, tickers: list[str]) -> pd.Series:
        nav = self.nav(px)
        w = {t: self.positions.get(t, 0.0) * float(px[t]) / nav for t in tickers}
        return pd.Series(w)

    def accrue_borrow(self, px: pd.Series, days: int, bps_annual: float) -> float:
        short_notional = sum(-sh * float(px[t]) for t, sh in self.positions.items() if sh < 0)
        fee = short_notional * bps_annual / 1e4 * days / 365
        self.cash -= fee
        return fee

    def rebalance(self, target: pd.Series, px: pd.Series, cost_bps: float, date: str) -> list[dict]:
        nav = self.nav(px)
        trades = []
        for t, w in target.items():
            want = w * nav / float(px[t])
            have = self.positions.get(t, 0.0)
            d = want - have
            if abs(d * float(px[t])) < 1.0:  # ignore sub-$1 dust
                continue
            notional = d * float(px[t])
            cost = abs(notional) * cost_bps / 1e4
            self.cash -= notional + cost
            self.positions[t] = want
            trades.append({"date": date, "ticker": t, "shares": round(d, 6), "price": round(float(px[t]), 4),
                           "notional": round(notional, 2), "cost": round(cost, 2)})
        self.positions = {t: s for t, s in self.positions.items() if abs(s) > 1e-9}
        return trades
