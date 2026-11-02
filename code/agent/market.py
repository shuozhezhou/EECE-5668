"""Read-only access to a frozen market-data snapshot.

The agent's world is the snapshot: "today" is the last trading day in it, and every
tool answers from these tables only. Prices are rounded to cents at load time so
that the numbers the agent sees, the numbers ground truth is computed from, and the
numbers the grounding metric looks for are exactly the same.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = CODE_DIR / "data" / "snapshots" / "2026-11-02"

PRICE_COLUMNS = ("open", "high", "low", "close")
PRICE_DECIMALS = 2


def normalize_ticker(ticker: str) -> str:
    """Map user/LLM spellings to the snapshot's Yahoo-style symbols (BRK.B -> BRK-B)."""
    return str(ticker).strip().upper().replace(".", "-").replace("/", "-")


class SnapshotMarket:
    """Prices, company info and the trading calendar of one snapshot."""

    def __init__(self, snapshot_dir: str | Path = DEFAULT_SNAPSHOT):
        self.dir = Path(snapshot_dir)
        self.metadata = json.loads((self.dir / "metadata.json").read_text(encoding="utf-8"))

        prices = self._read_table("prices")
        prices["date"] = pd.to_datetime(prices["date"]).dt.date
        for col in PRICE_COLUMNS:
            prices[col] = prices[col].round(PRICE_DECIMALS)
        prices["volume"] = prices["volume"].astype("int64")
        self.prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
        self._by_ticker = {
            t: g.set_index("date", drop=False) for t, g in self.prices.groupby("ticker", sort=False)
        }

        self.constituents = pd.read_csv(self.dir / "constituents.csv").set_index("ticker")
        self.info = self._read_table("company_info").set_index("ticker")

        self.as_of: date = date.fromisoformat(self.metadata["last_trading_date"])
        self.trading_days: list[date] = sorted(self.prices["date"].unique())
        self._trading_day_set = set(self.trading_days)

    def _read_table(self, stem: str) -> pd.DataFrame:
        parquet = self.dir / f"{stem}.parquet"
        if parquet.exists():
            try:
                return pd.read_parquet(parquet)
            except ImportError:  # no pyarrow/fastparquet: fall back to the CSV copy
                pass
        for suffix in (".csv.gz", ".csv"):
            path = self.dir / f"{stem}{suffix}"
            if path.exists():
                return pd.read_csv(path)
        raise FileNotFoundError(f"No {stem} table in {self.dir}")

    # ------------------------------------------------------------------ lookups

    @property
    def tickers(self) -> list[str]:
        return sorted(self._by_ticker)

    @property
    def as_of_text(self) -> str:
        """Human-readable 'current time' given to the agent."""
        return f"{self.as_of:%A}, {self.as_of:%B} {self.as_of.day}, {self.as_of.year}, 6:00 PM ET (after the market close)"

    def has_ticker(self, ticker: str) -> bool:
        return normalize_ticker(ticker) in self._by_ticker

    def is_trading_day(self, d: date) -> bool:
        return d in self._trading_day_set

    def history(self, ticker: str, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        """Daily rows for one ticker with start <= date <= end (both optional)."""
        frame = self._by_ticker[normalize_ticker(ticker)]
        mask = pd.Series(True, index=frame.index)
        if start is not None:
            mask &= frame["date"] >= start
        if end is not None:
            mask &= frame["date"] <= end
        return frame[mask.to_numpy()]

    def row(self, ticker: str, d: date) -> pd.Series | None:
        frame = self._by_ticker[normalize_ticker(ticker)]
        return frame.loc[d] if d in frame.index else None

    def name(self, ticker: str) -> str:
        t = normalize_ticker(ticker)
        if t not in self.constituents.index:
            return t
        name = str(self.constituents.loc[t, "security"])
        # Wikipedia lists some names as "Walt Disney Company (The)".
        if name.endswith(" (The)"):
            name = "The " + name[: -len(" (The)")]
        return name

    def trading_days_between(self, start: date, end: date) -> list[date]:
        return [d for d in self.trading_days if start <= d <= end]

    def trading_days_ending(self, end: date, n: int) -> list[date]:
        """The n trading days up to and including `end`."""
        days = [d for d in self.trading_days if d <= end]
        return days[-n:]
