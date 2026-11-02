"""Sanity checks for a market-data snapshot produced by collect_snapshot.py.

Usage:
    py check_snapshot.py                       # latest snapshot under ./snapshots
    py check_snapshot.py --snapshot-dir path/to/snapshots/2026-11-02
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "snapshots"


def load_prices(snap: Path) -> pd.DataFrame:
    pq = snap / "prices.parquet"
    df = pd.read_parquet(pq) if pq.exists() else pd.read_csv(snap / "prices.csv.gz")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def section(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 70 - len(title)))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--seed", type=int, default=5668)
    args = p.parse_args(argv)

    snap = args.snapshot_dir or max(d for d in args.out_dir.iterdir() if d.is_dir())
    print(f"Snapshot: {snap}")
    meta = json.loads((snap / "metadata.json").read_text(encoding="utf-8"))
    df = load_prices(snap)
    pd.set_option("display.width", 160, "display.max_columns", 20)

    section("Shape / range")
    print("shape:", df.shape)
    print("date range:", df["date"].min(), "->", df["date"].max())
    print("tickers:", df["ticker"].nunique(), "| trading days:", df["date"].nunique())
    print("metadata last_trading_date:", meta["last_trading_date"])
    print("duplicate (ticker,date):", int(df.duplicated(["ticker", "date"]).sum()))
    print("failed:", meta["tickers"]["failed_list"])
    print("flagged short:", [(f["ticker"], f["rows"], f["first_date"]) for f in meta["flagged_short"]["tickers"]])

    section("NaN counts per column")
    print(df.isna().sum().to_string())

    section("Price anomalies")
    px = ["open", "high", "low", "close", "adj_close"]
    nonpos = df[(df[px] <= 0).any(axis=1)]
    print("rows with zero/negative price:", len(nonpos))
    if len(nonpos):
        print(nonpos.head(10).to_string(index=False))
    hl = df[df["high"] < df["low"]]
    print("rows with high < low:", len(hl))
    if len(hl):
        print(hl.head(10).to_string(index=False))
    oc_out = df[(df["close"] > df["high"] * 1.0001) | (df["close"] < df["low"] * 0.9999)
                | (df["open"] > df["high"] * 1.0001) | (df["open"] < df["low"] * 0.9999)]
    print("rows with open/close outside [low, high]:", len(oc_out))
    if len(oc_out):
        print(oc_out.head(10).to_string(index=False))
    zero_vol = df[df["volume"] == 0]
    print("rows with zero volume:", len(zero_vol))
    # Large one-day moves in close (could reveal unadjusted splits or bad ticks).
    d = df.sort_values(["ticker", "date"]).copy()
    d["ret"] = d.groupby("ticker")["close"].pct_change()
    big = d[d["ret"].abs() > 0.25]
    print("daily |close return| > 25%:", len(big))
    if len(big):
        print(big[["ticker", "date", "close", "adj_close", "stock_splits", "ret"]].to_string(index=False))

    section("Last 3 rows: AAPL, NVDA, MSFT")
    for t in ("AAPL", "NVDA", "MSFT"):
        print(df[df["ticker"] == t].tail(3).to_string(index=False))

    section("Split events (close vs adj_close around split)")
    splits = df[df["stock_splits"] > 0]
    if splits.empty:
        print("no splits in window")
    for r in splits.itertuples():
        g = df[df["ticker"] == r.ticker].reset_index(drop=True)
        i = g.index[g["date"] == r.date][0]
        print(f"\n{r.ticker} split {r.stock_splits} on {r.date}")
        print(g.iloc[max(0, i - 2): i + 3][["date", "open", "close", "adj_close", "volume", "stock_splits"]]
              .to_string(index=False))

    section("Dividends")
    divs = df[df["dividends"] > 0]
    print("dividend events:", len(divs), "across", divs["ticker"].nunique(), "tickers")

    section("5 random (ticker, date, close) for manual cross-check")
    print(df.sample(5, random_state=args.seed)[["ticker", "date", "close"]].to_string(index=False))

    info_pq = snap / "company_info.parquet"
    if info_pq.exists():
        section("Company info")
        info = pd.read_parquet(info_pq)
        print("shape:", info.shape)
        print(info.isna().sum().to_string())
        print(info.head(10)[["ticker", "market_cap_rank", "longName", "marketCap", "trailingPE",
                             "dividendYield", "fetched_at_utc"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
