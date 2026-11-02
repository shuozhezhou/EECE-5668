"""Collect a frozen, reproducible snapshot of S&P 500 market data.

The snapshot is the single source of truth for the agent's replay tools and
for code-computed ground-truth answers. It contains:

    constituents.csv        S&P 500 universe (Wikipedia, Yahoo-format symbols)
    prices.parquet/.csv.gz  daily OHLCV + adj_close + dividends + splits (tidy)
    company_info.parquet/.csv  point-in-time yfinance .info for ~100 largest names
    trading_calendar.csv    trading dates present in the window
    metadata.json           provenance, versions, failures, field semantics

The script is idempotent and resumable: every download batch is written to
``<snapshot>/_partial/`` and skipped on re-run, so a rate-limit failure does
not force a full restart. Use ``--force`` to ignore partial results.

Usage (local or Colab):
    py collect_snapshot.py                       # defaults
    python collect_snapshot.py --out-dir /content/snapshots --batch-size 50
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

NY = ZoneInfo("America/New_York")
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "snapshots"
DEFAULT_START = "2025-11-01"

# yfinance column name -> output column name
PRICE_FIELDS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
    "Dividends": "dividends",
    "Stock Splits": "stock_splits",
}
PRICE_COLUMNS = ["date", "ticker"] + list(PRICE_FIELDS.values())

INFO_FIELDS = [
    "longName", "sector", "industry", "marketCap", "trailingPE", "forwardPE",
    "dividendYield", "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "currency", "exchange",
]

# Fallback universe if Wikipedia is unreachable: ~100 of the largest S&P 500
# names (Yahoo symbols). Sector fields are left empty in that case.
FALLBACK_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "BRK-B", "AVGO", "TSLA",
    "LLY", "JPM", "V", "UNH", "XOM", "MA", "JNJ", "PG", "HD", "COST",
    "ORCL", "ABBV", "MRK", "CVX", "WMT", "KO", "BAC", "PEP", "NFLX", "CRM",
    "AMD", "ADBE", "TMO", "LIN", "ACN", "MCD", "CSCO", "ABT", "WFC", "DHR",
    "INTU", "TXN", "PM", "DIS", "QCOM", "VZ", "CAT", "IBM", "AMGN", "GE",
    "NOW", "PFE", "CMCSA", "UNP", "SPGI", "ISRG", "NEE", "LOW", "AMAT", "RTX",
    "HON", "GS", "UBER", "T", "BKNG", "PGR", "ELV", "SYK", "BLK", "AXP",
    "LMT", "TJX", "MS", "C", "VRTX", "SCHW", "COP", "MDT", "PLD", "BSX",
    "ADI", "CB", "REGN", "MMC", "LRCX", "ADP", "MU", "PANW", "KLAC", "SBUX",
    "BMY", "DE", "GILD", "ETN", "ANET", "MDLZ", "CI", "SO", "BA", "PLTR",
]

log = logging.getLogger("snapshot")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def to_yahoo_symbol(symbol: str) -> str:
    """Wikipedia uses class-share dots (BRK.B); Yahoo uses dashes (BRK-B)."""
    return symbol.strip().upper().replace(".", "-")


def with_retries(fn, *, what: str, max_retries: int, base_sleep: float):
    """Call ``fn`` with exponential backoff (plus jitter). Re-raises at the end."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:  # yfinance raises a variety of exception types
            if attempt == max_retries:
                raise
            wait = base_sleep * (2 ** attempt) + random.uniform(0, 1)
            log.warning("%s failed (%s: %s); retry %d/%d in %.1fs",
                        what, type(exc).__name__, exc, attempt + 1, max_retries, wait)
            time.sleep(wait)


def default_end_date(snapshot_date: date, include_snapshot_day: bool) -> date:
    """Exclusive end date for yfinance.

    By default the snapshot date's own session is EXCLUDED, even after the
    16:00 ET close: Yahoo's same-day daily bar is built from the real-time feed
    and is only finalised overnight (observed on 2026-11-02: open/close outside
    [low, high] for 22 tickers and non-final consolidated volumes). The last
    included session is therefore the most recent *settled* one.
    ``--include-snapshot-day`` includes it once the session has closed.
    """
    now_ny = datetime.now(NY)
    closed = now_ny.date() > snapshot_date or now_ny.hour * 60 + now_ny.minute >= 16 * 60 + 30
    if include_snapshot_day and closed:
        return snapshot_date + timedelta(days=1)
    return snapshot_date


def write_table(df: pd.DataFrame, base: Path, csv_suffix: str = ".csv") -> list[str]:
    """Write ``df`` to ``base.parquet`` and ``base<csv_suffix>``; return file names."""
    written = []
    try:
        df.to_parquet(base.with_suffix(".parquet"), index=False)
        written.append(base.with_suffix(".parquet").name)
    except Exception as exc:  # pyarrow missing etc. -- CSV still gets written
        log.error("Parquet write failed for %s: %s", base.name, exc)
    csv_path = base.parent / (base.name + csv_suffix)
    df.to_csv(csv_path, index=False)
    written.append(csv_path.name)
    return written


# --------------------------------------------------------------------------- #
# 1. Universe
# --------------------------------------------------------------------------- #
def fetch_constituents() -> tuple[pd.DataFrame, str]:
    """Return (constituents, source). Falls back to a hard-coded list."""
    try:
        resp = requests.get(WIKI_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        table = next(t for t in tables if "Symbol" in t.columns and "GICS Sector" in t.columns)
        df = table.rename(columns={
            "Symbol": "wiki_symbol",
            "Security": "security",
            "GICS Sector": "gics_sector",
            "GICS Sub-Industry": "gics_sub_industry",
        })[["wiki_symbol", "security", "gics_sector", "gics_sub_industry"]]
        df["ticker"] = df["wiki_symbol"].map(to_yahoo_symbol)
        df = df.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)
        log.info("Wikipedia: %d constituents", len(df))
        return df[["ticker", "wiki_symbol", "security", "gics_sector", "gics_sub_industry"]], "wikipedia"
    except Exception as exc:
        log.error("Wikipedia fetch failed (%s); using hard-coded fallback list", exc)
        df = pd.DataFrame({"ticker": sorted(set(FALLBACK_TICKERS))})
        df["wiki_symbol"] = df["ticker"].str.replace("-", ".", regex=False)
        for col in ("security", "gics_sector", "gics_sub_industry"):
            df[col] = pd.NA
        return df, "fallback_hardcoded_top100"


# --------------------------------------------------------------------------- #
# 2. Prices
# --------------------------------------------------------------------------- #
def tidy_download(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Convert a yf.download(group_by='ticker') frame into long/tidy rows."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    if not isinstance(raw.columns, pd.MultiIndex):
        # Single-ticker downloads may come back flat; normalise to MultiIndex.
        raw = pd.concat({tickers[0]: raw}, axis=1)

    frames = []
    present = set(raw.columns.get_level_values(0))
    for t in tickers:
        if t not in present:
            continue
        sub = raw[t].copy()
        sub = sub.rename(columns=PRICE_FIELDS)
        for col in PRICE_FIELDS.values():
            if col not in sub.columns:
                sub[col] = pd.NA
        sub = sub[list(PRICE_FIELDS.values())]
        sub = sub.dropna(subset=["open", "high", "low", "close"], how="all")
        if sub.empty:
            continue
        idx = pd.DatetimeIndex(sub.index)
        if idx.tz is not None:
            idx = idx.tz_convert(NY).tz_localize(None)
        sub.insert(0, "date", idx.normalize().date)
        sub.insert(1, "ticker", t)
        frames.append(sub.reset_index(drop=True))
    if not frames:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def download_batch(tickers: list[str], start: str, end: str, args) -> pd.DataFrame:
    """Download one batch; raise if *nothing* came back (likely rate-limited)."""
    def _call():
        raw = yf.download(
            tickers, start=start, end=end, interval="1d",
            auto_adjust=False, actions=True, group_by="ticker",
            threads=True, progress=False, timeout=30,
        )
        tidy = tidy_download(raw, tickers)
        if tidy.empty:
            raise RuntimeError("empty result for whole batch")
        return tidy
    return with_retries(_call, what=f"download({tickers[0]}..{tickers[-1]})",
                        max_retries=args.max_retries, base_sleep=args.retry_sleep)


def collect_prices(tickers: list[str], start: str, end: str, partial: Path, args) -> pd.DataFrame:
    """Batched, resumable price download. Each batch is cached as parquet/pickle."""
    batch_dir = partial / "prices"
    batch_dir.mkdir(parents=True, exist_ok=True)
    batches = [tickers[i:i + args.batch_size] for i in range(0, len(tickers), args.batch_size)]

    def batch_path(name):
        return batch_dir / f"{name}.pkl"  # pickle: no extra deps, preserves dtypes

    def run(name, batch):
        path = batch_path(name)
        if path.exists():
            log.info("[%s] cached (%d tickers) -- skipping", name, len(batch))
            return pd.read_pickle(path)
        log.info("[%s] downloading %d tickers (%s..%s)", name, len(batch), batch[0], batch[-1])
        try:
            df = download_batch(batch, start, end, args)
        except Exception as exc:
            log.error("[%s] gave up: %s", name, exc)
            return None  # not cached -> retried on the next run
        df.to_pickle(path)
        time.sleep(args.sleep)
        return df

    frames = []
    for i, batch in enumerate(batches):
        df = run(f"batch_{i:03d}", batch)
        if df is not None:
            frames.append(df)

    # Second pass: tickers missing from otherwise-successful batches, in small batches.
    got = set(pd.concat(frames)["ticker"]) if frames else set()
    missing = [t for t in tickers if t not in got]
    if missing:
        log.info("Retry pass for %d missing tickers: %s", len(missing), missing)
        size = max(1, min(10, args.batch_size))
        for j in range(0, len(missing), size):
            sub = missing[j:j + size]
            df = run(f"retry_{'_'.join(sub)[:60]}_{len(sub)}", sub)
            if df is not None:
                frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PRICE_COLUMNS)


def clean_prices(prices: pd.DataFrame, last_allowed: date) -> pd.DataFrame:
    """Type-cast, drop empty/future rows, de-duplicate, sort."""
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[df["date"] < last_allowed]  # never include a partial/unsettled session
    for col in ("open", "high", "low", "close", "adj_close", "dividends", "stock_splits"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df = df.dropna(subset=["open", "high", "low", "close"], how="all")
    df["dividends"] = df["dividends"].fillna(0.0)
    df["stock_splits"] = df["stock_splits"].fillna(0.0)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").round().astype("Int64")
    df = (df.sort_values(["ticker", "date"])
            .drop_duplicates(["ticker", "date"], keep="last")
            .reset_index(drop=True))
    return df[PRICE_COLUMNS]


# --------------------------------------------------------------------------- #
# 3. Company info
# --------------------------------------------------------------------------- #
def rank_candidates(prices: pd.DataFrame, n: int) -> list[str]:
    """Pre-rank by median dollar volume over the last ~60 sessions.

    Dollar volume is a cheap proxy for size that needs no extra requests; the
    final top-N is chosen by the marketCap returned by .info itself.
    """
    recent_dates = sorted(prices["date"].unique())[-60:]
    recent = prices[prices["date"].isin(recent_dates)]
    dv = (recent["close"] * recent["volume"].astype("float64")).groupby(recent["ticker"]).median()
    return dv.sort_values(ascending=False).head(n).index.tolist()


def collect_info(candidates: list[str], partial: Path, args) -> pd.DataFrame:
    info_dir = partial / "info"
    info_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, t in enumerate(candidates):
        path = info_dir / f"{t}.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
            continue
        try:
            info = with_retries(lambda: yf.Ticker(t).info, what=f"info({t})",
                                max_retries=args.max_retries, base_sleep=args.retry_sleep)
        except Exception as exc:
            log.error("info(%s) gave up: %s", t, exc)
            continue
        row = {"ticker": t, **{k: info.get(k) for k in INFO_FIELDS},
               "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        path.write_text(json.dumps(row), encoding="utf-8")
        rows.append(row)
        if (i + 1) % 25 == 0:
            log.info("info: %d/%d", i + 1, len(candidates))
        time.sleep(args.info_sleep)
    return pd.DataFrame(rows, columns=["ticker"] + INFO_FIELDS + ["fetched_at_utc"])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help="Parent folder for snapshots (default: ./snapshots next to this script)")
    p.add_argument("--snapshot-date", default=None,
                   help="Snapshot label YYYY-MM-DD (default: today in America/New_York)")
    p.add_argument("--start", default=DEFAULT_START, help="First date (inclusive)")
    p.add_argument("--end", default=None,
                   help="Exclusive end date. Default: the snapshot date (its own, not-yet-settled session is excluded)")
    p.add_argument("--include-snapshot-day", action="store_true",
                   help="Include the snapshot date's session if it has closed (bar may be preliminary)")
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--sleep", type=float, default=3.0, help="Pause between price batches (s)")
    p.add_argument("--info-sleep", type=float, default=0.75, help="Pause between .info calls (s)")
    p.add_argument("--retry-sleep", type=float, default=10.0, help="Base backoff (s), doubled per retry")
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--info-top-n", type=int, default=100, help="Number of companies in company_info")
    p.add_argument("--info-candidates", type=int, default=150,
                   help="Candidates (by dollar volume) whose .info is fetched before ranking by marketCap")
    p.add_argument("--short-threshold", type=float, default=0.8,
                   help="Flag tickers with fewer than this fraction of trading days")
    p.add_argument("--skip-info", action="store_true")
    p.add_argument("--force", action="store_true", help="Ignore cached partial results")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout)
    t0 = time.time()
    snapshot_date = (date.fromisoformat(args.snapshot_date) if args.snapshot_date
                     else datetime.now(NY).date())
    end = (date.fromisoformat(args.end) if args.end
           else default_end_date(snapshot_date, args.include_snapshot_day))
    snap_dir = Path(args.out_dir) / snapshot_date.isoformat()
    partial = snap_dir / "_partial"
    if args.force and partial.exists():
        import shutil
        shutil.rmtree(partial)
    partial.mkdir(parents=True, exist_ok=True)
    log.info("Snapshot %s -> %s | window [%s, %s)", snapshot_date, snap_dir, args.start, end)
    timings = {}

    # 1. Universe (cached so batch composition is stable across resumes)
    cons_cache = partial / "constituents.csv"
    if cons_cache.exists():
        constituents = pd.read_csv(cons_cache)
        source = (partial / "constituents_source.txt").read_text().strip()
        log.info("Constituents loaded from cache (%d, source=%s)", len(constituents), source)
    else:
        constituents, source = fetch_constituents()
        constituents.to_csv(cons_cache, index=False)
        (partial / "constituents_source.txt").write_text(source)
    constituents.to_csv(snap_dir / "constituents.csv", index=False)
    tickers = constituents["ticker"].tolist()
    timings["constituents_s"] = round(time.time() - t0, 1)

    # 2. Prices
    t1 = time.time()
    raw_prices = collect_prices(tickers, args.start, end.isoformat(), partial, args)
    prices = clean_prices(raw_prices, last_allowed=end)
    timings["prices_s"] = round(time.time() - t1, 1)

    succeeded = sorted(prices["ticker"].unique())
    failed = sorted(set(tickers) - set(succeeded))

    # Trading calendar: every date that appears, with ticker coverage for auditing.
    cov = prices.groupby("date")["ticker"].nunique()
    calendar = pd.DataFrame({"date": cov.index, "n_tickers": cov.values}).sort_values("date")
    low_cov_dates = calendar.loc[calendar["n_tickers"] < 0.5 * len(succeeded), "date"].astype(str).tolist()
    n_days = len(calendar)

    rows_per = prices.groupby("ticker").size()
    short = rows_per[rows_per < args.short_threshold * n_days].sort_values()
    flagged_short = [{"ticker": t, "rows": int(n), "first_date": str(prices.loc[prices.ticker == t, "date"].min())}
                     for t, n in short.items()]

    splits = prices.loc[prices["stock_splits"] > 0, ["ticker", "date", "stock_splits"]]
    split_events = [{"ticker": r.ticker, "date": str(r.date), "ratio": float(r.stock_splits)}
                    for r in splits.itertuples()]

    # Data-quality counters (rows are kept; these are for auditing only).
    eps = 1e-4
    ohlc_bad = prices[(prices["high"] < prices["low"])
                      | (prices[["open", "close"]].max(axis=1) > prices["high"] * (1 + eps))
                      | (prices[["open", "close"]].min(axis=1) < prices["low"] * (1 - eps))]
    quality = {
        "ohlc_inconsistent_rows": int(len(ohlc_bad)),
        "ohlc_inconsistent_examples": [f"{r.ticker} {r.date}" for r in ohlc_bad.head(20).itertuples()],
        "nonpositive_price_rows": int((prices[["open", "high", "low", "close", "adj_close"]] <= 0).any(axis=1).sum()),
    }
    log.info("Prices: %d rows, %d tickers OK, %d failed, %d short, %d trading days",
             len(prices), len(succeeded), len(failed), len(flagged_short), n_days)

    # 3. Company info
    info_meta = {"skipped": True}
    if not args.skip_info and not prices.empty:
        t2 = time.time()
        candidates = rank_candidates(prices, args.info_candidates)
        info_all = collect_info(candidates, partial, args)
        info = (info_all.sort_values("marketCap", ascending=False, na_position="last")
                        .head(args.info_top_n).reset_index(drop=True))
        info.insert(1, "market_cap_rank", range(1, len(info) + 1))
        write_table(info, snap_dir / "company_info")
        fetched = info_all["fetched_at_utc"].dropna()
        info_meta = {
            "skipped": False,
            "n_candidates": len(candidates),
            "n_fetched": int(len(info_all)),
            "n_saved": int(len(info)),
            "info_failed": sorted(set(candidates) - set(info_all["ticker"])),
            "fetch_window_utc": [fetched.min(), fetched.max()] if len(fetched) else None,
            "selection": (f"Top {args.info_candidates} tickers by median dollar volume (last 60 sessions) "
                          f"were queried via yf.Ticker(t).info; the top {args.info_top_n} by returned "
                          "marketCap were kept."),
        }
        timings["info_s"] = round(time.time() - t2, 1)

    # 4. Save
    price_files = write_table(prices, snap_dir / "prices", csv_suffix=".csv.gz")
    calendar.to_csv(snap_dir / "trading_calendar.csv", index=False)
    timings["total_s"] = round(time.time() - t0, 1)

    metadata = {
        "snapshot_date": snapshot_date.isoformat(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": sys.version.split()[0], "pandas": pd.__version__,
                     "yfinance": yf.__version__},
        "universe_source": source,
        "universe_url": WIKI_URL if source == "wikipedia" else None,
        "request_window": {"start_inclusive": args.start, "end_exclusive": end.isoformat()},
        "date_range": {"first": str(calendar["date"].min()) if n_days else None,
                       "last": str(calendar["date"].max()) if n_days else None},
        "last_trading_date": str(calendar["date"].max()) if n_days else None,
        "n_trading_days": n_days,
        "low_coverage_dates": low_cov_dates,
        "n_rows": int(len(prices)),
        "tickers": {"requested": len(tickers), "succeeded": len(succeeded),
                    "failed": len(failed), "failed_list": failed},
        "flagged_short": {"threshold_fraction": args.short_threshold,
                          "min_rows": int(args.short_threshold * n_days), "tickers": flagged_short},
        "split_events": split_events,
        "split_events_note": ("Yahoo encodes spin-offs / special distributions as fractional 'splits' "
                              "(e.g. ratios like 1.067, 1.241, 0.9535). These are price-adjustment factors, "
                              "not share splits; pre-event OHLC is divided by the ratio."),
        "data_quality": quality,
        "company_info": info_meta,
        "files": {"prices": price_files, "company_info": ["company_info.parquet", "company_info.csv"]
                  if not info_meta.get("skipped") else [],
                  "other": ["constituents.csv", "trading_calendar.csv", "metadata.json"]},
        "field_semantics": {
            "date": "Trading session date (America/New_York), no time component.",
            "open/high/low/close": "Split-adjusted, NOT dividend-adjusted (yfinance auto_adjust=False). "
                                   "'close' matches the closing price shown by Yahoo/Google Finance.",
            "adj_close": "Split- AND dividend-adjusted close (use for total-return calculations).",
            "volume": "Shares traded (split-adjusted by Yahoo).",
            "dividends": "Cash dividend per share with ex-date on this row's date; 0 otherwise.",
            "stock_splits": "Split ratio effective on this date (e.g. 4.0 = 4-for-1); 0 otherwise.",
            "adjustment_reference": "Adjustments are relative to the snapshot date; a split/dividend "
                                    "after the snapshot is NOT reflected (snapshot is frozen).",
            "company_info": "Point-in-time values from Yahoo .info at fetched_at_utc (marketCap, PE and "
                            "52-week high/low may therefore reflect a session later than last_trading_date). "
                            "dividendYield is as returned by Yahoo, in PERCENT (e.g. 0.31 = 0.31%).",
        },
        "timings_s": timings,
    }
    (snap_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    log.info("Done in %.1fs -> %s", time.time() - t0, snap_dir)
    if failed:
        log.warning("Failed tickers (%d): %s", len(failed), failed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
