# Market-data snapshot

Frozen, reproducible S&P 500 data that the agent's tools replay from and that
ground-truth answers are computed from. Nothing is fetched live at evaluation time.

## Scripts

| Script | Purpose |
|---|---|
| `collect_snapshot.py` | Downloads the universe, daily prices and company info and writes a snapshot |
| `check_snapshot.py` | Prints sanity checks (shape, NaNs, anomalies, splits, random spot checks) |

```bash
py collect_snapshot.py                         # Windows (use `python` on Colab/Linux)
py collect_snapshot.py --out-dir /content/snapshots --batch-size 50 --skip-info
py check_snapshot.py                           # checks the latest snapshot
```

Options: `--start` (default 2025-11-01), `--end` (exclusive; defaults to the
snapshot date), `--include-snapshot-day`, `--snapshot-date`, `--batch-size`,
`--sleep`, `--retry-sleep`, `--max-retries`, `--info-top-n`, `--skip-info`,
`--force`. Run `--help` for the full list.

**Last session:** by default the snapshot date's own session is excluded even
after the close. Yahoo's same-day daily bar is preliminary until it is settled
overnight (on 2026-11-02 it had open/close outside [low, high] for 22 tickers
and non-final volumes). The last day in the snapshot is the most recent settled session.

**Resumable:** each price batch and each `.info` result is cached under
`snapshots/<date>/_partial/`. If Yahoo rate-limits, rerun the same command and
only the missing pieces are fetched. `--force` discards the cache.

Dependencies: `pandas`, `yfinance`, `requests`, `lxml` (for `read_html`), and
`pyarrow` (for parquet; the CSV copies are written regardless).

## Snapshot layout: `snapshots/<YYYY-MM-DD>/`

| File | Contents |
|---|---|
| `prices.parquet`, `prices.csv.gz` | Tidy table: `date, ticker, open, high, low, close, adj_close, volume, dividends, stock_splits` |
| `company_info.parquet`, `.csv` | ~100 largest names: `ticker, market_cap_rank, longName, sector, industry, marketCap, trailingPE, forwardPE, dividendYield, fiftyTwoWeekHigh, fiftyTwoWeekLow, currency, exchange, fetched_at_utc` |
| `constituents.csv` | S&P 500 list from Wikipedia (`ticker` in Yahoo format, e.g. `BRK-B`) |
| `trading_calendar.csv` | Trading dates in the window, with the number of tickers that have data |
| `metadata.json` | Timestamps, versions, window, failed/short tickers, split events, field semantics |

## Price field semantics (important for ground truth)

- `open/high/low/close` are **split-adjusted only** (yfinance `auto_adjust=False`).
  `close` is the closing price shown on Yahoo/Google Finance.
- `adj_close` is **split + dividend adjusted**; use it for total-return questions.
- `dividends` is the cash amount on its ex-date, `stock_splits` the ratio on its effective date (0 otherwise).
- Yahoo records spin-offs as fractional "splits" (e.g. CMCSA 1.067, FDX 1.241, HON 0.9535):
  they are adjustment factors, not share splits. See `split_events` in `metadata.json`.
- `company_info.dividendYield` is in percent (0.31 = 0.31%).
- Adjustments are as of the snapshot date. Later splits/dividends are not reflected.
- `company_info` values are point-in-time (see `fetched_at_utc`), not historical.
- `date` is the America/New_York trading date.

Tickers with fewer than 80% of the trading days (recent IPOs or spin-offs) are kept but
listed under `flagged_short` in `metadata.json`.
