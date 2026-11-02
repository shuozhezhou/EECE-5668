# EECE 5668 project code

Evaluating whether LLM agents ground their answers in live stock-market data.
Everything runs against a frozen snapshot, so results are reproducible.

## Layout

| Path | Contents |
|---|---|
| `data/` | Snapshot collection (`collect_snapshot.py`), sanity checks, and `snapshots/<date>/` |
| `agent/market.py` | `SnapshotMarket`: read-only access to one snapshot; "today" is its last trading day |
| `agent/tools.py` | `ToolBox`: the tools the LLM can call, their JSON schemas, and a call log |
| `questions/generate.py` | Template-based question generator with programmatic ground truth |
| `questions/questions_v1.jsonl` | The generated benchmark (200 questions) and `questions_v1.meta.json` |
| `tests/check_questions.py` | Re-derives every gold answer from the raw tables, independent of `agent/` |

## Quick start (run from this directory)

```bash
py -m questions.generate          # Windows; use `python` on Colab/Linux
py tests/check_questions.py
```

## Tools

| Tool | Arguments | Returns |
|---|---|---|
| `get_quote` | `ticker` | Latest session OHLCV plus change versus the previous close |
| `get_price_history` | `ticker`, `start_date`, `end_date` | Daily OHLCV rows in the range (trading days only) |
| `get_company_info` | `ticker` | GICS sector and sub-industry; for the 100 largest companies, also market cap, P/E, dividend yield and 52-week range |
| `python_exec` | `code` | stdout of a short Python program (condition C2 only) |

Errors come back as `{"error": ...}` so the agent can react, for example by abstaining on an unknown ticker.
Prices are rounded to cents when the snapshot is loaded. The tools, the gold answers and the grounding metric
therefore all see exactly the same numbers.

## Benchmark (`questions_v1`)

- **Snapshot**: as of Friday 2026-10-30 after the close. Questions only use dates from 2026-08-03 onward.
- **Excluded**: 22 tickers, namely those with split or spin-off adjustments, recent listings, and duplicate share classes.
- **Seed**: 5668. Regenerating gives a byte-identical file.

| Tier | Templates | Answer type |
|---|---|---|
| T1 Lookup (50) | close, intraday high/low, volume, latest close, market cap, trailing P/E | number |
| T2 Comparison (50) | best/worst return among 3–4 peers, higher close, highest volume, close difference, return gap | ticker or number |
| T3 Computation (50) | % change, N-day average close, max close in a range, date of the lowest close, total volume | number or date |
| T5 Unanswerable (50) | future dates, fake tickers, weekends and holidays, investment advice | abstain (`UNANSWERABLE`) |

Each question stores:
- `gold`, `unit` and `tolerance`. A numeric answer is correct iff `|pred - gold| <= max(abs, rel*|gold|)`.
- `required_data`: the (ticker, dates) the answer depends on. This is used for tool-call accuracy.
- `gold_calls`: one canonical tool trace that produces the answer.
