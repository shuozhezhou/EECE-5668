# EECE 5668 Course Project

**Evaluating LLM agents that answer questions with live stock-market data.**

LLM agents can call tools to fetch live data. Even so, they can still answer from stale
parametric memory, misuse the values they retrieve, or miscalculate. This project builds a
tool-using question-answering agent over S&P 500 market data and measures one thing:
*when the agent states a number, did that number actually come from the data?*

The key design choices:
- **Frozen snapshots.** A snapshot of market data makes every experiment reproducible.
- **Generated questions.** A template-based benchmark comes with programmatically
  computed gold answers.
- **Four agent conditions:**
  - C0: closed-book
  - C1: data tools
  - C2: data tools + Python
  - C3: + self-verification
- **Number-level grounding metric.** Every number in the final answer is traced back to
  a tool output.

Team: Shuozhe Zhou, Yifan Zhu, Weibo Jiang (Northeastern University).

## Repository layout

| Path | Contents |
|---|---|
| [`code/data/`](code/data) | Snapshot collection and sanity checks (yfinance) |
| [`code/agent/`](code/agent) | Snapshot market access and the tools exposed to the LLM |
| [`code/questions/`](code/questions) | Question generator and the generated benchmark |
| [`code/tests/`](code/tests) | Independent re-derivation of every gold answer |

See [`code/README.md`](code/README.md) for details.

## Data

The market-data snapshots come from Yahoo Finance via `yfinance` and are **not** committed.
You can rebuild one yourself, or use the team's shared Google Drive copy
(`MyDrive/EECE5668/`).

```bash
cd code
python data/collect_snapshot.py      # writes data/snapshots/<today>/
python -m questions.generate         # regenerates questions/questions_v1.jsonl
python tests/check_questions.py
```

The benchmark in this repo (`questions_v1.jsonl`) was generated from the
2026-11-02 snapshot, which runs through the 2026-10-30 session.
