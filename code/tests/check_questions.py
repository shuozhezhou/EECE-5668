"""Independently re-derive every gold answer from the raw snapshot tables.

The generator computes gold answers through the agent's ToolBox. This script does not
import the agent package at all: it reads the parquet/CSV files with pandas and
recomputes each answer from scratch, so a bug in the tool layer or the generator
shows up as a mismatch here. It also checks the structural invariants of T5.

Usage (from the `code/` directory):
    py tests/check_questions.py [--questions questions/questions_v1.jsonl] [--snapshot-dir ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]


def load(snapshot_dir: Path):
    prices = pd.read_parquet(snapshot_dir / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"]).dt.strftime("%Y-%m-%d")
    for col in ("open", "high", "low", "close"):
        prices[col] = prices[col].round(2)
    px = prices.set_index(["ticker", "date"]).sort_index()
    info = pd.read_parquet(snapshot_dir / "company_info.parquet").set_index("ticker")
    calendar = set(pd.read_csv(snapshot_dir / "trading_calendar.csv")["date"])
    meta = json.loads((snapshot_dir / "metadata.json").read_text(encoding="utf-8"))
    return px, info, calendar, meta


def recompute(q: dict, px: pd.DataFrame, info: pd.DataFrame, as_of: str):
    """Return the gold answer for q computed directly from the tables."""
    tpl, t = q["template"], q["meta"]["tickers"]
    need = q["required_data"]

    def close(tk, d):
        return float(px.loc[(tk, d), "close"])

    def ret(tk, d1, d2):
        return (close(tk, d2) / close(tk, d1) - 1) * 100

    dates = need[0].get("dates", []) if need else []
    if tpl == "t1_close":
        return close(t[0], dates[0])
    if tpl == "t1_high_low":
        return float(px.loc[(t[0], dates[0]), q["meta"]["field"]])
    if tpl == "t1_volume":
        return int(px.loc[(t[0], dates[0]), "volume"])
    if tpl == "t1_latest_close":
        assert dates == [as_of], "latest close must refer to the as-of session"
        return close(t[0], as_of)
    if tpl == "t1_market_cap":
        return int(info.loc[t[0], "marketCap"])
    if tpl == "t1_pe":
        return round(float(info.loc[t[0], "trailingPE"]), 2)
    if tpl in ("t2_best_return",):
        d1, d2 = dates
        rets = {tk: ret(tk, d1, d2) for tk in t}
        pick = max if q["meta"]["direction"] == "best" else min
        return pick(rets, key=rets.get)
    if tpl == "t2_higher_close":
        return max(t, key=lambda tk: close(tk, dates[0]))
    if tpl == "t2_most_volume":
        return max(t, key=lambda tk: int(px.loc[(tk, dates[0]), "volume"]))
    if tpl == "t2_close_diff":
        return round(close(t[0], dates[0]) - close(t[1], dates[0]), 2)
    if tpl == "t2_return_gap":
        d1, d2 = dates
        return round(ret(t[0], d1, d2) - ret(t[1], d1, d2), 2)
    if tpl == "t3_pct_change":
        return round(ret(t[0], dates[0], dates[1]), 2)
    if tpl == "t3_avg_close":
        assert len(dates) == q["meta"]["n_days"]
        return round(sum(close(t[0], d) for d in dates) / len(dates), 4)
    if tpl == "t3_max_close":
        return max(close(t[0], d) for d in dates)
    if tpl == "t3_min_close_date":
        return min(dates, key=lambda d: close(t[0], d))
    if tpl == "t3_total_volume":
        return int(sum(int(px.loc[(t[0], d), "volume"]) for d in dates))
    raise KeyError(tpl)


def check_span(q: dict, calendar: set[str]) -> None:
    """Range questions must list exactly the trading days in their range."""
    if q["template"] in ("t3_max_close", "t3_min_close_date", "t3_total_volume"):
        dates = q["required_data"][0]["dates"]
        lo, hi = dates[0], dates[-1]
        expected = sorted(d for d in calendar if lo <= d <= hi)
        assert dates == expected, f"{q['id']}: required dates are not the full trading range"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(CODE_DIR / "questions" / "questions_v1.jsonl"))
    parser.add_argument("--snapshot-dir", default=str(CODE_DIR / "data" / "snapshots" / "2026-11-02"))
    args = parser.parse_args()

    px, info, calendar, meta = load(Path(args.snapshot_dir))
    as_of = meta["last_trading_date"]
    questions = [json.loads(line) for line in open(args.questions, encoding="utf-8")]

    failures: list[str] = []
    for q in questions:
        try:
            if q["answer_type"] == "abstain":
                reason = q["meta"]["reason"]
                if reason == "future_date":
                    assert q["meta"]["date"] > as_of
                elif reason == "unknown_ticker":
                    assert q["meta"]["tickers"][0] not in px.index.get_level_values(0)
                elif reason == "market_closed":
                    assert q["meta"]["date"] not in calendar
                assert q["gold"] == "UNANSWERABLE"
                continue
            for need in q["required_data"]:
                for d in need.get("dates", []):
                    assert d in calendar and d <= as_of, f"{d} is not a trading day in the snapshot"
            check_span(q, calendar)
            expected = recompute(q, px, info, as_of)
            gold = q["gold"]
            if isinstance(expected, float):
                ok = abs(expected - gold) <= 1e-6 * max(1.0, abs(expected))
            else:
                ok = expected == gold
            if not ok:
                failures.append(f"{q['id']}: gold={gold!r} recomputed={expected!r}")
        except Exception as exc:  # noqa: BLE001 - report every broken question
            failures.append(f"{q['id']}: {type(exc).__name__}: {exc}")

    texts = Counter(q["question"] for q in questions)
    dupes = [t for t, n in texts.items() if n > 1]
    ids = Counter(q["id"] for q in questions)
    dup_ids = [i for i, n in ids.items() if n > 1]

    print(f"Checked {len(questions)} questions against {Path(args.snapshot_dir).name} (as of {as_of})")
    print("By tier:", dict(Counter(q["tier"] for q in questions)))
    print("Answer types:", dict(Counter(q["answer_type"] for q in questions)))
    print("Duplicate questions:", len(dupes), "| duplicate ids:", len(dup_ids))
    if failures:
        print(f"FAILED: {len(failures)}")
        for f in failures:
            print("  ", f)
        return 1
    if dupes or dup_ids:
        return 1
    print("All gold answers re-derived independently: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
