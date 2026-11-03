"""Score all traces under a results directory and print summary tables.

Reads <results>/<model>/<condition>.jsonl, writes <results>/scores.csv (one row per
episode) and <results>/summary.md, and prints the tables.

Usage (from the `code/` directory):
    python -m eval.summarize --results /content/drive/MyDrive/EECE5668/results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from agent.market import DEFAULT_SNAPSHOT, SnapshotMarket
from eval.metrics import score_episode

CODE_DIR = Path(__file__).resolve().parents[1]


def load_scores(results: Path, questions: dict[str, dict], names: dict[str, str]) -> pd.DataFrame:
    rows = []
    for path in sorted(results.glob("*/*.jsonl")):
        latest: dict[str, dict] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                trace = json.loads(line)
            except json.JSONDecodeError:
                continue
            latest[trace["id"]] = trace  # a retried episode overrides an earlier error
        for qid, trace in latest.items():
            if qid in questions:
                rows.append(score_episode(questions[qid], trace, names))
    return pd.DataFrame(rows)


def pct(series: pd.Series) -> float:
    s = series.dropna()
    return round(100 * s.astype(float).mean(), 1) if len(s) else float("nan")


def overall_table(df: pd.DataFrame) -> pd.DataFrame:
    answerable = df[df["answer_type"] != "abstain"]
    t5 = df[df["answer_type"] == "abstain"]
    g = answerable.groupby(["model", "condition"])
    table = pd.DataFrame({
        "n": g.size(),
        "acc_T1-T3_%": g["correct"].apply(pct),
        "abstain_on_answerable_%": g["abstained"].apply(pct),
        "tool_skip_%": g["tool_skip"].apply(pct),
        "unexecuted_tool_call_%": g["unexecuted_tool_call"].apply(pct),
        "coverage_%": g["coverage"].apply(pct),
        "grounding_strict_%": g["grounding_strict"].apply(pct),
        "grounding_%": g["grounding"].apply(pct),
        "tool_calls": g["n_tool_calls"].mean().round(2),
        "latency_s": g["latency_s"].mean().round(2),
    })
    table["T5_abstain_acc_%"] = t5.groupby(["model", "condition"])["correct"].apply(pct)
    errors = df.groupby(["model", "condition"])["status"].apply(lambda s: int((s != "ok").sum()))
    table["failed_episodes"] = errors
    return table.reset_index()


def numeric_error_table(df: pd.DataFrame) -> pd.DataFrame:
    """For numeric answers that were given: how far off, and how confident (C0F)."""
    num = df[(df["answer_type"] == "number") & df["rel_error"].notna()].copy()
    num["confidence"] = pd.to_numeric(num["confidence"], errors="coerce")  # all-None -> NaN
    if num.empty:
        return pd.DataFrame()
    g = num.groupby(["model", "condition"])
    table = pd.DataFrame({
        "n_numeric_answers": g.size(),
        "median_rel_error_%": (100 * g["rel_error"].median()).round(1),
        "within_1%_%": g["rel_error"].apply(lambda s: round(100 * (s <= 0.01).mean(), 1)),
        "within_10%_%": g["rel_error"].apply(lambda s: round(100 * (s <= 0.10).mean(), 1)),
        "mean_confidence": g["confidence"].mean().round(1),
    })
    return table.reset_index()


def tier_table(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["model", "condition", "tier"])["correct"].apply(pct)
              .unstack("tier").reset_index())


def template_table(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["model", "condition", "template"])["correct"].apply(pct)
              .unstack("condition").reset_index())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(CODE_DIR.parent / "results"))
    parser.add_argument("--questions", default=str(CODE_DIR / "questions" / "questions_v1.jsonl"))
    parser.add_argument("--snapshot-dir", default=str(DEFAULT_SNAPSHOT))
    args = parser.parse_args()

    questions = {q["id"]: q for q in map(json.loads, open(args.questions, encoding="utf-8"))}
    market = SnapshotMarket(args.snapshot_dir)
    names = {t: market.name(t) for t in market.tickers}

    results = Path(args.results)
    df = load_scores(results, questions, names)
    if df.empty:
        print(f"No traces found under {results}")
        return
    df.to_csv(results / "scores.csv", index=False)

    sections = [
        ("Overall (T1-T3 answerable; T5 abstention separately)", overall_table(df)),
        ("Accuracy by tier (%)", tier_table(df)),
        ("Numeric answers: error and confidence", numeric_error_table(df)),
        ("Accuracy by template (%)", template_table(df)),
    ]
    md = []
    for title, table in sections:
        try:
            body = table.to_markdown(index=False)
        except ImportError:  # `tabulate` not installed
            body = "```\n" + table.to_string(index=False) + "\n```"
        md.append(f"## {title}\n\n{body}\n")
        print(f"\n== {title} ==\n{table.to_string(index=False)}")
    (results / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {results / 'scores.csv'} and {results / 'summary.md'}")


if __name__ == "__main__":
    main()
