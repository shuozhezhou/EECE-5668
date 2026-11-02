"""Offline end-to-end test of the agent loop and scoring, using scripted fake LLMs.

No GPU or network needed. Three fake models exercise the scoring paths:
    oracle   calls the canonical tools, then answers with the gold value
    stale    never calls tools and answers with a made-up number
    refuser  always answers UNANSWERABLE

Usage (from the `code/` directory):
    py tests/test_pipeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.answers import grade, parse_date, parse_number  # noqa: E402
from agent.market import SnapshotMarket  # noqa: E402
from agent.runner import run_episode  # noqa: E402
from eval.metrics import score_episode  # noqa: E402

CODE_DIR = Path(__file__).resolve().parents[1]


def response(content=None, tool_calls=None):
    msg = NS(content=content, tool_calls=tool_calls)
    return NS(choices=[NS(message=msg)], usage=NS(prompt_tokens=10, completion_tokens=5))


def tool_call(i, name, args):
    return NS(id=f"call_{i}", type="function", function=NS(name=name, arguments=json.dumps(args)))


def format_gold(q):
    g = q["gold"]
    return f"{g:.4f}".rstrip("0").rstrip(".") if isinstance(g, float) else str(g)


class FakeClient:
    """Scripted stand-in for openai.OpenAI(...).chat.completions."""

    def __init__(self, behaviour, question):
        self.behaviour, self.q, self.turn = behaviour, question, 0
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kwargs):
        self.turn += 1
        q = self.q
        if self.behaviour == "refuser":
            return response("I can't help with that.\nFINAL ANSWER: UNANSWERABLE")
        if self.behaviour == "stale":
            return response("Based on my knowledge it was about 123.45.\nFINAL ANSWER: 123.45")
        # oracle
        if q["answer_type"] == "abstain":
            return response("That cannot be answered from the available data.\nFINAL ANSWER: UNANSWERABLE")
        if self.turn == 1 and "tools" in kwargs:
            calls = [tool_call(i, c["tool"], c["args"]) for i, c in enumerate(q["gold_calls"])]
            return response("", calls)
        return response(f"From the data, the answer is {format_gold(q)}.\nFINAL ANSWER: {format_gold(q)}")


def main() -> int:
    market = SnapshotMarket()
    names = {t: market.name(t) for t in market.tickers}
    questions = [json.loads(line) for line in open(CODE_DIR / "questions" / "questions_v1.jsonl", encoding="utf-8")]

    # Unit checks for the answer parser.
    assert parse_number("$1,234.56") == 1234.56
    assert parse_number("4.94 trillion") == 4.94e12
    assert parse_number("-3.12%") == -3.12
    assert parse_number("The answer is 80,785,500 shares") == 80785500
    assert parse_number("On October 9, 2026 it closed at 93.49") == 93.49
    assert parse_date("Sep 22, 2026") == "2026-09-22"
    assert parse_date("2026-09-22") == "2026-09-22"
    q_ticker = next(q for q in questions if q["answer_type"] == "ticker")
    assert grade(q_ticker, f"FINAL ANSWER: {names[q_ticker['gold']]}", names)["correct"]

    failures = []
    for behaviour in ("oracle", "stale", "refuser"):
        rows = []
        for q in questions:
            trace = run_episode(q, "C1", FakeClient(behaviour, q), "fake", market)
            assert trace["status"] == "ok", trace["status"]
            rows.append(score_episode(q, trace, names))
        acc = sum(r["correct"] for r in rows if r["answer_type"] != "abstain") / 150
        t5 = sum(r["correct"] for r in rows if r["answer_type"] == "abstain") / 50
        skip = sum(r["tool_skip"] for r in rows) / len(rows)
        cov = [r["coverage"] for r in rows if r["coverage"] is not None]
        strict = [r["grounding_strict"] for r in rows if r["grounding_strict"] is not None]
        print(f"{behaviour:8s} acc={acc:.2f}  T5={t5:.2f}  tool_skip={skip:.2f}  "
              f"coverage={sum(cov) / len(cov):.2f}  grounding_strict={sum(strict) / max(len(strict), 1):.2f}")
        if behaviour == "oracle":
            wrong = [r["id"] for r in rows if not r["correct"]]
            low_cov = [r["id"] for r in rows if r["coverage"] is not None and r["coverage"] < 1]
            if wrong or low_cov:
                failures.append(f"oracle wrong={wrong[:10]} low_coverage={low_cov[:10]}")
            lookups = [r for r in rows if r["tier"] == "T1" and r["grounding_strict"] is not None]
            if any(r["grounding_strict"] < 1 for r in lookups):
                failures.append("oracle T1 answers should be fully grounded in tool output")
        if behaviour == "stale" and (acc > 0.05 or t5 > 0 or skip < 0.7):
            failures.append(f"stale model scored too well: acc={acc} t5={t5} skip={skip}")
        if behaviour == "refuser" and (acc > 0 or t5 < 1):
            failures.append(f"refuser: acc={acc} t5={t5}")

    # C0 must not offer tools, C2 must offer python_exec.
    seen = {}

    class Spy(FakeClient):
        def create(self, **kwargs):
            seen[self.behaviour] = [t["function"]["name"] for t in kwargs.get("tools", [])]
            return response("FINAL ANSWER: 1")

    run_episode(questions[0], "C0", Spy("C0", questions[0]), "fake", market)
    run_episode(questions[0], "C2", Spy("C2", questions[0]), "fake", market)
    if seen["C0"] or "python_exec" not in seen["C2"]:
        failures.append(f"tool offering wrong: {seen}")

    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        return 1
    print("Pipeline test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
