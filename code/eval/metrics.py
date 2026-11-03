"""Per-episode metrics.

For each (question, trace) pair:
    correct          answer matches gold within tolerance (T5: the agent abstained)
    abstained        the agent declined to answer
    data_calls       successful data-tool calls (python_exec excluded)
    tool_skip        gave a concrete answer without any successful data call (C1/C2)
    unexecuted_tool_call  the reply is tool-call text the server did not execute (no answer)
    coverage         fraction of the question's required (ticker, date) data the agent retrieved
    numbers          every number asserted in the final reply, each labelled
                       tool      - equals a value in some tool output (incl. python stdout)
                       question  - appears in the question itself (e.g. "20 trading days")
                       derived   - equals the gold answer but not any tool output
                                   (mental arithmetic that happened to be right)
                       ungrounded- none of the above
                     A reply with no final answer (e.g. an unexecuted tool call) asserts nothing.
    grounding_strict (tool + question) / all numbers
    grounding        (tool + question + derived) / all numbers
"""

from __future__ import annotations

import re
from typing import Any

from agent.answers import collect_numbers, grade, is_unexecuted_tool_call, matches, numbers_in_text
from agent.market import normalize_ticker
from agent.runner import NO_TOOL_CONDITIONS

DATA_TOOLS = {"get_quote", "get_price_history", "get_company_info"}


def _successful(log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [c for c in log if isinstance(c.get("result"), dict) and "error" not in c["result"]]


def coverage(question: dict[str, Any], log: list[dict[str, Any]]) -> float | None:
    """Share of required (ticker, date) / (ticker, field) items present in tool outputs."""
    needs: list[tuple[str, str]] = []
    for need in question.get("required_data", []):
        if "dates" in need:
            needs += [(need["ticker"], d) for d in need["dates"]]
        else:
            needs.append((need["ticker"], "field:" + need["field"]))
    if not needs:
        return None
    have: set[tuple[str, str]] = set()
    for call in _successful(log):
        res = call["result"]
        t = normalize_ticker(res.get("ticker", ""))
        if call["tool"] == "get_price_history":
            have |= {(t, r["date"]) for r in res.get("rows", [])}
        elif call["tool"] == "get_quote" and "date" in res:
            have.add((t, res["date"]))
        elif call["tool"] == "get_company_info":
            have |= {(t, "field:" + k) for k, v in res.items() if v is not None}
    return sum(1 for n in needs if n in have) / len(needs)


def grounding(question: dict[str, Any], trace: dict[str, Any], grade_result: dict[str, Any]) -> dict[str, Any]:
    final_text = (trace.get("final_text") or "") if grade_result["final"] is not None else ""
    nums = numbers_in_text(final_text)
    evidence = [x for c in trace.get("tool_log", []) for x in collect_numbers(c.get("result"))]
    question_nums = [n["value"] for n in numbers_in_text(question["question"])]
    gold = question.get("gold")
    labelled = []
    for n in nums:
        if matches(n, evidence):
            label = "tool"
        elif matches(n, question_nums):
            label = "question"
        elif isinstance(gold, (int, float)) and grade_result["correct"] and matches(n, [float(gold)]):
            label = "derived"
        else:
            label = "ungrounded"
        labelled.append({**n, "label": label})
    total = len(labelled)
    count = {k: sum(1 for n in labelled if n["label"] == k) for k in ("tool", "question", "derived", "ungrounded")}
    return {
        "numbers": labelled,
        "n_numbers": total,
        **{f"n_{k}": v for k, v in count.items()},
        "grounding_strict": (count["tool"] + count["question"]) / total if total else None,
        "grounding": (total - count["ungrounded"]) / total if total else None,
    }


def _confidence(text: str | None) -> float | None:
    """The 0-100 value of a 'CONFIDENCE: <n>' line (condition C0F), if present."""
    m = re.search(r"CONFIDENCE\s*[:：]\s*(\d{1,3}(?:\.\d+)?)", text or "", re.IGNORECASE)
    return min(float(m.group(1)), 100.0) if m else None


def score_episode(question: dict[str, Any], trace: dict[str, Any], names: dict[str, str] | None = None) -> dict[str, Any]:
    g = grade(question, trace.get("final_text"), names)
    log = trace.get("tool_log", [])
    data_calls = [c for c in _successful(log) if c["tool"] in DATA_TOOLS]
    answered = not g["abstained"] and g["final"] is not None
    row = {
        "id": question["id"],
        "tier": question["tier"],
        "template": question["template"],
        "answer_type": question["answer_type"],
        "model": trace.get("model"),
        "condition": trace.get("condition"),
        "status": trace.get("status"),
        "final": g["final"],
        "parsed": g["parsed"],
        "gold": question["gold"],
        "correct": bool(g["correct"]),
        "abstained": bool(g["abstained"]),
        "answered": answered,
        "unexecuted_tool_call": is_unexecuted_tool_call(trace.get("final_text")),
        "n_tool_calls": len(log),
        "n_data_calls": len(data_calls),
        "n_tool_errors": len(log) - len(_successful(log)),
        "tool_skip": answered and not data_calls and trace.get("condition") not in NO_TOOL_CONDITIONS,
        "coverage": coverage(question, log),
        # How far off a numeric answer is (C0F: how stale/fabricated the recalled value is).
        "rel_error": (abs(g["parsed"] - float(question["gold"])) / abs(float(question["gold"]))
                      if question["answer_type"] == "number" and isinstance(g["parsed"], float)
                      and float(question["gold"]) != 0 else None),
        "confidence": _confidence(trace.get("final_text")),
        "prompt_tokens": trace.get("usage", {}).get("prompt_tokens"),
        "completion_tokens": trace.get("usage", {}).get("completion_tokens"),
        "latency_s": trace.get("latency_s"),
    }
    row.update({k: v for k, v in grounding(question, trace, g).items() if k != "numbers"})
    return row
