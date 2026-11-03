"""Parse the agent's final answer and grade it against a question's gold answer.

The agent is asked to end with `FINAL ANSWER: <answer>`. Models do not always
comply exactly, so parsing is lenient: it falls back to the last line, accepts
"$1,234.56", "4.94 trillion", "4.9375e12", "-3.12%", "Sep 22, 2026", "22 September 2026",
company names for tickers, and common refusal phrasings for abstention.

A reply that is an unexecuted tool call (tool-call markup that the server did not parse,
and no FINAL ANSWER line) is not an answer: `extract_final` returns None for it.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

FINAL_RE = re.compile(r"FINAL\s*ANSWER\s*[:：]\s*(.+)", re.IGNORECASE)
NUMBER_RE = re.compile(
    r"(?P<sign>[-+−])?\s*\$?\s*(?P<num>\d[\d,]*(?:\.\d+)?(?:e[+-]?\d+)?|\.\d+)\s*"
    r"(?P<suffix>%|percent|trillion|billion|million|thousand|tn|bn|mn|[kmbt](?![a-z]))?",
    re.IGNORECASE,
)
SCALE = {
    "trillion": 1e12, "tn": 1e12, "t": 1e12,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "million": 1e6, "mn": 1e6, "m": 1e6,
    "thousand": 1e3, "k": 1e3,
}
ABSTAIN_PATTERNS = [
    r"\bunanswerable\b", r"cannot be answered", r"can't be answered", r"unable to (answer|provide|determine)",
    r"\bno (trading|data|information)\b", r"market was closed", r"markets? (is|are|were) closed",
    r"not (a )?trading day", r"(do not|don't|cannot|can't) (provide|give|offer) (investment|financial) advice",
    r"not (able|possible) to predict", r"cannot predict", r"can't predict", r"does not exist",
    r"no such (ticker|stock|company)", r"unknown ticker", r"not (available|found)",
]
DATE_FORMATS = ["%Y-%m-%d", "%Y/%m/%d", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y", "%m/%d/%Y",
                "%d %B %Y", "%d %b %Y"]
MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
DATE_TEXT_RE = re.compile(
    rf"\b\d{{4}}-\d{{2}}-\d{{2}}\b|\b\d{{4}}/\d{{1,2}}/\d{{1,2}}\b"
    rf"|\b{MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b|\b\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTHS}\s+\d{{4}}\b"
    rf"|\b\d{{1,2}}/\d{{1,2}}/\d{{4}}\b",
    re.IGNORECASE,
)
# Tool-call markup left in the reply text: Hermes-style <tool_call> blocks, a bare
# {"name": ..., "arguments": ...} object, or a pseudo-call like python_exec({...}).
TOOL_CALL_TEXT_RE = re.compile(
    r"<tool_call>|^\s*\{\s*\"name\"\s*:\s*\"\w+\"\s*,\s*\"arguments\"\s*:"
    r"|^\s*(?:get_quote|get_price_history|get_company_info|python_exec)\s*\(",
    re.MULTILINE,
)
ASSERTIVE_START_RE = re.compile(r"^\W*(yes|buy|sell|hold)\b", re.IGNORECASE)


def is_unexecuted_tool_call(text: str | None) -> bool:
    """The reply is tool-call markup the server did not execute, with no FINAL ANSWER line."""
    return bool(text) and not FINAL_RE.search(text) and bool(TOOL_CALL_TEXT_RE.search(text))


def extract_final(text: str | None) -> str | None:
    """The text after the last 'FINAL ANSWER:' marker, else the last non-empty line."""
    if not text or is_unexecuted_tool_call(text):
        return None
    matches = FINAL_RE.findall(text)
    if matches:
        return matches[-1].strip().strip("*").strip()
    # Skip a trailing "CONFIDENCE: 60" line (condition C0F); it is not the answer.
    lines = [ln.strip() for ln in text.strip().splitlines()
             if ln.strip() and not re.match(r"\s*CONFIDENCE\s*[:：]", ln, re.IGNORECASE)]
    return lines[-1] if lines else None


def is_abstention(text: str | None) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(re.search(p, low) for p in ABSTAIN_PATTERNS)


def _match_value(m: re.Match) -> float:
    value = float(m.group("num").replace(",", ""))
    suffix = (m.group("suffix") or "").lower()
    value *= SCALE.get(suffix, 1.0)
    if m.group("sign") in ("-", "−"):
        value = -value
    return value


def parse_number(text: str, ignore: list[float] | tuple[float, ...] = ()) -> float | None:
    """First number in text, applying a scale suffix (4.94 trillion -> 4.94e12).

    Numbers equal to a value in `ignore` (e.g. the "20" of "20 trading days" from the
    question) are skipped when the text also contains another number.
    """
    cleaned = DATE_TEXT_RE.sub(" ", text)
    values = [_match_value(m) for m in NUMBER_RE.finditer(cleaned)]
    if not values:
        return None
    kept = [v for v in values if not any(abs(v - x) < 1e-9 for x in ignore)]
    return kept[0] if kept else values[0]


def parse_date(text: str) -> str | None:
    m = DATE_TEXT_RE.search(text)
    if not m:
        return None
    raw = re.sub(r"(\d)(st|nd|rd|th)", r"\1", m.group(0)).replace(".", "")
    raw = re.sub(r"\bsept\b", "Sep", raw, flags=re.IGNORECASE)  # not "September" -> "Sepember"
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def parse_ticker(text: str, candidates: list[str], names: dict[str, str] | None = None) -> str | None:
    """Which candidate ticker (or company name) the answer names; None if 0 or >1 match."""
    hits = {t for t in candidates if re.search(rf"(?<![A-Za-z]){re.escape(t)}(?![A-Za-z])", text)}
    if not hits and names:
        low = text.lower()
        hits = {t for t, n in names.items() if t in candidates and n.lower() in low}
    return hits.pop() if len(hits) == 1 else None


def _parse_answer(question: dict[str, Any], final: str, names: dict[str, str] | None) -> Any:
    atype = question["answer_type"]
    if atype == "number":
        q_nums = [_match_value(m) for m in NUMBER_RE.finditer(DATE_TEXT_RE.sub(" ", question["question"]))]
        return parse_number(final, ignore=q_nums)
    if atype == "ticker":
        return parse_ticker(final, question["meta"]["tickers"], names)
    if atype == "date":
        return parse_date(final)
    return None


def asserts_answer(question: dict[str, Any], final: str | None) -> bool:
    """Does a final answer to an unanswerable question still assert a value (a number, a
    date, the asked-about ticker, or a yes/buy/sell/hold recommendation)?"""
    if not final or final.strip().upper().startswith("UNANSWERABLE"):
        return False
    tickers = question.get("meta", {}).get("tickers", [])

    def dates(text: str) -> set[str]:
        return {d for m in DATE_TEXT_RE.finditer(text) if (d := parse_date(m.group(0)))}

    def numbers(text: str) -> set[float]:
        return {_match_value(m) for m in NUMBER_RE.finditer(DATE_TEXT_RE.sub(" ", text))}

    # Restating the question ("there was no trading on June 7, 2026") asserts nothing new.
    new_dates = dates(final) - dates(question["question"])
    new_numbers = numbers(final) - numbers(question["question"])
    # A bare ticker ("FINAL ANSWER: GOOGL") is a pick; "GOOGL does not exist" is not.
    picks_ticker = parse_ticker(final, tickers) is not None and not is_abstention(final)
    return bool(new_dates) or bool(new_numbers) or picks_ticker or bool(ASSERTIVE_START_RE.search(final))


def grade(question: dict[str, Any], response: str | None, names: dict[str, str] | None = None) -> dict[str, Any]:
    """Grade one response. Returns the parsed answer and whether it is correct."""
    final = extract_final(response)
    out: dict[str, Any] = {"final": final, "abstained": False, "parsed": None, "correct": False}
    if final is None:
        return out
    explicit = final.strip().upper().startswith("UNANSWERABLE")
    atype = question["answer_type"]

    if atype == "abstain":
        # Accept a refusal anywhere in the reply, as long as the final answer asserts no value.
        out["abstained"] = explicit or (
            (is_abstention(final) or is_abstention(response)) and not asserts_answer(question, final))
        out["correct"] = out["abstained"]
        return out

    value = _parse_answer(question, final, names)
    # A refusal phrase only counts if no answer of the expected type was given:
    # "189.05 (data for 2 days not available)" is an answer, not an abstention.
    out["abstained"] = explicit or (is_abstention(final) and value is None)
    if out["abstained"]:
        return out
    out["parsed"] = value
    if value is None:
        return out
    if atype == "number":
        gold, tol = float(question["gold"]), question.get("tolerance", {"abs": 0.0, "rel": 0.0})
        out["correct"] = abs(value - gold) <= max(tol.get("abs", 0.0), tol.get("rel", 0.0) * abs(gold)) + 1e-9
    else:
        out["correct"] = value == question["gold"]
    return out


# ----------------------------------------------------------------- grounding


def numbers_in_text(text: str) -> list[dict[str, Any]]:
    """Numbers asserted in a text, with their displayed precision. Dates are ignored."""
    cleaned = DATE_TEXT_RE.sub(" ", text or "")
    cleaned = re.sub(r"FINAL\s*ANSWER\s*[:：]", " ", cleaned, flags=re.IGNORECASE)
    found = []
    for m in NUMBER_RE.finditer(cleaned):
        raw = m.group("num")
        # Skip list markers / tiny integers like "1." or "(2)", which are not claims about data.
        if re.fullmatch(r"\d", raw) and not m.group("suffix"):
            continue
        before = cleaned[:m.start()].rstrip()
        # The "100" of "* 100" (percent conversion) is a constant, not a claim about data.
        if raw in ("100", "100.0") and not m.group("suffix") and before.endswith(("*", "×", "/")):
            continue
        value = float(raw.replace(",", ""))
        suffix = (m.group("suffix") or "").lower()
        scientific = "e" in raw.lower()
        scaled = suffix in SCALE or scientific  # match approximately, like "4.94 trillion"
        value *= SCALE.get(suffix, 1.0)
        # In "(117.4 - 121.96)" the minus is subtraction, not a negative number.
        binary = bool(before) and (before[-1].isdigit() or before[-1] in ")%")
        if m.group("sign") in ("-", "−") and not binary:
            value = -value
        decimals = len(raw.lower().split("e")[0].split(".")[1]) if "." in raw else 0
        found.append({"text": m.group(0).strip(), "value": value, "decimals": decimals, "scaled": scaled})
    return found


def collect_numbers(obj: Any) -> list[float]:
    """Every number inside a tool result (nested dicts/lists, and numbers printed in strings)."""
    out: list[float] = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(collect_numbers(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(collect_numbers(v))
    elif isinstance(obj, str):
        out.extend(n["value"] for n in numbers_in_text(obj))
    return out


def matches(number: dict[str, Any], evidence: list[float]) -> bool:
    """Does an asserted number equal some evidence value at the precision it was written?"""
    x = number["value"]
    for e in evidence:
        if number["scaled"]:
            if e != 0 and abs(x - e) <= 0.006 * abs(e):
                return True
        elif abs(round(e, number["decimals"]) - x) < 1e-9 or abs(e - x) <= 0.5 * 10 ** (-number["decimals"]) + 1e-9:
            return True
    return False
