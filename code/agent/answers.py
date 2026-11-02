"""Parse the agent's final answer and grade it against a question's gold answer.

The agent is asked to end with `FINAL ANSWER: <answer>`. Models do not always
comply exactly, so parsing is lenient: it falls back to the last line, accepts
"$1,234.56", "4.94 trillion", "-3.12%", "Sep 22, 2026", company names for tickers,
and common refusal phrasings for abstention.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

FINAL_RE = re.compile(r"FINAL\s*ANSWER\s*[:：]\s*(.+)", re.IGNORECASE)
NUMBER_RE = re.compile(
    r"(?P<sign>[-+−])?\s*\$?\s*(?P<num>\d[\d,]*(?:\.\d+)?|\.\d+)\s*"
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
DATE_FORMATS = ["%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y", "%m/%d/%Y", "%d %B %Y"]
MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
DATE_TEXT_RE = re.compile(
    rf"\b\d{{4}}-\d{{2}}-\d{{2}}\b|\b{MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b|\b\d{{1,2}}/\d{{1,2}}/\d{{4}}\b",
    re.IGNORECASE,
)


def extract_final(text: str | None) -> str | None:
    """The text after the last 'FINAL ANSWER:' marker, else the last non-empty line."""
    if not text:
        return None
    matches = FINAL_RE.findall(text)
    if matches:
        return matches[-1].strip().strip("*").strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else None


def is_abstention(text: str | None) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(re.search(p, low) for p in ABSTAIN_PATTERNS)


def parse_number(text: str) -> float | None:
    """First number in text, applying a scale suffix (4.94 trillion -> 4.94e12)."""
    cleaned = DATE_TEXT_RE.sub(" ", text)
    m = NUMBER_RE.search(cleaned)
    if not m:
        return None
    value = float(m.group("num").replace(",", ""))
    suffix = (m.group("suffix") or "").lower()
    value *= SCALE.get(suffix, 1.0)
    if m.group("sign") in ("-", "−"):
        value = -value
    return value


def parse_date(text: str) -> str | None:
    m = DATE_TEXT_RE.search(text)
    if not m:
        return None
    raw = re.sub(r"(\d)(st|nd|rd|th)", r"\1", m.group(0)).replace(".", "").replace("Sept", "Sep")
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


def grade(question: dict[str, Any], response: str | None, names: dict[str, str] | None = None) -> dict[str, Any]:
    """Grade one response. Returns the parsed answer and whether it is correct."""
    final = extract_final(response)
    abstained = final is not None and (final.strip().upper().startswith("UNANSWERABLE") or is_abstention(final))
    out: dict[str, Any] = {"final": final, "abstained": abstained, "parsed": None, "correct": False}
    atype = question["answer_type"]

    if atype == "abstain":
        # Accept a refusal anywhere in the reply, as long as no concrete answer is asserted.
        out["abstained"] = abstained or (is_abstention(response) and parse_number(final or "") is None)
        out["correct"] = out["abstained"]
        return out
    if final is None or abstained:
        return out

    if atype == "number":
        value = parse_number(final)
        out["parsed"] = value
        if value is not None:
            gold, tol = float(question["gold"]), question.get("tolerance", {"abs": 0.0, "rel": 0.0})
            out["correct"] = abs(value - gold) <= max(tol.get("abs", 0.0), tol.get("rel", 0.0) * abs(gold)) + 1e-9
    elif atype == "ticker":
        value = parse_ticker(final, question["meta"]["tickers"], names)
        out["parsed"] = value
        out["correct"] = value == question["gold"]
    elif atype == "date":
        value = parse_date(final)
        out["parsed"] = value
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
        value = float(raw.replace(",", ""))
        suffix = (m.group("suffix") or "").lower()
        scaled = suffix in SCALE
        value *= SCALE.get(suffix, 1.0)
        if m.group("sign") in ("-", "−"):
            value = -value
        decimals = len(raw.split(".")[1]) if "." in raw else 0
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
