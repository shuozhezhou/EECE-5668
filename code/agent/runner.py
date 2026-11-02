"""Run one question through an LLM agent under an experimental condition.

Conditions:
    C0  closed-book: no tools (measures stale answers and willingness to abstain)
    C1  data tools: get_quote, get_price_history, get_company_info
    C2  data tools + python_exec

The client is anything with an OpenAI-style `chat.completions.create`, e.g.
`openai.OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")` pointed at a
vLLM server started with `--enable-auto-tool-choice --tool-call-parser hermes`.
"""

from __future__ import annotations

import json
import time
from typing import Any

from .market import SnapshotMarket
from .tools import ToolBox

CONDITIONS = ("C0", "C1", "C2")

ANSWER_FORMAT = (
    "End your reply with one final line in exactly this form:\n"
    "FINAL ANSWER: <answer>\n"
    "where <answer> is one of: a single number (no units, no thousands separators; "
    "percentages as a number of percent, e.g. -3.12; large values written out in full), "
    "a ticker symbol, a date in YYYY-MM-DD format, or UNANSWERABLE."
)

RULES = (
    "Answer factual questions about stock prices, trading volumes and companies. "
    "Do not give investment advice. If a question cannot be answered with the information "
    "available to you, or asks for investment advice or predictions, answer UNANSWERABLE."
)


def system_prompt(condition: str, as_of_text: str) -> str:
    head = f"You are a financial data assistant. The current date and time is {as_of_text}."
    if condition == "C0":
        tools = "You do not have access to any tools, live data or the internet."
    else:
        tools = (
            "You have tools that return market data for S&P 500 companies. Use them to look up "
            "any price, volume or company figure you need; do not rely on memory for market data, "
            "because it may be outdated."
        )
        if condition == "C2":
            tools += " Use the python_exec tool for any arithmetic instead of computing in your head."
    return "\n\n".join([head, tools, RULES, ANSWER_FORMAT])


def _message_to_dict(msg: Any) -> dict[str, Any]:
    """Convert an OpenAI SDK message object into a plain dict for the next request."""
    out: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
    if getattr(msg, "tool_calls", None):
        out["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
            for tc in msg.tool_calls
        ]
    return out


def run_episode(
    question: dict[str, Any],
    condition: str,
    client: Any,
    model: str,
    market: SnapshotMarket,
    max_tool_rounds: int = 8,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Run the agent loop for one question and return a JSON-serializable trace."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; expected one of {CONDITIONS}")
    toolbox = ToolBox(market, enable_python=(condition == "C2"))
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt(condition, market.as_of_text)},
        {"role": "user", "content": question["question"]},
    ]
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    trace: dict[str, Any] = {"id": question["id"], "condition": condition, "model": model}
    started = time.time()
    final_text: str | None = None
    status = "ok"

    try:
        for round_idx in range(max_tool_rounds + 1):
            kwargs: dict[str, Any] = {
                "model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
            }
            # Last round: no tools offered, so the model must answer.
            if condition != "C0" and round_idx < max_tool_rounds:
                kwargs["tools"] = toolbox.schemas
                kwargs["tool_choice"] = "auto"
            resp = client.chat.completions.create(**kwargs)
            if resp.usage:
                usage["prompt_tokens"] += resp.usage.prompt_tokens or 0
                usage["completion_tokens"] += resp.usage.completion_tokens or 0
            msg = resp.choices[0].message
            messages.append(_message_to_dict(msg))
            if not getattr(msg, "tool_calls", None):
                final_text = msg.content or ""
                break
            for tc in msg.tool_calls:
                result = toolbox.call(tc.function.name, tc.function.arguments)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(result, ensure_ascii=False)})
        else:
            status = "max_rounds"
    except Exception as exc:  # noqa: BLE001 - record and move on; one bad episode must not stop a run
        status = f"error: {type(exc).__name__}: {exc}"

    trace.update({
        "status": status,
        "final_text": final_text,
        "tool_log": toolbox.log,
        "n_tool_calls": len(toolbox.log),
        "n_llm_calls": sum(1 for m in messages if m["role"] == "assistant"),
        "usage": usage,
        "latency_s": round(time.time() - started, 3),
        "messages": messages,
    })
    return trace
