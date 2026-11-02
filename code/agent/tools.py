"""Tools exposed to the LLM agent, answering only from a SnapshotMarket.

Every tool returns a JSON-serializable dict. Errors are returned as {"error": ...}
instead of raised, because the agent should see them and react (e.g. abstain on an
unknown ticker). `ToolBox` also keeps a log of every call, which the evaluator uses
for tool-call accuracy and for the number-level grounding metric.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from typing import Any

import pandas as pd

from .market import SnapshotMarket, normalize_ticker

MAX_HISTORY_ROWS = 260
PYTHON_TIMEOUT_S = 10
PYTHON_MAX_OUTPUT = 4000

DATA_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_quote",
            "description": (
                "Latest daily quote for a stock: the most recent completed trading session's "
                "open, high, low, close, volume, and the change versus the previous close."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL."},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price_history",
            "description": (
                "Daily OHLCV prices for a stock between two dates (inclusive). Only trading "
                "days are returned; weekends and market holidays have no rows."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL."},
                    "start_date": {"type": "string", "description": "First date, YYYY-MM-DD."},
                    "end_date": {"type": "string", "description": "Last date, YYYY-MM-DD."},
                },
                "required": ["ticker", "start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_info",
            "description": (
                "Company profile: name, GICS sector and sub-industry, and, for the largest "
                "companies, market capitalization, P/E ratios, dividend yield and 52-week range."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL."},
                },
                "required": ["ticker"],
            },
        },
    },
]

PYTHON_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "python_exec",
        "description": (
            "Run a short Python 3 program and return its stdout. Use print() to output "
            "results. No network access to market data; pass the numbers you need in the code."
        ),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python source code to run."}},
            "required": ["code"],
        },
    },
}


def _price_row(row: pd.Series) -> dict[str, Any]:
    return {
        "date": row["date"].isoformat(),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": int(row["volume"]),
    }


def _optional_number(value: Any, decimals: int | None = 2) -> float | int | None:
    if value is None or pd.isna(value):
        return None
    if decimals is None:
        return int(value)
    return round(float(value), decimals)


class ToolBox:
    """Dispatches tool calls against one snapshot and logs them."""

    def __init__(self, market: SnapshotMarket, enable_python: bool = False):
        self.market = market
        self.enable_python = enable_python
        self.log: list[dict[str, Any]] = []

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return DATA_TOOL_SCHEMAS + ([PYTHON_TOOL_SCHEMA] if self.enable_python else [])

    def reset(self) -> None:
        self.log = []

    def call(self, name: str, arguments: dict[str, Any] | str | None) -> dict[str, Any]:
        """Run one tool call. `arguments` may be a dict or the raw JSON string from the LLM."""
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                result = {"error": f"Arguments are not valid JSON: {exc}"}
                self.log.append({"tool": name, "args": arguments, "result": result})
                return result
        arguments = arguments or {}

        handlers = {
            "get_quote": self.get_quote,
            "get_price_history": self.get_price_history,
            "get_company_info": self.get_company_info,
        }
        if self.enable_python:
            handlers["python_exec"] = self.python_exec

        if name not in handlers:
            result = {"error": f"Unknown tool '{name}'. Available tools: {sorted(handlers)}"}
        else:
            try:
                result = handlers[name](**arguments)
            except TypeError as exc:  # missing/unexpected arguments
                result = {"error": f"Bad arguments for {name}: {exc}"}
        self.log.append({"tool": name, "args": arguments, "result": result})
        return result

    # ------------------------------------------------------------------ helpers

    def _unknown_ticker(self, ticker: str) -> dict[str, Any] | None:
        if not self.market.has_ticker(ticker):
            return {"error": f"Unknown ticker '{ticker}'. No data is available for this symbol."}
        return None

    # -------------------------------------------------------------------- tools

    def get_quote(self, ticker: str) -> dict[str, Any]:
        if err := self._unknown_ticker(ticker):
            return err
        t = normalize_ticker(ticker)
        rows = self.market.history(t, end=self.market.as_of).tail(2)
        last = rows.iloc[-1]
        quote = {"ticker": t, "name": self.market.name(t), **_price_row(last)}
        if len(rows) == 2:
            prev_close = float(rows.iloc[0]["close"])
            quote["previous_close"] = prev_close
            quote["change"] = round(float(last["close"]) - prev_close, 2)
            quote["change_percent"] = round((float(last["close"]) / prev_close - 1) * 100, 2)
        quote["currency"] = "USD"
        return quote

    def get_price_history(self, ticker: str, start_date: str, end_date: str) -> dict[str, Any]:
        if err := self._unknown_ticker(ticker):
            return err
        try:
            start, end = date.fromisoformat(str(start_date)), date.fromisoformat(str(end_date))
        except ValueError:
            return {"error": "Dates must be in YYYY-MM-DD format."}
        if start > end:
            return {"error": "start_date is after end_date."}
        as_of = self.market.as_of
        if start > as_of:
            return {"error": f"No data after {as_of.isoformat()}, the most recent trading session."}

        t = normalize_ticker(ticker)
        result: dict[str, Any] = {"ticker": t, "start_date": start.isoformat(), "end_date": end.isoformat()}
        if end > as_of:
            end = as_of
            result["note"] = f"Data is only available through {as_of.isoformat()}."
        frame = self.market.history(t, start, end)
        if len(frame) > MAX_HISTORY_ROWS:
            frame = frame.tail(MAX_HISTORY_ROWS)
            result["note"] = f"Truncated to the last {MAX_HISTORY_ROWS} trading days of the range."
        first_available = self.market.history(t).iloc[0]["date"]
        if start < first_available and len(frame) == 0:
            result["note"] = f"No data before {first_available.isoformat()} for this ticker."
        elif len(frame) == 0:
            result["note"] = "No trading sessions in this date range (weekend or market holiday)."
        result["n_rows"] = len(frame)
        result["rows"] = [_price_row(r) for _, r in frame.iterrows()]
        return result

    def get_company_info(self, ticker: str) -> dict[str, Any]:
        if err := self._unknown_ticker(ticker):
            return err
        t = normalize_ticker(ticker)
        c = self.market.constituents.loc[t] if t in self.market.constituents.index else None
        info: dict[str, Any] = {
            "ticker": t,
            "name": self.market.name(t),
            "sector": None if c is None else str(c["gics_sector"]),
            "sub_industry": None if c is None else str(c["gics_sub_industry"]),
        }
        if t in self.market.info.index:
            i = self.market.info.loc[t]
            info.update(
                {
                    "market_cap": _optional_number(i["marketCap"], None),
                    "trailing_pe": _optional_number(i["trailingPE"]),
                    "forward_pe": _optional_number(i["forwardPE"]),
                    "dividend_yield_percent": _optional_number(i["dividendYield"]),
                    "fifty_two_week_high": _optional_number(i["fiftyTwoWeekHigh"]),
                    "fifty_two_week_low": _optional_number(i["fiftyTwoWeekLow"]),
                    "currency": "USD",
                }
            )
        else:
            info["note"] = "Valuation data (market cap, P/E, dividend yield) is not available for this company."
        return info

    def python_exec(self, code: str) -> dict[str, Any]:
        # Minimal sandbox: isolated interpreter, time limit, truncated output. Good enough
        # for arithmetic over numbers the agent already has; not a security boundary.
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", str(code)],
                capture_output=True,
                text=True,
                timeout=PYTHON_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return {"error": f"Execution timed out after {PYTHON_TIMEOUT_S} seconds."}
        result: dict[str, Any] = {"stdout": proc.stdout[-PYTHON_MAX_OUTPUT:]}
        if proc.returncode != 0:
            result["stderr"] = proc.stderr[-PYTHON_MAX_OUTPUT:]
            result["returncode"] = proc.returncode
        return result
