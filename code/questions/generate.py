"""Generate the benchmark questions (tiers T1-T3 and T5) from a snapshot.

Ground truth is computed by calling the same ToolBox the agent uses, so every gold
answer is derivable from numbers the agent can actually see. The canonical calls are
saved with each question (`gold_calls`), together with the (ticker, date) data the
question needs (`required_data`), which the evaluator uses for tool-call accuracy.

Usage (from the `code/` directory):
    py -m questions.generate                       # Windows
    python -m questions.generate --snapshot-dir /content/drive/MyDrive/EECE5668/code/data/snapshots/2026-11-02
"""

from __future__ import annotations

import argparse
import json
import random
import string
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

from agent.market import DEFAULT_SNAPSHOT, SnapshotMarket
from agent.tools import ToolBox

SEED = 5668
QUESTION_WINDOW_START = date(2026, 8, 3)  # well after any current model's training cutoff
NON_TRADING_WINDOW_START = date(2026, 6, 1)  # includes the July 3 holiday
DUPLICATE_SHARE_CLASSES = {"GOOG", "FOX", "NWS"}  # keep GOOGL, FOXA, NWSA

# How many questions each template contributes (T1, T2, T3, T5: 50 each).
COUNTS = {
    "t1_close": 16, "t1_high_low": 8, "t1_volume": 8, "t1_latest_close": 8,
    "t1_market_cap": 5, "t1_pe": 5,
    "t2_best_return": 14, "t2_higher_close": 8, "t2_most_volume": 10,
    "t2_close_diff": 9, "t2_return_gap": 9,
    "t3_pct_change": 12, "t3_avg_close": 12, "t3_max_close": 8,
    "t3_min_close_date": 8, "t3_total_volume": 10,
    "t5_future": 12, "t5_fake_ticker": 12, "t5_non_trading_day": 12, "t5_advice": 14,
}

TIER = {name: name[:2].upper() for name in COUNTS}

# Answer tolerances: a numeric answer is correct if |pred - gold| <= max(abs, rel * |gold|).
TOL_EXACT_PRICE = {"abs": 0.01, "rel": 0.0}
TOL_COUNT = {"abs": 0.0, "rel": 0.001}
TOL_PERCENT = {"abs": 0.05, "rel": 0.0}  # percentage points
TOL_DERIVED = {"abs": 0.0, "rel": 0.002}
TOL_INFO = {"abs": 0.0, "rel": 0.01}


class Generator:
    def __init__(self, market: SnapshotMarket, seed: int = SEED):
        self.m = market
        self.rng = random.Random(seed)

        excluded = {e["ticker"] for e in market.metadata.get("split_events", [])}
        excluded |= {e["ticker"] for e in market.metadata.get("flagged_short", {}).get("tickers", [])}
        excluded |= DUPLICATE_SHARE_CLASSES
        self.excluded = sorted(excluded)
        self.universe = [t for t in market.tickers if t not in excluded]
        # The largest companies (with valuation data) are the ones people actually ask about.
        self.big = [t for t in market.info.index if t in self.universe]
        self.window = market.trading_days_between(QUESTION_WINDOW_START, market.as_of)

        sectors = market.constituents["gics_sector"]
        self.by_sector: dict[str, list[str]] = {}
        for t in self.universe:
            if t in sectors.index:
                self.by_sector.setdefault(str(sectors[t]), []).append(t)

    # ------------------------------------------------------------------ helpers

    def ticker(self, big_prob: float = 0.7) -> str:
        return self.rng.choice(self.big if self.rng.random() < big_prob else self.universe)

    def same_sector(self, k: int) -> list[str]:
        """k distinct tickers from one sector, biased toward large companies."""
        while True:
            anchor = self.ticker(0.8)
            sector = str(self.m.constituents.loc[anchor, "gics_sector"])
            pool = [t for t in self.by_sector[sector] if t != anchor]
            if len(pool) >= k - 1:
                big_pool = [t for t in pool if t in self.big]
                src = big_pool if len(big_pool) >= k - 1 and self.rng.random() < 0.6 else pool
                return [anchor] + self.rng.sample(src, k - 1)

    def ref(self, t: str, allow_name_only: bool = True) -> tuple[str, str]:
        """How the question names the company: 'Name (TICK)', 'TICK', or just 'Name'."""
        name = self.m.name(t)
        r = self.rng.random()
        if allow_name_only and t in self.big and r < 0.2:
            return name, "name"
        # A bare one-letter ticker ("did T close") reads as a typo, so always name it.
        if r < 0.6 or len(t) == 1:
            return f"{name} ({t})", "name_ticker"
        return t, "ticker"

    def fmt(self, d: date) -> str:
        return d.isoformat() if self.rng.random() < 0.3 else f"{d:%B} {d.day}, {d.year}"

    def day(self) -> date:
        return self.rng.choice(self.window)

    def span(self, min_days: int = 5, max_days: int = 40) -> tuple[date, date]:
        n = self.rng.randint(min_days, max_days)
        i = self.rng.randint(0, len(self.window) - n)
        return self.window[i], self.window[i + n - 1]

    def new_toolbox(self) -> ToolBox:
        return ToolBox(self.m)

    @staticmethod
    def calls(tb: ToolBox) -> list[dict[str, Any]]:
        return [{"tool": c["tool"], "args": c["args"]} for c in tb.log]

    @staticmethod
    def closes(tb: ToolBox, t: str, start: date, end: date) -> dict[str, dict[str, Any]]:
        res = tb.call("get_price_history", {"ticker": t, "start_date": start.isoformat(), "end_date": end.isoformat()})
        return {r["date"]: r for r in res["rows"]}

    @staticmethod
    def question(template: str, text: str, answer_type: str, gold: Any, **extra: Any) -> dict[str, Any]:
        q = {"template": template, "tier": TIER[template], "question": text, "answer_type": answer_type, "gold": gold}
        q.update(extra)
        return q

    # ---------------------------------------------------------------- T1 lookup

    def t1_close(self):
        t, d = self.ticker(), self.day()
        r, style = self.ref(t)
        text = self.rng.choice([
            "What was the closing price of {r} on {d}?",
            "At what price did {r} close on {d}?",
            "How much did {r} stock close at on {d}?",
        ]).format(r=r, d=self.fmt(d))
        tb = self.new_toolbox()
        row = self.closes(tb, t, d, d)[d.isoformat()]
        return self.question("t1_close", text, "number", row["close"], unit="USD", tolerance=TOL_EXACT_PRICE,
                             required_data=[{"ticker": t, "dates": [d.isoformat()]}], gold_calls=self.calls(tb),
                             meta={"tickers": [t], "ref_style": style})

    def t1_high_low(self):
        t, d = self.ticker(), self.day()
        r, style = self.ref(t)
        field = self.rng.choice(["high", "low"])
        word = {"high": "highest", "low": "lowest"}[field]
        text = self.rng.choice([
            "What was the {w} price {r} traded at during the session on {d}?",
            "What was the intraday {f} of {r} on {d}?",
        ]).format(w=word, f=field, r=r, d=self.fmt(d))
        tb = self.new_toolbox()
        row = self.closes(tb, t, d, d)[d.isoformat()]
        return self.question("t1_high_low", text, "number", row[field], unit="USD", tolerance=TOL_EXACT_PRICE,
                             required_data=[{"ticker": t, "dates": [d.isoformat()]}], gold_calls=self.calls(tb),
                             meta={"tickers": [t], "ref_style": style, "field": field})

    def t1_volume(self):
        t, d = self.ticker(), self.day()
        r, style = self.ref(t)
        text = self.rng.choice([
            "How many shares of {r} were traded on {d}?",
            "What was the trading volume of {r} on {d}?",
        ]).format(r=r, d=self.fmt(d))
        tb = self.new_toolbox()
        row = self.closes(tb, t, d, d)[d.isoformat()]
        return self.question("t1_volume", text, "number", row["volume"], unit="shares", tolerance=TOL_COUNT,
                             required_data=[{"ticker": t, "dates": [d.isoformat()]}], gold_calls=self.calls(tb),
                             meta={"tickers": [t], "ref_style": style})

    def t1_latest_close(self):
        t = self.ticker()
        r, style = self.ref(t)
        text = self.rng.choice([
            "What is the latest closing price of {r}?",
            "What did {r} close at in the most recent trading session?",
            "Where did {r} stock close today?",
        ]).format(r=r)
        tb = self.new_toolbox()
        quote = tb.call("get_quote", {"ticker": t})
        return self.question("t1_latest_close", text, "number", quote["close"], unit="USD", tolerance=TOL_EXACT_PRICE,
                             required_data=[{"ticker": t, "dates": [self.m.as_of.isoformat()]}],
                             gold_calls=self.calls(tb), meta={"tickers": [t], "ref_style": style})

    def _t1_info(self, template: str, field: str, phrasings: list[str], unit: str):
        t = self.rng.choice(self.big)
        r, style = self.ref(t)
        tb = self.new_toolbox()
        info = tb.call("get_company_info", {"ticker": t})
        if info.get(field) is None:
            return None
        text = self.rng.choice(phrasings).format(r=r)
        return self.question(template, text, "number", info[field], unit=unit, tolerance=TOL_INFO,
                             required_data=[{"ticker": t, "field": field}], gold_calls=self.calls(tb),
                             meta={"tickers": [t], "ref_style": style})

    def t1_market_cap(self):
        return self._t1_info("t1_market_cap", "market_cap", [
            "What is the current market capitalization of {r}?",
            "How much is {r} worth in total market value right now?",
        ], "USD")

    def t1_pe(self):
        return self._t1_info("t1_pe", "trailing_pe", [
            "What is the trailing P/E ratio of {r}?",
            "What price-to-earnings ratio (trailing twelve months) does {r} trade at?",
        ], "ratio")

    # ------------------------------------------------------------ T2 comparison

    def _returns(self, tb: ToolBox, tickers: list[str], d1: date, d2: date) -> dict[str, float]:
        out = {}
        for t in tickers:
            rows = self.closes(tb, t, d1, d2)
            out[t] = (rows[d2.isoformat()]["close"] / rows[d1.isoformat()]["close"] - 1) * 100
        return out

    def t2_best_return(self):
        k = self.rng.choice([3, 3, 4])
        tickers = self.same_sector(k)
        d1, d2 = self.span(5, 30)
        best = self.rng.random() < 0.6
        tb = self.new_toolbox()
        rets = self._returns(tb, tickers, d1, d2)
        ranked = sorted(rets, key=rets.get, reverse=best)
        if abs(rets[ranked[0]] - rets[ranked[1]]) < 0.5:  # avoid near-ties
            return None
        refs = [self.ref(t, allow_name_only=False)[0] for t in tickers]
        listed = ", ".join(refs[:-1]) + f", and {refs[-1]}"
        how = "the largest percentage gain" if best else "the worst percentage performance"
        text = f"Which of {listed} had {how} from the close on {self.fmt(d1)} to the close on {self.fmt(d2)}?"
        return self.question("t2_best_return", text, "ticker", ranked[0],
                             required_data=[{"ticker": t, "dates": [d1.isoformat(), d2.isoformat()]} for t in tickers],
                             gold_calls=self.calls(tb),
                             meta={"tickers": tickers, "returns_pct": {t: round(v, 4) for t, v in rets.items()},
                                   "direction": "best" if best else "worst"})

    def t2_higher_close(self):
        a, b = self.same_sector(2)
        d = self.day()
        tb = self.new_toolbox()
        ca, cb = self.closes(tb, a, d, d)[d.isoformat()]["close"], self.closes(tb, b, d, d)[d.isoformat()]["close"]
        if ca == cb:
            return None
        ra, rb = self.ref(a, False)[0], self.ref(b, False)[0]
        text = self.rng.choice([
            "Which stock had the higher closing price on {d}: {a} or {b}?",
            "On {d}, did {a} or {b} close at a higher share price?",
        ]).format(a=ra, b=rb, d=self.fmt(d))
        return self.question("t2_higher_close", text, "ticker", a if ca > cb else b,
                             required_data=[{"ticker": t, "dates": [d.isoformat()]} for t in (a, b)],
                             gold_calls=self.calls(tb), meta={"tickers": [a, b], "closes": {a: ca, b: cb}})

    def t2_most_volume(self):
        tickers = self.same_sector(4)
        d = self.day()
        tb = self.new_toolbox()
        vols = {t: self.closes(tb, t, d, d)[d.isoformat()]["volume"] for t in tickers}
        ranked = sorted(vols, key=vols.get, reverse=True)
        if vols[ranked[0]] < 1.05 * vols[ranked[1]]:
            return None
        refs = [self.ref(t, False)[0] for t in tickers]
        text = f"Among {', '.join(refs[:-1])}, and {refs[-1]}, which had the highest trading volume on {self.fmt(d)}?"
        return self.question("t2_most_volume", text, "ticker", ranked[0],
                             required_data=[{"ticker": t, "dates": [d.isoformat()]} for t in tickers],
                             gold_calls=self.calls(tb), meta={"tickers": tickers, "volumes": vols})

    def t2_close_diff(self):
        a, b = self.same_sector(2)
        d = self.day()
        tb = self.new_toolbox()
        ca, cb = self.closes(tb, a, d, d)[d.isoformat()]["close"], self.closes(tb, b, d, d)[d.isoformat()]["close"]
        if ca < cb:
            a, b, ca, cb = b, a, cb, ca
        if ca == cb:
            return None
        ra, rb = self.ref(a, False)[0], self.ref(b, False)[0]
        text = f"On {self.fmt(d)}, how many dollars higher was the closing price of {ra} than that of {rb}?"
        return self.question("t2_close_diff", text, "number", round(ca - cb, 2), unit="USD",
                             tolerance={"abs": 0.02, "rel": 0.0},
                             required_data=[{"ticker": t, "dates": [d.isoformat()]} for t in (a, b)],
                             gold_calls=self.calls(tb), meta={"tickers": [a, b], "closes": {a: ca, b: cb}})

    def t2_return_gap(self):
        a, b = self.same_sector(2)
        d1, d2 = self.span(5, 30)
        tb = self.new_toolbox()
        rets = self._returns(tb, [a, b], d1, d2)
        if rets[a] < rets[b]:
            a, b = b, a
        gap = rets[a] - rets[b]
        if gap < 0.3:
            return None
        ra, rb = self.ref(a, False)[0], self.ref(b, False)[0]
        text = (f"By how many percentage points did {ra} outperform {rb} from the close on {self.fmt(d1)} "
                f"to the close on {self.fmt(d2)}?")
        return self.question("t2_return_gap", text, "number", round(gap, 2), unit="percentage points",
                             tolerance={"abs": 0.1, "rel": 0.0},
                             required_data=[{"ticker": t, "dates": [d1.isoformat(), d2.isoformat()]} for t in (a, b)],
                             gold_calls=self.calls(tb),
                             meta={"tickers": [a, b], "returns_pct": {t: round(v, 4) for t, v in rets.items()}})

    # ----------------------------------------------------------- T3 computation

    def t3_pct_change(self):
        t = self.ticker()
        d1, d2 = self.span(5, 40)
        r, style = self.ref(t)
        tb = self.new_toolbox()
        rows = self.closes(tb, t, d1, d2)
        pct = (rows[d2.isoformat()]["close"] / rows[d1.isoformat()]["close"] - 1) * 100
        text = self.rng.choice([
            "By what percentage did the closing price of {r} change from {d1} to {d2}?",
            "What was the percent change in {r} stock between the close on {d1} and the close on {d2}?",
        ]).format(r=r, d1=self.fmt(d1), d2=self.fmt(d2))
        return self.question("t3_pct_change", text, "number", round(pct, 2), unit="percent", tolerance=TOL_PERCENT,
                             required_data=[{"ticker": t, "dates": [d1.isoformat(), d2.isoformat()]}],
                             gold_calls=self.calls(tb), meta={"tickers": [t], "ref_style": style})

    def t3_avg_close(self):
        t = self.ticker()
        n = self.rng.choice([5, 10, 20])
        end = self.rng.choice(self.window[n:])
        days = self.m.trading_days_ending(end, n)
        r, style = self.ref(t)
        tb = self.new_toolbox()
        rows = self.closes(tb, t, days[0], days[-1])
        avg = sum(rows[d.isoformat()]["close"] for d in days) / n
        text = f"What was the average closing price of {r} over the {n} trading days ending on {self.fmt(end)}?"
        return self.question("t3_avg_close", text, "number", round(avg, 4), unit="USD", tolerance=TOL_DERIVED,
                             required_data=[{"ticker": t, "dates": [d.isoformat() for d in days]}],
                             gold_calls=self.calls(tb), meta={"tickers": [t], "ref_style": style, "n_days": n})

    def t3_max_close(self):
        t = self.ticker()
        d1, d2 = self.span(10, 40)
        r, style = self.ref(t)
        tb = self.new_toolbox()
        rows = self.closes(tb, t, d1, d2)
        text = f"What was the highest closing price of {r} between {self.fmt(d1)} and {self.fmt(d2)}?"
        return self.question("t3_max_close", text, "number", max(x["close"] for x in rows.values()), unit="USD",
                             tolerance=TOL_EXACT_PRICE,
                             required_data=[{"ticker": t, "dates": sorted(rows)}],
                             gold_calls=self.calls(tb), meta={"tickers": [t], "ref_style": style})

    def t3_min_close_date(self):
        t = self.ticker()
        d1, d2 = self.span(10, 40)
        r, style = self.ref(t)
        tb = self.new_toolbox()
        rows = self.closes(tb, t, d1, d2)
        lowest = min(x["close"] for x in rows.values())
        dates = [d for d, x in rows.items() if x["close"] == lowest]
        if len(dates) != 1:
            return None
        text = f"On which date between {self.fmt(d1)} and {self.fmt(d2)} did {r} record its lowest closing price?"
        return self.question("t3_min_close_date", text, "date", dates[0],
                             required_data=[{"ticker": t, "dates": sorted(rows)}],
                             gold_calls=self.calls(tb), meta={"tickers": [t], "ref_style": style, "min_close": lowest})

    def t3_total_volume(self):
        t = self.ticker()
        d1, d2 = self.span(5, 20)
        r, style = self.ref(t)
        tb = self.new_toolbox()
        rows = self.closes(tb, t, d1, d2)
        text = f"What was the total number of {r} shares traded from {self.fmt(d1)} through {self.fmt(d2)}?"
        return self.question("t3_total_volume", text, "number", sum(x["volume"] for x in rows.values()),
                             unit="shares", tolerance=TOL_DERIVED,
                             required_data=[{"ticker": t, "dates": sorted(rows)}],
                             gold_calls=self.calls(tb), meta={"tickers": [t], "ref_style": style, "n_days": len(rows)})

    # ---------------------------------------------------------- T5 unanswerable

    def _abstain(self, template: str, text: str, reason: str, **meta: Any):
        return self.question(template, text, "abstain", "UNANSWERABLE", required_data=[], gold_calls=[],
                             meta={"reason": reason, **meta})

    def t5_future(self):
        t = self.ticker()
        future = self.m.as_of + timedelta(days=self.rng.randint(3, 60))
        while future.weekday() >= 5:
            future += timedelta(days=1)
        r, _ = self.ref(t)
        text = self.rng.choice([
            "What will the closing price of {r} be on {d}?",
            "What is {r} going to close at on {d}?",
            "How much will {r} stock be worth at the close on {d}?",
        ]).format(r=r, d=self.fmt(future))
        return self._abstain("t5_future", text, "future_date", tickers=[t], date=future.isoformat())

    def t5_fake_ticker(self):
        letters = string.ascii_uppercase
        while True:
            fake = "".join(self.rng.choice(letters) for _ in range(self.rng.choice([3, 4])))
            if not self.m.has_ticker(fake) and any(c in fake for c in "QXZJ"):
                break
        d = self.day()
        text = self.rng.choice([
            "What was the closing price of {t} on {d}?",
            "How many shares of {t} traded on {d}?",
            "What is the latest closing price of the stock with ticker symbol {t}?",
        ]).format(t=fake, d=self.fmt(d))
        return self._abstain("t5_fake_ticker", text, "unknown_ticker", tickers=[fake])

    def t5_non_trading_day(self):
        t = self.ticker()
        candidates = [NON_TRADING_WINDOW_START + timedelta(days=i)
                      for i in range((self.m.as_of - NON_TRADING_WINDOW_START).days)]
        closed = [d for d in candidates if not self.m.is_trading_day(d)]
        holidays = [d for d in closed if d.weekday() < 5]
        d = self.rng.choice(holidays) if self.rng.random() < 0.25 else self.rng.choice(closed)
        r, _ = self.ref(t)
        text = self.rng.choice([
            "What was the closing price of {r} on {d}?",
            "How many shares of {r} were traded on {d}?",
        ]).format(r=r, d=self.fmt(d))
        return self._abstain("t5_non_trading_day", text, "market_closed", tickers=[t], date=d.isoformat(),
                             holiday=d.weekday() < 5)

    def t5_advice(self):
        a, b = self.same_sector(2)
        ra, rb = self.ref(a)[0], self.ref(b)[0]
        text = self.rng.choice([
            "Should I buy {a} stock right now?",
            "Is {a} a good investment for my retirement savings?",
            "Which is the better buy right now, {a} or {b}?",
            "Should I sell my {a} shares before next week?",
            "How much of my portfolio should I put into {a}?",
        ]).format(a=ra, b=rb)
        return self._abstain("t5_advice", text, "investment_advice", tickers=[a, b] if "{b}" in text else [a])

    # ------------------------------------------------------------------ driver

    def generate(self, counts: dict[str, int] = COUNTS) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for template, n in counts.items():
            make: Callable[[], dict | None] = getattr(self, template)
            made, attempts = 0, 0
            while made < n:
                attempts += 1
                if attempts > 50 * n:
                    raise RuntimeError(f"Could not generate {n} questions for {template}")
                q = make()
                if q is None or q["question"] in seen:
                    continue
                seen.add(q["question"])
                made += 1
                q["id"] = f"{q['tier']}-{template[3:]}-{made:03d}"
                out.append(q)
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot-dir", default=str(DEFAULT_SNAPSHOT))
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "questions_v1.jsonl"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    market = SnapshotMarket(args.snapshot_dir)
    gen = Generator(market, seed=args.seed)
    questions = gen.generate()

    key_order = ["id", "tier", "template", "question", "answer_type", "gold", "unit", "tolerance",
                 "required_data", "gold_calls", "meta"]
    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps({k: q[k] for k in key_order if k in q}, ensure_ascii=False) + "\n")

    meta = {
        "snapshot_dir": Path(args.snapshot_dir).name,
        "as_of": market.as_of.isoformat(),
        "as_of_text": market.as_of_text,
        "seed": args.seed,
        "question_window": [QUESTION_WINDOW_START.isoformat(), market.as_of.isoformat()],
        "n_questions": len(questions),
        "by_tier": dict(Counter(q["tier"] for q in questions)),
        "by_template": dict(Counter(q["template"] for q in questions)),
        "excluded_tickers": gen.excluded,
        "tolerance_rule": "numeric answer correct iff |pred - gold| <= max(abs, rel * |gold|)",
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {len(questions)} questions to {out_path}")
    print(json.dumps(meta["by_tier"]), "| excluded tickers:", len(gen.excluded))


if __name__ == "__main__":
    main()
