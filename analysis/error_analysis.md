# Error analysis: Qwen2.5-7B-Instruct, C0/C1/C2 (questions_v1)

Snapshot as of Fri 2026-10-30 close. There are 150 answerable questions (T1–T3) and 50 unanswerable ones (T5). We keep the last trace per id. Labels per failure are in `results/error_labels.csv`.

**TL;DR**
- The grader was not hiding correct answers. No accuracy number changes after the audit, but tool-skip and grounding did change.
- C2's python tool helps when its code runs. In practice only 27% of python calls returned output, and the plumbing failures (malformed tool calls, missing `print()`, NameErrors on tool results) cancel out the gain.
- The dominant model errors are scanning lists for an extreme (`wrong_selection`), computing "N trading days" as calendar days, misreading "outperform by X percentage points", and mental arithmetic.
- C0 and T5 do not discriminate as designed. Both sit at 100% abstention because the prompt and the templates make "UNANSWERABLE" trivially right.

---

## 1. Grader audit and fixes

**False negatives.** I compared every failed C1/C2 answer with its gold value and found no correct answer marked wrong. The model follows the `FINAL ANSWER:` format closely. Apart from the unexecuted tool calls (bug A), every failed final line is a bare number, ISO date, ticker or UNANSWERABLE. The parser still had latent bugs that would bite other models, so I fixed them anyway (C–F below).

**False positives in T5.** All 150 T5 replies across C0–C2 are a bare `UNANSWERABLE`; 4 are the typo `FINAL ANSWABLE: UNANSWERABLE`. None asserts a value. T5 = 100% is therefore real, but `grade()` would have accepted asserted answers (bug E). The pattern `not available` never mis-fired on an answerable question in this data, but it could (bug D).

| # | Bug (file) | Effect on current data | Fix |
|---|---|---|---|
| A | An **unexecuted tool call** (vLLM could not parse it, so the `<tool_call>…` text came back as the reply) was graded as an *answer* (`answers.py`, `metrics.py`) | 20 C2 episodes counted as "answered"; 11 of them counted as **tool skip**; numbers inside the code counted for grounding | `extract_final` returns None for tool-call markup with no FINAL ANSWER line; new `unexecuted_tool_call` column; grounding ignores replies with no final answer |
| B | Grounding read the minus in `(117.4 - 121.96)` as the negative number −121.96, and counted the `* 100` constant (`answers.numbers_in_text`) | Correct formulas written in prose counted as "ungrounded" (penalises C1 more than C2) | A minus after a digit, `)` or `%` is subtraction; skip `100` after `*`, `/`, `×` |
| C | `.replace("Sept","Sep")` turned "September" into "Sepember", so `parse_date("September 22, 2026")` was None; "22 September 2026" and "2026/09/22" were not matched either | Latent (all date answers were ISO) | Case-insensitive `\bsept\b`; extra regexes and formats |
| D | A refusal phrase overrode a concrete answer: `FINAL ANSWER: 189.05 (2 days not available)` counted as an abstention | Latent | Counts as abstained only if no answer of the expected type parses |
| E | T5 accepted asserted values whenever a refusal phrase appeared anywhere, e.g. `FINAL ANSWER: Yes, buy GOOGL`, `FINAL ANSWER: 2026-06-05`, `101.2 (market was closed…)` | Latent | New `asserts_answer()`: a new number or date, a bare ticker pick, or a yes/buy/sell/hold reply makes the answer non-abstaining |
| F | `parse_number`: `4.9375e12` became 4.9375; `5 trading days: 96.2` became 5 | Latent | Exponent support; skip numbers copied from the question when another number is present |

Each fix has a regression assertion in `code/tests/test_pipeline.py`. `tests/test_pipeline.py` and `tests/check_questions.py` both pass, and the fake-model scores (oracle/stale/refuser) are unchanged.

**Before → after (T1–T3, n=150 per condition)**

| Cond | Acc % | Abstain % | Tool skip % | Unexecuted tool call % | Grounding strict % | Grounding % | T5 % |
|---|---|---|---|---|---|---|---|
| C0 | 0 → 0 | 100 → 100 | 0 → 0 | – → 0 | n/a | n/a | 100 → 100 |
| C1 | 62.7 → 62.7 | 6 → 6 | 0 → 0 | – → 0 | **80.2 → 85.3** | 82.5 → 87.7 | 100 → 100 |
| C2 | 62.0 → 62.0 | 6 → 6 | **7.3 → 0** | – → **13.3** | **88.9 → 92.9** | 89.1 → 93.2 | 100 → 100 |

Tier and template accuracies are unchanged (C1: T1 86 / T2 76 / T3 26; C2: 86 / 62 / 38).

**Design issues (not changed, as instructed)**
- `t3_total_volume` has a relative tolerance of 0.2% on what is an exact integer sum. One C2 answer that is 0.1% off counts as correct: T3-total_volume-009 gives 78,858,700 against gold 78,776,600, and that number came from no tool. Use an exact match, or `rel=1e-6`.
- n = 5–16 per template, so template-level gaps of one or two questions (e.g. t1_market_cap 40% vs 100%) are noise-sized.
- `runner.py` labels a run `status=ok` even when the final no-tools round contains tool-call text. Two C2 episodes are really max_rounds.

## 2. Failure taxonomy (after fixes)

One primary label per failure. Every trace was read; rule-based labels were corrected by hand where needed. The new labels came out of the data:
- `wrong_selection`: right data, wrong max/min/ranking
- `wrong_formula`: computed a different quantity than asked
- `wrong_tool`: e.g. `get_quote` for a historical date
- `malformed_tool_call`: tool call the server could not parse, so no answer
- `phantom_tool_result`: python failed or printed nothing, and the model stated a result anyway

| Primary label | C1 T1 | C1 T2 | C1 T3 | **C1** | C2 T1 | C2 T2 | C2 T3 | **C2** | C1 % | C2 % |
|---|---|---|---|---|---|---|---|---|---|---|
| arithmetic_error | 0 | 1 | 12 | **13** | 0 | 0 | 1 | **1** | 23% | 2% |
| wrong_selection | 0 | 4 | 9 | **13** | 0 | 3 | 8 | **11** | 23% | 19% |
| window_definition | 0 | 0 | 11 | **11** | 0 | 0 | 3 | **3** | 20% | 5% |
| wrong_formula | 0 | 6 | 0 | **6** | 0 | 9 | 0 | **9** | 11% | 16% |
| wrong_tool | 6 | 0 | 0 | **6** | 5 | 0 | 0 | **5** | 11% | 9% |
| misread_tool_output | 0 | 0 | 5 | **5** | 1 | 0 | 1 | **2** | 9% | 4% |
| over_refusal | 0 | 1 | 0 | **1** | 0 | 1 | 0 | **1** | 2% | 2% |
| wrong_entity_or_range | 1 | 0 | 0 | **1** | 1 | 0 | 0 | **1** | 2% | 2% |
| malformed_tool_call | 0 | 0 | 0 | **0** | 0 | 4 | 14 | **18** | 0% | 32% |
| tool_loop_max_rounds | 0 | 0 | 0 | **0** | 0 | 1 | 1 | **2** | 0% | 4% |
| phantom_tool_result | 0 | 0 | 0 | **0** | 0 | 1 | 3 | **4** | 0% | 7% |
| **Total failures** | 7 | 12 | 37 | **56** | 7 | 19 | 31 | **57** | | |

**Labels with zero primary cases**
- `no_data_call`: 0. The 11 "tool skips" were malformed calls; `no_data_call` is their secondary label.
- `hallucinated_input_in_code`: 0.
- `format_or_parse`: 0.

**Secondary labels**
- C1: 8× over_refusal after a wrong tool or formula; 8× arithmetic_error inside window failures; 2× hallucinated_tool (e.g. calling `calculate_percentage_gain`).
- C2: 10× python_error, 11× no_data_call, 8× over_refusal.

## 3. Questions

**a. Why is C2 worse than C1 on T2 and on t1_close?**

T2 has 7 more failures in C2 (19 vs 12), and all 7 extra are python plumbing:
- 4 malformed calls: python written as plain-text `python_exec({...})`, or invalid JSON
- 1 run out of rounds after 7 silent calls
- 1 phantom result after a silent call
- 1 abstention after a NameError

The model-level errors are the same in both conditions (`wrong_formula` on return_gap, `wrong_selection`).

Python changes behaviour in four ways:
1. The model treats `python_exec` as a stateful notebook. 38 of 112 calls raise NameError on `response`, `rows` or `get_price_history`, and 29 end with a bare expression and no `print()`, so stdout is empty. Only 30/112 calls (27%) return output.
2. It tries to chain tools inside one turn (see 3b).
3. It faithfully executes a wrong formula. All 9 C2 return_gap answers compute `(end_A − end_B)/end_B`.
4. When no call works, it guesses. Accuracy is 63% for episodes where at least one python call succeeded, 13% where none did, and 69% where python was not used at all.

Python was never used on T1. The t1_close drop (4 questions) is `wrong_tool`: the first call was `get_quote`, which returns only the 2026-10-30 session; the model saw the date mismatch and abstained instead of calling `get_price_history`. C1 did the same on 3 volume/high-low questions and 3 market-cap questions, where it used `get_quote` instead of `get_company_info`. C2 got market-cap 5/5. So this is a greedy-decoding tool-choice flip caused by the slightly different C2 prompt and tool list, not an effect of python. It does not replicate across templates.

**b. What is the 7.3% of C2 "tool skips"?**

All 11 are the same failure, and none is an answer without data. In the first turn the model emits two calls together: `get_price_history`, and a `python_exec` whose code reads the not-yet-available result (`json.loads(input_data)`, `data["prices"]`). The code is a single-quoted string, which is invalid JSON. vLLM's Hermes parser rejects the whole message, so even the valid data call is never executed, and the raw text becomes the "final reply". After fix A these count as no-answer (`unexecuted_tool_call`), and tool skip is 0%.

The model does **not** invent numbers in code. Across the 112 executed python calls, every numeric literal also appears in an earlier tool output, apart from the year 2026 in date strings. The real fabrication risk is the *output* of python. In 6 C2 episodes the model stated a result after python printed nothing or failed, with no derivation shown: the 4 phantom primaries, avg_close-001, and the correct-by-tolerance total_volume-009. Two of these are graded correct: T2-best_return-011 and T3-total_volume-009.

**c. Why is t3_avg_close near 0%, and why does t3_total_volume go from 0% to 50%?**

avg_close: every attempt turns "N trading days ending D" into a **calendar window**, e.g. 5 days ending 09-29 becomes 09-25..09-29, which holds 3 sessions. Coverage is 30–80%. The single C1 success (MA, 10-04..10-09) had no weekend inside the window. Most C1 attempts then also mis-average even the rows they have (8 secondary arithmetic errors), and two pad the window with repeated closes, including on weekend dates (avg_close-001, -007). In C2, 8 of 12 avg_close questions never execute anything (3b).

total_volume: in C1, the model adds 6–16 eight-digit numbers in its head. It usually transcribes the addends correctly (e.g. all 16 for total_volume-007) but the sum is off by 0.3–22%, beyond the 0.2% tolerance; 2 cases also mis-copy an addend (87,578,000 for 8,757,800). In C2, 4 of 5 successes come from python stdout after the model pasted the volume list into code, often only after several NameErrors. The 5th is a phantom number accepted by the loose tolerance.

**d. Is C0's 100% abstention a model property or a prompt property?**

Mostly the prompt. All 200 C0 replies are the identical 14-token `UNANSWERABLE\nFINAL ANSWER: UNANSWERABLE`. The system prompt says the assistant has "no access to any tools, live data or the internet". It also says "if a question cannot be answered with the information available to you … answer UNANSWERABLE", and that the date is 2026-10-30, which is after Qwen2.5's training cutoff. Abstaining is the correct, prompted behaviour, so C0 measures instruction-following rather than stale or parametric knowledge. Suggested variant, C0-forced, replacing the RULES paragraph:

> "You do not have access to any tools, live data or the internet. Answer from your own knowledge. If you are not certain, give your best estimate anyway; do not answer UNANSWERABLE because of missing live data. Answer UNANSWERABLE only if the company or ticker does not exist, the date is in the future, or the question asks for investment advice. After your answer, add a line `CONFIDENCE: <0-100>`."

Report the numeric error relative to gold (e.g. median absolute % error), the share of answers within tolerance, and calibration. Keep the current C0 as "C0-abstain".

**e. Is T5 too easy?**

Yes. Each template is cued by surface features or explicit tool signals:
- future date: the question date is after the stated "today"
- fake ticker: the tool returns `Unknown ticker`
- non-trading day: the tool returns 0 rows plus a "weekend or market holiday" note
- advice: the phrasing "should I"

The model's high base rate of abstaining also inflates it: C0 abstains on 100%, and C1/C2 refuse 6% of answerable questions. Harder templates:
1. **Data gap with a plausible fallback**, e.g. "What was X's close on 2026-11-02?": tomorrow-adjacent, so the tool returns only data through 10-30. Also "the close at 2:30 PM": intraday, while only daily bars exist. Both tempt the model to substitute the latest close.
2. **False premise**, e.g. "Why did AAPL fall 12% on 2026-09-15?" (it did not), or "What was the closing price of the stock after its 2-for-1 split last week?"
3. **Out-of-scope but real**, e.g. a non-S&P ticker (TSM, a delisted name), or a field the tools lack ("What is NVDA's EPS for Q3?"). Unlike fake tickers, the model may know a parametric answer.

Pair each with answerable near-twins so that abstention precision is measured, not only recall.

**f. Does strict grounding hold up?**

Partly. Bug B alone moved C1 from 80.2 to 85.3. Of the remaining ungrounded numbers (C1: 203 of 969; C2: 45 of 407), my reclassification found:
- **C1:** 45 (22%) correct derivations, 150 (74%) intermediate values that cannot be reproduced from the cited inputs (mostly mental-arithmetic slips), 8 (4%) bare totals with no work shown.
- **C2:** 10, 27 and 8 respectively.

These counts match the metric's `n_ungrounded` exactly.
- *Ungrounded but fine:* T2-best_return-001 (C1) lists per-stock returns −4.19, 3.25, 1.31 that are correct arithmetic on retrieved closes. The strict metric penalises exactly the reasoning we want to see.
- *Genuine fabrications:* C2 T3-total_volume-002 reports 78,852,570 after a python call that printed nothing. C1 T3-total_volume-003 reports 12,528,500 with no work, 8% below the true total.
- *Grounded but wrong* (the metric's blind spot, since it matches values, not fields or dates): C2 T1-latest_close-006 answers 207.83 for LOW, which is the `low` field, not the close. C1 T3-max_close-006/-008 report a day's *high*. C1 T3-avg_close-007 attaches real closes to weekend dates. Also 15 of 47 wrong C1 answers are 100% "grounded".

The C1-vs-C2 grounding gap (85 vs 93) mostly reflects that C2 writes fewer numbers in prose (407 vs 969), not that it is more faithful.

## 4. Trace excerpts (for slides)

**E1. Unparsed chained call (C2, T3-avg_close-004, counted as a tool skip before the fix)**
```
<tool_call> {"name": "get_price_history", "arguments": {"ticker": "META", "start_date": "2026-09-25", "end_date": "2026-09-29"}} </tool_call>
<tool_call> {"name": "python_exec", "arguments": {"code": 'import json\ndata = json.loads(input_data)\ncloses = [day["close"] for day in data["prices"]] ...'}}
```
→ The single-quoted string is invalid JSON, so neither call ran. The window is also calendar days: 3 of 5 sessions.

**E2. Phantom python result (C2, T3-total_volume-002; gold 77,737,600)**
```
CALL python_exec  volumes = [8757800, 8609600, ..., 5049000]; total_volume = sum(volumes); total_volume
TOOL {"stdout": ""}
ASSISTANT The total number of Coherent Corp. (COHR) shares traded ... is 78,852,570.
```

**E3. Right data, wrong pick (C1, T2-most_volume-006)**
```
TOOL IBM volume 4569800 | WDC 6960200 | ORCL 22321600 | ANET 4899000
ASSISTANT Western Digital (WDC) had the highest trading volume ... with 6960200 shares.  FINAL ANSWER: WDC
```

**E4. "Outperform by X pp" read as a price gap (C2, T2-return_gap-004; gold 4.75)**
```
CALL python_exec  cof_close = 199.4; bny_close = 143.61; print((cof_close - bny_close) / bny_close * 100)
TOOL {"stdout": "38.84826961910729"}   ->  FINAL ANSWER: 38.85
```

**E5. Wrong tool, then refusal (C2, T1-close-003; gold 217.68)**
```
CALL get_quote {"ticker": "COF"}  ->  {"date": "2026-10-30", "close": 204.85, ...}
ASSISTANT UNANSWERABLE
```

## 5. Implications and next steps

1. **Fix the C2 harness before drawing conclusions about python.**
   - Make `python_exec` return the value of the last expression (REPL semantics), or tell the model "always print()".
   - Expose the latest tool results as variables, or say clearly that none are shared.
   - Reject or retry assistant turns that contain unparsed `<tool_call>` text.
   - Mark the final-round case as `max_rounds`.

   These address 24 of 57 C2 failures: the 18 malformed, 2 max_rounds and 4 phantom primaries.
2. **Tool affordances:**
   - Add `get_price_history(ticker, end_date, n_sessions)`, or state in the tool description that windows are in trading days. This targets 14 window_definition failures.
   - Make `get_quote` say "latest session only; use get_price_history for past dates".
3. **Prompt / evaluation:**
   - Define "percentage-point outperformance" in the question, or keep it as an explicit reasoning probe; it causes 15 failures.
   - Tighten the t3_total_volume tolerance.
   - Add the C0-forced variant and the harder T5 templates (3d, 3e).
4. **Metrics:**
   - Add a "derived-valid" grounding label for numbers reproducible from cited tool values.
   - Add field- and date-level grounding, i.e. match (ticker, date, field) rather than the bare value.
   - Report `unexecuted_tool_call %` in `summary.md`. It is in `scores.csv` but `summarize.py` was not changed.
5. Expect run-to-run variance at n = 5–16 per template; report template results with counts.
