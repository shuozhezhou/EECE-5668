"""Figures for the mid-term report.

Reads results/error_labels.csv (one hand-checked primary label per failed episode)
and writes midterm/figures/failures.pdf. Run from the repo root:
    py analysis/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "midterm" / "figures"

COLORS = {"C1": "#2a78d6", "C2": "#eb6834"}  # categorical slots 1-2
LABELS = {
    "malformed_tool_call": "Malformed tool call",
    "wrong_selection": "Wrong max/min/rank",
    "arithmetic_error": "Arithmetic error",
    "window_definition": "Trading vs. calendar days",
    "wrong_formula": "Wrong formula",
    "wrong_tool": "Wrong tool",
    "misread_tool_output": "Misread tool output",
    "phantom_tool_result": "Phantom Python result",
    "tool_loop_max_rounds": "Ran out of tool rounds",
    "over_refusal": "Over-refusal",
    "wrong_entity_or_range": "Wrong ticker or range",
}


def failures_figure(labels: pd.DataFrame) -> None:
    counts = (labels.groupby(["primary", "condition"]).size().unstack("condition")
              .reindex(columns=["C1", "C2"]).fillna(0).astype(int))
    counts = counts.loc[counts.sum(axis=1).sort_values().index]  # largest at the top

    fig, ax = plt.subplots(figsize=(6.8, 2.9))
    y = range(len(counts))
    h = 0.38
    for k, cond in enumerate(["C1", "C2"]):
        pos = [i + (0.5 - k) * h for i in y]
        bars = ax.barh(pos, counts[cond], height=h - 0.04, color=COLORS[cond],
                       label=f"{cond} ({counts[cond].sum()} failures)")
        for b, v in zip(bars, counts[cond]):
            if v:
                ax.text(b.get_width() + 0.3, b.get_y() + b.get_height() / 2, str(v),
                        va="center", fontsize=6.5, color="#333333")
    ax.set_yticks(list(y), [LABELS.get(p, p) for p in counts.index], fontsize=7.5)
    ax.set_xlabel("Failed answerable questions (of 150)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.tight_layout(pad=0.3)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "failures.pdf")
    fig.savefig(OUT / "failures.png", dpi=200)
    print(f"Wrote {OUT / 'failures.pdf'}")


if __name__ == "__main__":
    failures_figure(pd.read_csv(REPO / "results" / "error_labels.csv"))
