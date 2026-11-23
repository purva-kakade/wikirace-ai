"""Benchmark charts (requires matplotlib)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
BASELINE = "#c3c2b7"
BAR = "#2a78d6"


def make_charts(report: dict[str, Any], rows: Sequence[dict[str, Any]], path: str | Path) -> list[Path]:
    """Draw one figure of small multiples comparing the agents to ``path`` (a .png). Returns the files written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    agents = [r["agent"] for r in rows]
    n_scenarios = len({(g["start"], g["target"]) for g in report["games"]})

    panels = [
        ("Success rate", [r["success_rate"] * 100 for r in rows], lambda v: f"{v:.0f}%", (0, 100)),
        (
            "Extra clicks vs. shortest path\n(games won; lower is better)",
            [r["mean_extra_clicks"] for r in rows],
            lambda v: f"+{v:.1f}",
            None,
        ),
        ("Seconds per game", [r["mean_seconds"] for r in rows], lambda v: f"{v:.1f}s", None),
    ]

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
    fig, axes = plt.subplots(1, len(panels), figsize=(12, 0.5 * len(agents) + 1.9), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    y = list(range(len(agents)))

    for ax, (title, values, label, xlim) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        plotted = [v if v is not None else 0 for v in values]
        ax.barh(y, plotted, height=0.55, color=BAR)
        top = max(plotted) if any(plotted) else 1
        ax.set_xlim(*(xlim or (0, top * 1.25)))
        for yi, value in zip(y, values):
            text = "n/a" if value is None else label(value)
            ax.text((value or 0) + (ax.get_xlim()[1] * 0.02), yi, text, va="center", color=INK_SECONDARY)
        ax.set_title(title, loc="left", color=INK, fontsize=10.5, pad=10)
        ax.set_yticks(y, [f"{a}" for a in agents], color=INK)
        ax.invert_yaxis()
        ax.set_xticks([])
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(BASELINE)
        ax.tick_params(axis="y", length=0, pad=8)

    fig.suptitle(
        f"Wikipedia Game agents on {n_scenarios} scenarios",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=13,
        fontweight="bold",
    )
    model = report.get("meta", {}).get("llm_model")
    if model:
        fig.text(0.01, 0.005, f"LLM agents use {model} via Ollama", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))

    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return [path]
