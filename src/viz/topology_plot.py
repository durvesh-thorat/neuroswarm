"""Render evolved NEAT neural network topologies with excitatory, inhibitory, and recurrent edges."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

from src.evolution.neat_core import Genome, load_genome_json

INPUT_LABELS = [
    "Wall N", "Wall S", "Wall E", "Wall W",
    "Vis N", "Vis S", "Vis E", "Vis W",
    "Mate dX", "Mate dY", "Mate Prox",
    "Comm 0", "Comm 1",
    "Goal Cos", "Goal Sin",
]

OUTPUT_LABELS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY", "TX 0", "TX 1"]


def plot_genome_topology(
    genome: Genome,
    title: str = "Evolved NEAT Topology",
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Draw input, hidden, and output nodes with weighted connections (green=+, red=-) and recurrent loops."""
    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=150)
    else:
        fig = ax.figure

    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")

    input_keys = genome.input_keys
    output_keys = genome.output_keys
    hidden_keys = sorted(genome.hidden_keys)

    coords: dict[int, tuple[float, float]] = {}
    # Left column: inputs
    ys_in = np.linspace(0.95, 0.05, len(input_keys))
    for k, y in zip(input_keys, ys_in):
        coords[k] = (0.12, float(y))

    # Right column: outputs
    ys_out = np.linspace(0.88, 0.12, len(output_keys))
    for k, y in zip(output_keys, ys_out):
        coords[k] = (0.88, float(y))

    # Middle column(s): hidden nodes
    if hidden_keys:
        ys_hid = np.linspace(0.84, 0.16, len(hidden_keys)) if len(hidden_keys) > 1 else [0.5]
        for idx, (k, y) in enumerate(zip(hidden_keys, ys_hid)):
            x_jitter = 0.45 + 0.12 * ((idx % 2) - 0.5) if len(hidden_keys) > 3 else 0.50
            coords[k] = (x_jitter, float(y))

    # Draw enabled connections
    recurrent_count = 0
    for c in genome.connections.values():
        if not c.enabled or c.in_node not in coords or c.out_node not in coords:
            continue
        x1, y1 = coords[c.in_node]
        x2, y2 = coords[c.out_node]
        color = "#22c55e" if c.weight >= 0 else "#ef4444"
        lw = float(np.clip(0.5 + abs(c.weight) * 0.75, 0.6, 3.2))
        alpha = float(np.clip(0.32 + abs(c.weight) * 0.16, 0.35, 0.90))

        is_recurrent = (c.in_node == c.out_node) or (x1 >= x2)
        if is_recurrent:
            recurrent_count += 1

        if c.in_node == c.out_node:
            # Self-recurrent loop
            loop = FancyArrowPatch(
                (x1 - 0.015, y1 + 0.02),
                (x1 + 0.015, y1 + 0.02),
                connectionstyle="arc3,rad=-2.4",
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=lw,
                color="#f59e0b",
                alpha=0.9,
                zorder=3,
            )
            ax.add_patch(loop)
        elif x1 >= x2:
            # Feedback recurrent edge (dashed amber/red/green)
            arrow = FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                connectionstyle="arc3,rad=0.28",
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=lw,
                linestyle="--",
                color="#f59e0b",
                alpha=0.85,
                zorder=2,
            )
            ax.add_patch(arrow)
        else:
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=alpha, zorder=1)

    # Draw nodes
    for idx, k in enumerate(input_keys):
        x, y = coords[k]
        ax.scatter([x], [y], s=130, color="#38bdf8", edgecolors="#e2e8f0", linewidths=1.0, zorder=4)
        label = INPUT_LABELS[idx] if idx < len(INPUT_LABELS) else f"I{idx}"
        ax.text(x - 0.025, y, label, color="#cbd5e1", fontsize=7.5, ha="right", va="center", family="monospace")

    for k in hidden_keys:
        x, y = coords[k]
        ax.scatter([x], [y], s=170, color="#a855f7", edgecolors="#f8fafc", linewidths=1.2, zorder=5)
        ax.text(x, y, f"H{k}", color="#ffffff", fontsize=6.5, ha="center", va="center", fontweight="bold", zorder=6)

    for idx, k in enumerate(output_keys):
        x, y = coords[k]
        col = "#f97316" if idx >= 5 else "#10b981"
        ax.scatter([x], [y], s=155, color=col, edgecolors="#f8fafc", linewidths=1.1, zorder=4)
        label = OUTPUT_LABELS[idx] if idx < len(OUTPUT_LABELS) else f"O{idx}"
        ax.text(x + 0.025, y, label, color="#f1f5f9", fontsize=8.0, ha="left", va="center", fontweight="bold", family="monospace")

    subtitle = (
        f"Nodes: {len(genome.nodes)} ({len(hidden_keys)} hidden) | "
        f"Enabled Conns: {genome.num_enabled_connections} | "
        f"Recurrent: {recurrent_count}"
    )
    ax.set_title(f"{title}\n{subtitle}", color="#f8fafc", fontsize=10.5, fontweight="bold", pad=10)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.02, 1.04)
    ax.axis("off")

    if created_fig and save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)
    return fig


def plot_champion_file(genome_path: str | Path, save_path: str | Path, title: str | None = None) -> None:
    genome, meta = load_genome_json(genome_path)
    t = title or f"{meta.get('run_id', 'Champion')} ({meta.get('topology', 'NEAT')})"
    plot_genome_topology(genome, title=t, save_path=save_path)
