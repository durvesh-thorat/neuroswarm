"""Generate honest benchmark charts, causal comm ablation plot, and WallFollower-vs-EXP4 GIF."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.telemetry.replay import load_replay

ASSETS_DIR = Path("assets")
BENCH_DIR = Path("data/benchmarks")

BG_DARK = "#0b1120"
PANEL_BG = "#111827"
CARD_BG = "#1e293b"
BORDER = "#334155"
TEXT_MAIN = "#f8fafc"
TEXT_MUTED = "#94a3b8"


def generate_honest_benchmark_chart() -> None:
    """Plot 80-maze OOD solve rates (with 95% bootstrap CIs) & causal comm ablation."""
    summary = pd.read_csv(BENCH_DIR / "ood_summary.csv")
    comm_df = pd.read_csv(BENCH_DIR / "ood_comm_ablation.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.8), dpi=170)
    fig.patch.set_facecolor(BG_DARK)
    for ax in (ax1, ax2):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_MUTED, labelsize=9)
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.grid(axis="y", color=BORDER, alpha=0.4, linestyle="--", zorder=0)

    # Panel 1: All baselines + evolved models (honest Wall-Follower headline)
    order = [
        "BASELINE-WallFollower-MultiHand",
        "BASELINE-WallFollower",
        "BASELINE-WallFollower-RandTie",
        "EXP-3",
        "EXP-4",
        "EXP-2",
        "EXP-1",
        "BASELINE-GreedyCompass",
        "BASELINE-Random",
    ]
    short_names = [
        "WF\nMultiHand",
        "WF\nSingle",
        "WF\nRandTie",
        "EXP-3\nComm Rec",
        "EXP-4\nHybrid+QD",
        "EXP-2\nComm FF",
        "EXP-1\nSilent FF",
        "Greedy\nCompass",
        "Random\nWalk",
    ]
    colors = ["#22c55e", "#16a34a", "#4ade80", "#a855f7", "#10b981", "#38bdf8", "#64748b", "#f59e0b", "#475569"]
    sub = summary.set_index("Experiment").loc[order]
    means = sub["Solve_Rate_Mean"].values
    lo = np.maximum(0.0, means - sub["Solve_Rate_CI95_Low"].values)
    hi = np.maximum(0.0, sub["Solve_Rate_CI95_High"].values - means)

    x = np.arange(len(order))
    ax1.bar(x, means, color=colors, edgecolor=BG_DARK, lw=0.8, zorder=3)
    ax1.errorbar(x, means, yerr=[lo, hi], fmt="none", ecolor=TEXT_MAIN, elinewidth=1.4, capsize=4, zorder=4)
    for idx, (m, h_err) in enumerate(zip(means, hi)):
        ax1.text(idx, m + h_err + 1.8, f"{m:.1f}%", color=TEXT_MAIN, ha="center", va="bottom", fontsize=8.2, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, color=TEXT_MAIN, fontsize=8.3, fontweight="bold")
    ax1.set_ylim(0, 114)
    ax1.set_ylabel("Zero-Shot OOD Solve Rate (%)", color=TEXT_MAIN, fontweight="bold")
    ax1.set_title("80-Maze OOD Solve Rate (95% Bootstrap CI)\nWall-Follower (99-100%) vs. Evolved NEAT (6-12%)", color=TEXT_MAIN, fontweight="bold", pad=10)

    # Panel 2: Causal Communication Control (Normal vs Zero vs Shuffle)
    exps = ["EXP-2", "EXP-3", "EXP-4"]
    modes = [("normal", "#38bdf8", "Normal Comm"), ("zero", "#64748b", "Zeroed Comm"), ("shuffle", "#f59e0b", "Shuffled Comm")]
    x2 = np.arange(len(exps))
    w = 0.25
    for i, (mode, col, lab) in enumerate(modes):
        vals = []
        for e in exps:
            row = comm_df[(comm_df["experiment_id"] == e) & (comm_df["comm_mode"] == mode)]
            vals.append(float(row["solve_rate_pct"].iloc[0]) if not row.empty else 0.0)
        bars = ax2.bar(x2 + (i - 1) * w, vals, w, label=lab, color=col, edgecolor=BG_DARK, lw=0.8, zorder=3)
        for b, v in zip(bars, vals):
            ax2.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}%", color=TEXT_MAIN, ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(["EXP-2\n(Comm FF)", "EXP-3\n(Comm Rec)", "EXP-4\n(Hybrid+QD)"], color=TEXT_MAIN, fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0, 22)
    ax2.set_ylabel("OOD Solve Rate Across 240 Episodes (%)", color=TEXT_MAIN, fontweight="bold")
    ax2.set_title("Causal Comm Control: Normal vs. Zero vs. Shuffled\nShuffled Noise Improves EXP-3/4 -> Perturbation, Not Language", color=TEXT_MAIN, fontweight="bold", pad=10)
    ax2.legend(facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT_MAIN, fontsize=9)

    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "benchmark_comparison.png", dpi=170, facecolor=BG_DARK)
    plt.close(fig)


def _draw_maze_frame(ax: plt.Axes, replay: dict, frame_idx: int, title: str, accent: str) -> None:
    grid = np.array(replay["maze"]["grid"], dtype=np.uint8)
    h, w = grid.shape
    frames = replay["frames"]
    idx = min(frame_idx, len(frames) - 1)
    frame = frames[idx]
    fog = np.array(frame["fog_mask"], dtype=np.uint8)
    open_cells = max(1, int(np.sum(grid != 1)))
    live_cov = float(np.sum((fog == 1) & (grid != 1))) / float(open_cells) * 100.0

    img = np.zeros((h, w, 3), dtype=np.float32)
    for r in range(h):
        for c in range(w):
            if fog[r, c] == 0:
                img[r, c] = [0.055, 0.082, 0.145]
            elif grid[r, c] == 1:
                img[r, c] = [0.18, 0.24, 0.34]
            elif grid[r, c] == 2:
                img[r, c] = [0.12, 0.32, 0.56]
            elif grid[r, c] == 3:
                img[r, c] = [0.06, 0.68, 0.46]
            else:
                img[r, c] = [0.87, 0.91, 0.96]

    ax.clear()
    ax.imshow(img, interpolation="nearest", extent=(-0.5, w - 0.5, h - 0.5, -0.5))
    gr, gc = replay["maze"]["goal"]
    ax.add_patch(patches.Circle((gc, gr), radius=0.42, fill=False, edgecolor="#10b981", lw=2.0, zorder=4, clip_on=True))

    agent_colors = ["#38bdf8", "#f97316", "#c084fc"]
    comm_enabled = bool(replay["metadata"].get("comm_enabled", False))
    for a in range(len(frame["positions"])):
        t0 = max(0, idx - 18)
        tr = [frames[t]["positions"][a][0] for t in range(t0, idx + 1)]
        tc = [frames[t]["positions"][a][1] for t in range(t0, idx + 1)]
        ax.plot(tc, tr, color=agent_colors[a % 3], lw=2.0, alpha=0.7, zorder=5, clip_on=True)
        r, c = frame["positions"][a]
        ax.scatter([c], [r], s=125, color=agent_colors[a % 3], edgecolors="#ffffff", linewidths=1.3, zorder=6, clip_on=True)
        if comm_enabled:
            b = frame["broadcasts"][a]
            strength = max(abs(b[0]), abs(b[1]))
            if strength >= 0.25:
                ax.add_patch(patches.Circle((c, r), radius=1.2 + strength * 0.9, fill=False, edgecolor="#facc15", lw=1.2, alpha=0.55, linestyle="--", zorder=5, clip_on=True))

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(accent)
        sp.set_linewidth(1.8)

    solved = bool(replay["metrics"]["solved"]) and idx == len(frames) - 1
    status = f"SOLVED @ {frame['tick']}t" if solved else f"Tick {frame['tick']}/{replay['metrics']['max_steps']}"
    ax.set_title(f"{title}\n{status} | Live Cov: {live_cov:.1f}%", color=TEXT_MAIN, fontsize=10, fontweight="bold", pad=8)


def generate_wallfollower_vs_evolved_gif() -> None:
    """Compare BASELINE-WallFollower vs EXP-4 side-by-side on the same unseen maze (Item #27)."""
    showcase = json.loads((BENCH_DIR / "showcase_index.json").read_text(encoding="utf-8"))
    maze_id = showcase["mazes"][0]
    rep_wf = load_replay(showcase["replays"][maze_id]["BASELINE-WallFollower"])
    rep_e4 = load_replay(showcase["replays"][maze_id]["EXP-4"])

    max_f = min(max(len(rep_wf["frames"]), len(rep_e4["frames"])), 65)
    frame_indices = list(range(0, max_f))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 5.6), dpi=100)
    fig.patch.set_facecolor(BG_DARK)

    images = []
    for f_idx in frame_indices:
        _draw_maze_frame(ax1, rep_wf, f_idx, f"BASELINE-WallFollower ({maze_id})", "#22c55e")
        _draw_maze_frame(ax2, rep_e4, f_idx, f"EXP-4 Evolved Swarm ({maze_id})", "#10b981")
        fig.tight_layout()
        fig.canvas.draw()
        images.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())

    plt.close(fig)
    durations = [115] * (len(images) - 1) + [1800]
    imageio.mimsave(ASSETS_DIR / "demo_wallfollower_vs_exp4.gif", images, duration=durations, loop=0)


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    generate_honest_benchmark_chart()
    generate_wallfollower_vs_evolved_gif()
    print("Generated updated honest assets in assets/:")
    for name in ("benchmark_comparison.png", "demo_wallfollower_vs_exp4.gif"):
        p = ASSETS_DIR / name
        print(f"  - {p.name} ({p.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
