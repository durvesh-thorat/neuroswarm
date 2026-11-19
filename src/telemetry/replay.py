"""Compact episode trajectory serialization to JSON (and optional NPZ)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.env.maze_generator import MazeSpec
from src.env.swarm_env import EpisodeMetrics


def save_replay(
    path: str | Path,
    maze: MazeSpec,
    trajectory: list[dict[str, Any]],
    metrics: EpisodeMetrics,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Serialize a full episode replay (maze layout, fog masks, positions, signals) to JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for f in trajectory:
        fog = f["fog_mask"]
        fog_list = fog.tolist() if isinstance(fog, np.ndarray) else fog
        frames.append(
            {
                "tick": int(f["tick"]),
                "positions": f["positions"],
                "actions": f["actions"],
                "broadcasts": f["broadcasts"],
                "fog_mask": fog_list,
                "obs_sample": f.get("obs_sample", []),
            }
        )

    payload: dict[str, Any] = {
        "metadata": metadata or {},
        "maze": {
            "grid": maze.grid.tolist(),
            "start": list(maze.start),
            "goal": list(maze.goal),
            "maze_type": maze.maze_type,
            "seed": int(maze.seed),
            "height": maze.height,
            "width": maze.width,
        },
        "metrics": {
            "solved": bool(metrics.solved),
            "makespan": int(metrics.makespan),
            "max_steps": int(metrics.max_steps),
            "coverage": round(float(metrics.coverage), 4),
            "min_bfs_dist": int(metrics.min_bfs_dist),
            "start_bfs_dist": int(metrics.start_bfs_dist),
            "wall_hits": int(metrics.wall_hits),
            "agent_collisions": int(metrics.agent_collisions),
            "team_spread": round(float(metrics.team_spread), 4),
            "signal_activity": round(float(metrics.signal_activity), 4),
            "optimality_ratio": (
                None
                if metrics.optimality_ratio == float("inf")
                else round(float(metrics.optimality_ratio), 4)
            ),
        },
        "frames": frames,
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return out_path


def load_replay(path: str | Path) -> dict[str, Any]:
    """Load a saved JSON episode replay."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
