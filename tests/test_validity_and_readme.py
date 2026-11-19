"""Tests for validity checks, comm controls, new baselines, and README-vs-CSV consistency."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.env.comm_bus import CommBus
from src.env.maze_generator import compute_deceptive_difficulty, generate_maze
from src.env.swarm_env import (
    GreedyCompassPolicy,
    MultiAgentWallFollowerPolicy,
    SwarmConfig,
    SwarmMazeEnv,
    WallFollowerRandomTiePolicy,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestValidityAndReadmeSync(unittest.TestCase):
    def test_deceptive_difficulty_metric(self) -> None:
        maze = generate_maze(size=15, maze_type="deceptive", seed=50200)
        d = compute_deceptive_difficulty(maze)
        self.assertGreater(d["bfs_optimal"], 0)
        self.assertEqual(d["greedy_compass_trapped"], 1.0)
        self.assertGreaterEqual(d["detour_ratio"], 1.0)

    def test_comm_bus_zero_and_shuffle_modes(self) -> None:
        positions = np.array([[1, 1], [1, 2], [2, 1]], dtype=np.int32)
        bus_zero = CommBus(num_agents=3, comm_enabled=True, comm_mode="zero")
        bus_zero.broadcast(np.ones((3, 2), dtype=np.float32))
        self.assertTrue(np.allclose(bus_zero.receive(positions), 0.0))

        bus_shuf = CommBus(num_agents=3, comm_enabled=True, comm_mode="shuffle", shuffle_seed=7)
        bus_shuf.broadcast(np.array([[1.0, -1.0], [0.5, 0.5], [-0.8, 0.2]], dtype=np.float32))
        rec = bus_shuf.receive(positions)
        self.assertEqual(rec.shape, (3, 2))
        self.assertGreater(float(np.abs(rec).sum()), 0.0)

    def test_new_baselines_and_behavior_metrics(self) -> None:
        maze = generate_maze(size=11, maze_type="perfect_dfs", seed=42)
        for pol in (
            GreedyCompassPolicy(num_agents=3),
            WallFollowerRandomTiePolicy(num_agents=3, seed=42),
            MultiAgentWallFollowerPolicy(num_agents=3),
        ):
            env = SwarmMazeEnv(maze, SwarmConfig(num_agents=3, max_steps=80, comm_enabled=False, goal_cue_mode="compass"))
            obs = env.reset()
            pol.reset()
            for _ in range(80):
                acts, comms = pol.act(obs)
                obs, done, _ = env.step(acts, comms)
                if done:
                    break
            m = env.get_metrics()
            self.assertGreaterEqual(m.duplicate_coverage, 0.0)
            self.assertLessEqual(m.duplicate_coverage, 1.0)
            self.assertGreaterEqual(m.stay_rate, 0.0)
            self.assertLessEqual(m.stay_rate, 1.0)

    def test_config_matches_readme_budget(self) -> None:
        cfg = yaml.safe_load((REPO_ROOT / "configs/experiments.yaml").read_text(encoding="utf-8"))
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertEqual(cfg["defaults"]["pop_size"], 40)
        self.assertEqual(cfg["defaults"]["generations"], 30)
        self.assertIn("pop_size=40", readme)
        self.assertIn("generations=30", readme)

    def test_readme_table_matches_ood_summary_csv(self) -> None:
        csv_path = REPO_ROOT / "data/benchmarks/ood_summary.csv"
        readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            exp = str(row["Experiment"])
            solved_str = f"{int(row['Solved_Episodes'])}/{int(row['Episodes'])}"
            rate_str = f"{row['Solve_Rate_Mean']:.1f}%"
            pattern = rf"\|\s*\*\*{re.escape(exp)}\*\*\s*\|[^|]+\|\s*{re.escape(solved_str)}\s*\|\s*{re.escape(rate_str)}"
            self.assertRegex(
                readme_text,
                pattern,
                msg=f"README OOD table row mismatch for {exp} (expected {solved_str}, {rate_str})",
            )


if __name__ == "__main__":
    unittest.main()
