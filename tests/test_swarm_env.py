"""Unit tests for headless SwarmMazeEnv physics, collisions, fog-of-war, comm, and throughput."""

from __future__ import annotations

import time
import unittest

import numpy as np

from src.env.maze_generator import generate_maze
from src.env.swarm_env import (
    ACTION_DOWN,
    ACTION_RIGHT,
    ACTION_STAY,
    OBS_DIM,
    RandomWalkPolicy,
    SwarmConfig,
    SwarmMazeEnv,
    WallFollowerPolicy,
)


class TestSwarmEnv(unittest.TestCase):
    def test_determinism_bit_identical_trajectories(self) -> None:
        maze = generate_maze(size=11, maze_type="braided", seed=7)
        cfg = SwarmConfig(num_agents=3, max_steps=60, comm_enabled=True, record_trajectory=True)

        env1 = SwarmMazeEnv(maze, cfg)
        pol1 = RandomWalkPolicy(seed=123)
        obs1 = env1.reset()
        for _ in range(60):
            act, comm = pol1.act(obs1)
            obs1, done, _ = env1.step(act, comm)
            if done:
                break

        env2 = SwarmMazeEnv(maze, cfg)
        pol2 = RandomWalkPolicy(seed=123)
        obs2 = env2.reset()
        for _ in range(60):
            act, comm = pol2.act(obs2)
            obs2, done, _ = env2.step(act, comm)
            if done:
                break

        np.testing.assert_array_equal(env1.positions, env2.positions)
        np.testing.assert_array_equal(env1.fog_mask, env2.fog_mask)
        self.assertEqual(env1.wall_hits, env2.wall_hits)
        self.assertEqual(env1.agent_collisions, env2.agent_collisions)

    def test_comm_bus_enabled_vs_silent_ablation(self) -> None:
        maze = generate_maze(size=11, maze_type="perfect_dfs", seed=3)
        broadcast = np.array([[0.9, -0.8], [0.5, 0.5], [-0.4, 0.7]], dtype=np.float32)

        env_comm = SwarmMazeEnv(maze, SwarmConfig(num_agents=3, comm_enabled=True, comm_radius=5.0))
        env_comm.reset()
        obs_comm, _, _ = env_comm.step(np.full(3, ACTION_STAY), broadcast)
        # Agent 0 receives mean of agents 1 & 2
        np.testing.assert_allclose(obs_comm[0, 11:13], np.array([0.05, 0.6], dtype=np.float32), atol=1e-5)

        env_silent = SwarmMazeEnv(maze, SwarmConfig(num_agents=3, comm_enabled=False, comm_radius=5.0))
        env_silent.reset()
        obs_silent, _, _ = env_silent.step(np.full(3, ACTION_STAY), broadcast)
        np.testing.assert_array_equal(obs_silent[:, 11:13], np.zeros((3, 2), dtype=np.float32))

    def test_collision_resolution_index_priority(self) -> None:
        maze = generate_maze(size=11, maze_type="deceptive", seed=5)
        env = SwarmMazeEnv(maze, SwarmConfig(num_agents=2, collision_mode="index_priority"))
        env.reset()
        # Move both agents right from start simultaneously into the same open cell
        actions = np.array([ACTION_RIGHT, ACTION_RIGHT], dtype=np.int32)
        env.step(actions)
        # If the cell to the right is open, agent 0 wins and agent 1 stays at start
        sr, sc = maze.start
        if maze.grid[sr, sc + 1] != 1:
            self.assertEqual(tuple(env.positions[0]), (sr, sc + 1))
            self.assertEqual(tuple(env.positions[1]), (sr, sc))
            self.assertEqual(env.agent_collisions, 1)

    def test_wall_follower_solves_perfect_maze(self) -> None:
        maze = generate_maze(size=11, maze_type="perfect_dfs", seed=0)
        env = SwarmMazeEnv(maze, SwarmConfig(num_agents=3, max_steps=300))
        policy = WallFollowerPolicy(num_agents=3)
        obs = env.reset()
        policy.reset()
        self.assertEqual(obs.shape, (3, OBS_DIM))
        for _ in range(300):
            act, comm = policy.act(obs)
            obs, done, _ = env.step(act, comm)
            if done:
                break
        metrics = env.get_metrics()
        self.assertTrue(metrics.solved)
        self.assertGreater(metrics.coverage, 0.1)

    def test_headless_cpu_throughput(self) -> None:
        maze = generate_maze(size=11, maze_type="braided", seed=1)
        env = SwarmMazeEnv(maze, SwarmConfig(num_agents=3, max_steps=100))
        obs = env.reset()
        rng = np.random.default_rng(0)
        n_ticks = 1000
        actions = rng.integers(0, 5, size=(n_ticks, 3), dtype=np.int32)
        broadcasts = np.zeros((3, 2), dtype=np.float32)

        t0 = time.perf_counter()
        for t in range(n_ticks):
            obs, done, _ = env.step(actions[t], broadcasts)
            if done:
                env.reset()
        elapsed = time.perf_counter() - t0
        agent_steps_per_sec = (n_ticks * 3) / max(1e-6, elapsed)
        self.assertGreaterEqual(
            agent_steps_per_sec,
            5000.0,
            f"Throughput too low: {agent_steps_per_sec:.1f} agent-steps/sec",
        )


if __name__ == "__main__":
    unittest.main()
