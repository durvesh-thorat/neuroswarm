"""Unit tests for procedural maze generation and BFS oracle."""

from __future__ import annotations

import unittest

import numpy as np

from src.env.maze_generator import CELL_GOAL, CELL_START, generate_maze
from src.env.oracle import BFSOracle


class TestMazeGenerator(unittest.TestCase):
    def test_all_maze_types_solvable_across_sizes(self) -> None:
        for size in (11, 15, 21):
            for maze_type in ("perfect_dfs", "braided", "deceptive"):
                for seed in range(5):
                    maze = generate_maze(size=size, maze_type=maze_type, seed=seed)
                    oracle = BFSOracle.from_maze(maze)
                    self.assertTrue(
                        oracle.is_solvable,
                        f"Unsolvable maze: type={maze_type}, size={size}, seed={seed}",
                    )
                    self.assertLess(oracle.shortest_path_length, float("inf"))
                    self.assertEqual(maze.grid[maze.start], CELL_START)
                    self.assertEqual(maze.grid[maze.goal], CELL_GOAL)
                    path = oracle.shortest_path()
                    self.assertEqual(path[0], maze.start)
                    self.assertEqual(path[-1], maze.goal)
                    self.assertEqual(len(path) - 1, int(oracle.shortest_path_length))

    def test_determinism_identical_seeds(self) -> None:
        for maze_type in ("perfect_dfs", "braided", "deceptive"):
            m1 = generate_maze(size=15, maze_type=maze_type, seed=42)
            m2 = generate_maze(size=15, maze_type=maze_type, seed=42)
            np.testing.assert_array_equal(m1.grid, m2.grid)
            self.assertEqual(m1.start, m2.start)
            self.assertEqual(m1.goal, m2.goal)

    def test_different_seeds_produce_distinct_mazes(self) -> None:
        m1 = generate_maze(size=15, maze_type="perfect_dfs", seed=1)
        m2 = generate_maze(size=15, maze_type="perfect_dfs", seed=2)
        self.assertFalse(np.array_equal(m1.grid, m2.grid))


if __name__ == "__main__":
    unittest.main()
