"""Exact BFS shortest-path oracle and distance-field solver."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from src.env.maze_generator import CELL_WALL, MazeSpec

UNREACHABLE: int = -1


def compute_bfs_distance_field(grid: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    """Compute exact shortest-path BFS steps from target to every reachable cell.

    Walls and unreachable cells have value -1 (UNREACHABLE).
    """
    h, w = grid.shape
    dist = np.full((h, w), UNREACHABLE, dtype=np.int32)
    tr, tc = target
    if not (0 <= tr < h and 0 <= tc < w) or grid[tr, tc] == CELL_WALL:
        return dist

    dist[tr, tc] = 0
    queue: deque[tuple[int, int]] = deque([(tr, tc)])

    while queue:
        r, c = queue.popleft()
        next_d = dist[r, c] + 1
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != CELL_WALL and dist[nr, nc] == UNREACHABLE:
                dist[nr, nc] = next_d
                queue.append((nr, nc))
    return dist


@dataclass(frozen=True)
class BFSOracle:
    """Precomputed BFS distance field and optimal path properties for a maze."""

    distance_field: np.ndarray  # shape (H, W), int32
    start: tuple[int, int]
    goal: tuple[int, int]
    reachable_cells: int

    @classmethod
    def from_maze(cls, maze: MazeSpec) -> "BFSOracle":
        field = compute_bfs_distance_field(maze.grid, maze.goal)
        reachable = int(np.sum(field >= 0))
        return cls(
            distance_field=field,
            start=maze.start,
            goal=maze.goal,
            reachable_cells=reachable,
        )

    @property
    def shortest_path_length(self) -> float:
        d = int(self.distance_field[self.start])
        return float("inf") if d < 0 else float(d)

    @property
    def is_solvable(self) -> bool:
        return int(self.distance_field[self.start]) >= 0

    def distance_at(self, pos: tuple[int, int]) -> int:
        return int(self.distance_field[pos[0], pos[1]])

    def shortest_path(self) -> list[tuple[int, int]]:
        """Reconstruct one exact shortest path from start to goal."""
        if not self.is_solvable:
            return []
        path = [self.start]
        r, c = self.start
        h, w = self.distance_field.shape
        while (r, c) != self.goal:
            cur_d = self.distance_field[r, c]
            moved = False
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and self.distance_field[nr, nc] == cur_d - 1:
                    r, c = nr, nc
                    path.append((r, c))
                    moved = True
                    break
            if not moved:
                break
        return path
