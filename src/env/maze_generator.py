"""Procedural seeded maze generators: perfect_dfs, braided, and deceptive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

CELL_OPEN: int = 0
CELL_WALL: int = 1
CELL_START: int = 2
CELL_GOAL: int = 3
CELL_SWITCH: int = 4

MazeType = Literal["perfect_dfs", "braided", "deceptive"]


@dataclass(frozen=True)
class MazeSpec:
    """Immutable specification of a generated procedural maze."""

    grid: np.ndarray  # shape (H, W), dtype=np.uint8
    start: tuple[int, int]
    goal: tuple[int, int]
    maze_type: MazeType
    seed: int
    switch: tuple[int, int] | None = None

    @property
    def height(self) -> int:
        return int(self.grid.shape[0])

    @property
    def width(self) -> int:
        return int(self.grid.shape[1])


def _carve_perfect_dfs(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """Carve a perfect (tree-structured) maze using iterative randomized DFS."""
    grid = np.ones((height, width), dtype=np.uint8)
    start_r, start_c = 1, 1
    grid[start_r, start_c] = CELL_OPEN
    stack: list[tuple[int, int]] = [(start_r, start_c)]
    dirs = [(-2, 0), (2, 0), (0, -2), (0, 2)]

    while stack:
        r, c = stack[-1]
        neighbors: list[tuple[int, int, int, int]] = []
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 1 <= nr < height - 1 and 1 <= nc < width - 1 and grid[nr, nc] == CELL_WALL:
                neighbors.append((nr, nc, r + dr // 2, c + dc // 2))
        if neighbors:
            idx = int(rng.integers(0, len(neighbors)))
            nr, nc, wr, wc = neighbors[idx]
            grid[wr, wc] = CELL_OPEN
            grid[nr, nc] = CELL_OPEN
            stack.append((nr, nc))
        else:
            stack.pop()
    return grid


def _braid_maze(grid: np.ndarray, braid_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Remove a fraction of dead-end walls to create loops and multiple routes."""
    grid = grid.copy()
    h, w = grid.shape
    cardinals = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(1, h - 1, 2):
        for c in range(1, w - 1, 2):
            if grid[r, c] != CELL_OPEN:
                continue
            open_neighbors = 0
            wall_candidates: list[tuple[int, int]] = []
            for dr, dc in cardinals:
                wr, wc = r + dr, c + dc
                nr, nc = r + 2 * dr, c + 2 * dc
                if grid[wr, wc] == CELL_OPEN:
                    open_neighbors += 1
                elif 1 <= nr < h - 1 and 1 <= nc < w - 1 and grid[nr, nc] == CELL_OPEN:
                    wall_candidates.append((wr, wc))
            if open_neighbors == 1 and wall_candidates and rng.random() < braid_prob:
                wr, wc = wall_candidates[int(rng.integers(0, len(wall_candidates)))]
                grid[wr, wc] = CELL_OPEN
    return grid


def _build_deceptive_maze(height: int, width: int, rng: np.random.Generator) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """Construct a deceptive cul-de-sac maze with a U-shaped barrier trapping greedy agents.

    Start is placed near the left-center, goal near the right-center, and a deep
    U-shaped cul-de-sac opens toward the start while blocking the direct line to the goal.
    Agents must route around the outer corridors (top or bottom) to reach the exit.
    """
    grid = _carve_perfect_dfs(height, width, rng)
    grid = _braid_maze(grid, braid_prob=0.15, rng=rng)

    mid_r = height // 2
    if mid_r % 2 == 0:
        mid_r = max(1, mid_r - 1)

    start = (mid_r, 1)
    goal = (mid_r, width - 2)

    # Carve an inviting open corridor from start directly into a U-shaped trap
    trap_left = max(3, width // 4)
    trap_right = min(width - 4, (3 * width) // 4)
    trap_top = max(2, mid_r - max(2, height // 5))
    trap_bot = min(height - 3, mid_r + max(2, height // 5))

    # Open the interior of the deceptive pocket and approach lane
    grid[mid_r, 1 : trap_right] = CELL_OPEN
    for r in range(trap_top + 1, trap_bot):
        for c in range(trap_left, trap_right):
            grid[r, c] = CELL_OPEN

    # Build the U-shaped barrier (top arm, bottom arm, and right back wall)
    grid[trap_top, trap_left : trap_right + 1] = CELL_WALL
    grid[trap_bot, trap_left : trap_right + 1] = CELL_WALL
    grid[trap_top : trap_bot + 1, trap_right] = CELL_WALL

    # Ensure outer perimeter corridors exist so the maze is always cleanly solvable
    # Choose whether top, bottom, or both outer bypasses are open (seeded)
    bypass_mode = int(rng.integers(0, 3))
    if bypass_mode in (0, 2):
        # Open top bypass
        grid[1, 1 : width - 1] = CELL_OPEN
        grid[1 : mid_r + 1, 1] = CELL_OPEN
        grid[1 : mid_r + 1, width - 2] = CELL_OPEN
    if bypass_mode in (1, 2):
        # Open bottom bypass
        grid[height - 2, 1 : width - 1] = CELL_OPEN
        grid[mid_r : height - 1, 1] = CELL_OPEN
        grid[mid_r : height - 1, width - 2] = CELL_OPEN

    # Guarantee start and goal are open
    grid[start] = CELL_OPEN
    grid[goal] = CELL_OPEN
    return grid, start, goal


def generate_maze(
    size: int = 11,
    maze_type: MazeType = "perfect_dfs",
    seed: int = 0,
    braid_prob: float = 0.20,
    with_switch: bool = False,
) -> MazeSpec:
    """Generate a deterministic, guaranteed-solvable procedural maze.

    Args:
        size: Odd grid dimension (H == W). Even sizes are incremented by 1.
        maze_type: One of "perfect_dfs", "braided", or "deceptive".
        seed: Deterministic RNG seed.
        braid_prob: Fraction of dead-end walls removed in "braided" mode.
        with_switch: If True, places an optional cooperative switch cell (code 4).

    Returns:
        MazeSpec containing uint8 grid and start/goal coordinates.
    """
    if size < 7:
        raise ValueError("Maze size must be at least 7.")
    if size % 2 == 0:
        size += 1

    rng = np.random.default_rng(seed)

    if maze_type == "perfect_dfs":
        grid = _carve_perfect_dfs(size, size, rng)
        start = (1, 1)
        goal = (size - 2, size - 2)
    elif maze_type == "braided":
        grid = _carve_perfect_dfs(size, size, rng)
        grid = _braid_maze(grid, braid_prob=braid_prob, rng=rng)
        start = (1, 1)
        goal = (size - 2, size - 2)
    elif maze_type == "deceptive":
        grid, start, goal = _build_deceptive_maze(size, size, rng)
    else:
        raise ValueError(f"Unknown maze_type: {maze_type!r}")

    grid[start] = CELL_START
    grid[goal] = CELL_GOAL

    switch_pos: tuple[int, int] | None = None
    if with_switch:
        open_cells = np.argwhere(grid == CELL_OPEN)
        if len(open_cells) > 0:
            choice = open_cells[int(rng.integers(0, len(open_cells)))]
            switch_pos = (int(choice[0]), int(choice[1]))
            grid[switch_pos] = CELL_SWITCH

    return MazeSpec(
        grid=grid,
        start=start,
        goal=goal,
        maze_type=maze_type,
        seed=seed,
        switch=switch_pos,
    )


def compute_deceptive_difficulty(maze: MazeSpec, max_steps: int = 150) -> dict[str, float]:
    """Measure maze path length, detour ratio, and greedy-compass trap rate.

    Simulates a deterministic single-agent greedy compass walker (always choosing
    the open neighbor reducing Euclidean distance to the goal) to verify whether
    a maze actually traps greedy agents and quantify its path-length confound.
    """
    from src.env.oracle import BFSOracle

    oracle = BFSOracle.from_maze(maze)
    bfs_opt = int(oracle.shortest_path_length)
    sr, sc = maze.start
    gr, gc = maze.goal
    manhattan = abs(gr - sr) + abs(gc - sc)
    detour_ratio = float(bfs_opt) / float(max(1, manhattan))

    r, c = sr, sc
    min_bfs = bfs_opt
    solved = False
    for _ in range(max_steps):
        if (r, c) == (gr, gc):
            solved = True
            min_bfs = 0
            break
        best_pos = (r, c)
        best_dist = float("inf")
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < maze.height and 0 <= nc < maze.width and maze.grid[nr, nc] != CELL_WALL:
                d = float((nr - gr) ** 2 + (nc - gc) ** 2)
                if d < best_dist:
                    best_dist = d
                    best_pos = (nr, nc)
        r, c = best_pos
        d_bfs = oracle.distance_at((r, c))
        if d_bfs >= 0 and d_bfs < min_bfs:
            min_bfs = d_bfs
        if (r, c) == (gr, gc):
            solved = True
            min_bfs = 0
            break

    return {
        "bfs_optimal": float(bfs_opt),
        "manhattan_dist": float(manhattan),
        "detour_ratio": round(detour_ratio, 3),
        "greedy_compass_trapped": 0.0 if solved else 1.0,
        "greedy_min_bfs_dist": float(min_bfs),
    }
