"""Pure NumPy headless multi-agent fog-of-war maze simulation engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from src.env.comm_bus import CommBus, CommMode
from src.env.maze_generator import CELL_WALL, MazeSpec
from src.env.oracle import BFSOracle

# Movement actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_STAY = 4
NUM_ACTIONS = 5

# Action deltas (dr, dc)
ACTION_DELTAS = np.array(
    [
        [-1, 0],  # UP
        [1, 0],   # DOWN
        [0, -1],  # LEFT
        [0, 1],   # RIGHT
        [0, 0],   # STAY
    ],
    dtype=np.int32,
)

# Observation layout (D = 15):
#   [0:4]   Local wall sensors (N, S, E, W) in {0.0, 1.0}
#   [4:8]   Visited/breadcrumb memory for adjacent (N, S, E, W) in [0.0, 1.0]
#   [8:11]  Nearest teammate local radar (dx_norm, dy_norm, proximity) in [-1, 1]
#   [11:13] Received communication signals [s0, s1] in [-1, 1]
#   [13:15] Goal cue (cos_theta, sin_theta) in [-1, 1] (zero unless visible in fog mode)
OBS_DIM = 15
COMM_DIM = 2
OUTPUT_DIM = NUM_ACTIONS + COMM_DIM  # 7


@dataclass
class SwarmConfig:
    """Configuration for the headless multi-agent swarm environment."""

    num_agents: int = 3
    max_steps: int = 120
    vision_radius: int = 2
    comm_radius: float = 4.0
    comm_enabled: bool = True
    comm_mode: CommMode = "normal"
    comm_shuffle_seed: int = 0
    heterogeneous_roles: bool = False  # Option B: Scouts (r=2) + Relays (R_comm=6)
    goal_cue_mode: Literal["fog", "compass"] = "fog"
    strict_cooperative: bool = False  # If True, ALL agents must reach goal
    collision_mode: Literal["index_priority", "block_both"] = "index_priority"
    record_trajectory: bool = False


@dataclass
class EpisodeMetrics:
    """Aggregated telemetry for a single multi-agent episode."""

    solved: bool = False
    makespan: int = 0
    max_steps: int = 0
    coverage: float = 0.0
    min_bfs_dist: int = 0
    start_bfs_dist: int = 1
    wall_hits: int = 0
    agent_collisions: int = 0
    team_spread: float = 0.0
    signal_activity: float = 0.0
    exploration_efficiency: float = 0.0
    duplicate_coverage: float = 0.0
    stay_rate: float = 0.0
    oscillation_rate: float = 0.0
    wall_hug_rate: float = 0.0
    optimality_ratio: float = float("inf")
    final_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    quadrant_coverage: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))


class SwarmMazeEnv:
    """Headless vectorized multi-agent fog-of-war maze environment."""

    def __init__(self, maze: MazeSpec, config: SwarmConfig | None = None) -> None:
        self.maze = maze
        self.config = config or SwarmConfig()
        self.oracle = BFSOracle.from_maze(maze)
        self.height, self.width = maze.grid.shape
        self.diagonal = float(np.hypot(self.height, self.width))

        # Per-agent vision & comm radii (Option A homogeneous vs Option B Scout/Relay)
        n = self.config.num_agents
        self.vision_radii = np.full(n, self.config.vision_radius, dtype=np.int32)
        comm_radii = np.full(n, self.config.comm_radius, dtype=np.float32)
        if self.config.heterogeneous_roles and n >= 2:
            split = n // 2
            self.vision_radii[:split] = max(2, self.config.vision_radius)
            comm_radii[:split] = max(3.0, self.config.comm_radius - 1.0)
            self.vision_radii[split:] = max(1, self.config.vision_radius - 1)
            comm_radii[split:] = 6.0

        self.comm_bus = CommBus(
            num_agents=n,
            comm_dim=COMM_DIM,
            default_radius=self.config.comm_radius,
            comm_enabled=self.config.comm_enabled,
            agent_radii=comm_radii,
            comm_mode=self.config.comm_mode,
            shuffle_seed=self.config.comm_shuffle_seed,
        )

        # Runtime state
        self.positions = np.zeros((n, 2), dtype=np.int32)
        self._prev_positions = np.zeros((n, 2), dtype=np.int32)
        self._prev2_positions = np.zeros((n, 2), dtype=np.int32)
        self.reached_goal = np.zeros(n, dtype=bool)
        self.fog_mask = np.zeros((self.height, self.width), dtype=bool)
        self.visited_mask = np.zeros((self.height, self.width), dtype=bool)
        self.visit_counts = np.zeros((self.height, self.width), dtype=np.float32)
        self._agent_visited: list[set[tuple[int, int]]] = [set() for _ in range(n)]
        self.step_count: int = 0
        self.wall_hits: int = 0
        self.agent_collisions: int = 0
        self.min_bfs_dist: int = int(self.oracle.distance_at(self.maze.start))
        self.solved: bool = False
        self.solve_tick: int = self.config.max_steps
        self._spread_accum: float = 0.0
        self._total_moves: int = 0
        self._stay_ticks: int = 0
        self._oscillation_ticks: int = 0
        self._wall_hug_ticks: int = 0
        self._behavior_ticks: int = 0
        self.trajectory: list[dict[str, Any]] = []

    def reset(self) -> np.ndarray:
        """Reset environment state and return initial observations (N, OBS_DIM)."""
        n = self.config.num_agents
        self.positions[:] = np.array(self.maze.start, dtype=np.int32)
        self._prev_positions[:] = self.positions
        self._prev2_positions[:] = self.positions
        self.reached_goal.fill(False)
        self.fog_mask.fill(False)
        self.visited_mask.fill(False)
        self.visit_counts.fill(0.0)
        self._agent_visited = [set() for _ in range(n)]
        self.comm_bus.reset()

        self.step_count = 0
        self.wall_hits = 0
        self.agent_collisions = 0
        self.min_bfs_dist = max(0, int(self.oracle.distance_at(self.maze.start)))
        self.solved = False
        self.solve_tick = self.config.max_steps
        self._spread_accum = 0.0
        self._total_moves = 0
        self._stay_ticks = 0
        self._oscillation_ticks = 0
        self._wall_hug_ticks = 0
        self._behavior_ticks = 0
        self.trajectory = []

        self._update_fog_and_visited()
        obs = self.get_observations()
        if self.config.record_trajectory:
            self._record_frame(
                actions=np.full(n, ACTION_STAY, dtype=np.int32),
                broadcasts=np.zeros((n, COMM_DIM), dtype=np.float32),
                obs=obs,
            )
        return obs

    def _update_fog_and_visited(self) -> None:
        """Update shared fog-of-war visibility mask and visited cell breadcrumbs."""
        for i in range(self.config.num_agents):
            r, c = int(self.positions[i, 0]), int(self.positions[i, 1])
            self.visited_mask[r, c] = True
            self.visit_counts[r, c] += 1.0
            self._agent_visited[i].add((r, c))
            rad = int(self.vision_radii[i])
            r0 = max(0, r - rad)
            r1 = min(self.height, r + rad + 1)
            c0 = max(0, c - rad)
            c1 = min(self.width, c + rad + 1)
            self.fog_mask[r0:r1, c0:c1] = True

    def get_observations(self) -> np.ndarray:
        """Build fixed-length normalized observation matrix of shape (N, 15)."""
        n = self.config.num_agents
        obs = np.zeros((n, OBS_DIM), dtype=np.float32)
        received = self.comm_bus.receive(self.positions)
        gr, gc = self.maze.goal
        grid = self.maze.grid

        for i in range(n):
            r, c = int(self.positions[i, 0]), int(self.positions[i, 1])
            rad = int(self.vision_radii[i])

            # 1. Local Wall Sensors (4 floats: N, S, E, W) + 2. Visited Breadcrumbs (4 floats)
            for idx, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, 1), (0, -1))):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    is_wall = grid[nr, nc] == CELL_WALL
                    obs[i, idx] = 1.0 if is_wall else 0.0
                    obs[i, 4 + idx] = min(1.0, float(self.visit_counts[nr, nc]) / 4.0) if not is_wall else 1.0
                else:
                    obs[i, idx] = 1.0
                    obs[i, 4 + idx] = 0.0

            # 3. Teammate Local Radar (3 floats: dx, dy, proximity)
            if n > 1:
                best_dist = float("inf")
                best_dr = 0.0
                best_dc = 0.0
                sense_range = max(float(rad) * 2.0, float(self.comm_bus.agent_radii[i]))
                for j in range(n):
                    if j == i:
                        continue
                    dr_t = float(self.positions[j, 0] - r)
                    dc_t = float(self.positions[j, 1] - c)
                    d = float(np.hypot(dr_t, dc_t))
                    if d <= sense_range and d < best_dist:
                        best_dist = d
                        best_dr = dr_t
                        best_dc = dc_t
                if best_dist < float("inf"):
                    scale = max(1.0, sense_range)
                    obs[i, 8] = float(np.clip(best_dc / scale, -1.0, 1.0))
                    obs[i, 9] = float(np.clip(best_dr / scale, -1.0, 1.0))
                    obs[i, 10] = float(np.clip(1.0 - best_dist / scale, 0.0, 1.0))

            # 4. Communication Receiver (2 floats)
            obs[i, 11] = received[i, 0]
            obs[i, 12] = received[i, 1]

            # 5. Goal Cue (2 floats: cos_theta, sin_theta)
            dr_g = float(gr - r)
            dc_g = float(gc - c)
            dist_g = float(np.hypot(dr_g, dc_g))
            if self.config.goal_cue_mode == "compass" or dist_g <= float(rad):
                if dist_g > 1e-6:
                    obs[i, 13] = dc_g / dist_g
                    obs[i, 14] = dr_g / dist_g

        return obs

    def step(
        self,
        actions: np.ndarray,
        broadcasts: np.ndarray | None = None,
    ) -> tuple[np.ndarray, bool, dict[str, Any]]:
        """Execute one simultaneous simulation tick for all N agents."""
        n = self.config.num_agents
        actions_arr = np.asarray(actions, dtype=np.int32)
        if broadcasts is None:
            broadcasts_arr = np.zeros((n, COMM_DIM), dtype=np.float32)
        else:
            broadcasts_arr = np.asarray(broadcasts, dtype=np.float32)

        self.step_count += 1
        grid = self.maze.grid

        # Proposed next positions
        proposals = self.positions.copy()
        for i in range(n):
            if self.reached_goal[i]:
                continue
            act = int(actions_arr[i])
            if act < 0 or act >= NUM_ACTIONS:
                act = ACTION_STAY
            dr, dc = int(ACTION_DELTAS[act, 0]), int(ACTION_DELTAS[act, 1])
            nr = int(self.positions[i, 0]) + dr
            nc = int(self.positions[i, 1]) + dc
            if 0 <= nr < self.height and 0 <= nc < self.width and grid[nr, nc] != CELL_WALL:
                proposals[i, 0] = nr
                proposals[i, 1] = nc
            else:
                self.wall_hits += 1

        # Deterministic collision & tie-breaking (skip start/goal stacking restrictions)
        target_counts: dict[tuple[int, int], list[int]] = {}
        for i in range(n):
            key = (int(proposals[i, 0]), int(proposals[i, 1]))
            target_counts.setdefault(key, []).append(i)

        final_positions = proposals.copy()
        for key, claimants in target_counts.items():
            if len(claimants) <= 1 or key == self.maze.start or key == self.maze.goal:
                continue
            self.agent_collisions += len(claimants) - 1
            if self.config.collision_mode == "index_priority":
                for loser in claimants[1:]:
                    final_positions[loser] = self.positions[loser]
            else:
                for idx in claimants:
                    final_positions[idx] = self.positions[idx]

        # Behavioral failure-mode telemetry (stay, 2-cycle oscillation, wall-hugging)
        for i in range(n):
            if self.reached_goal[i]:
                continue
            self._behavior_ticks += 1
            same_as_prev = (
                final_positions[i, 0] == self.positions[i, 0]
                and final_positions[i, 1] == self.positions[i, 1]
            )
            if same_as_prev:
                self._stay_ticks += 1
            elif self.step_count >= 2:
                back_to_prev2 = (
                    final_positions[i, 0] == self._prev2_positions[i, 0]
                    and final_positions[i, 1] == self._prev2_positions[i, 1]
                )
                if back_to_prev2:
                    self._oscillation_ticks += 1

            r_new, c_new = int(final_positions[i, 0]), int(final_positions[i, 1])
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r_new + dr, c_new + dc
                if nr < 0 or nr >= self.height or nc < 0 or nc >= self.width or grid[nr, nc] == CELL_WALL:
                    self._wall_hug_ticks += 1
                    break

        moved_mask = np.any(final_positions != self.positions, axis=1)
        self._total_moves += int(np.sum(moved_mask))
        self._prev2_positions = self._prev_positions.copy()
        self._prev_positions = self.positions.copy()
        self.positions = final_positions

        # Broadcast signals for next tick
        emitted = self.comm_bus.broadcast(broadcasts_arr)

        # Update fog-of-war, BFS distance, and goal status
        self._update_fog_and_visited()
        gr, gc = self.maze.goal
        for i in range(n):
            r, c = int(self.positions[i, 0]), int(self.positions[i, 1])
            d = self.oracle.distance_at((r, c))
            if d >= 0 and d < self.min_bfs_dist:
                self.min_bfs_dist = d
            if r == gr and c == gc:
                self.reached_goal[i] = True

        # Accumulate mean pairwise spread
        if n > 1:
            diff = self.positions[:, None, :].astype(np.float32) - self.positions[None, :, :].astype(np.float32)
            dists = np.sqrt(np.sum(diff * diff, axis=-1))
            tri = dists[np.triu_indices(n, k=1)]
            self._spread_accum += float(np.mean(tri)) / max(1.0, self.diagonal)

        # Check termination
        if self.config.strict_cooperative:
            now_solved = bool(np.all(self.reached_goal))
        else:
            now_solved = bool(np.any(self.reached_goal))

        if now_solved and not self.solved:
            self.solved = True
            self.solve_tick = self.step_count

        done = self.solved or (self.step_count >= self.config.max_steps)
        obs = self.get_observations()

        if self.config.record_trajectory:
            self._record_frame(actions=actions_arr, broadcasts=emitted, obs=obs)

        info = {
            "step": self.step_count,
            "solved": self.solved,
            "wall_hits": self.wall_hits,
            "agent_collisions": self.agent_collisions,
        }
        return obs, done, info

    def _record_frame(
        self,
        actions: np.ndarray,
        broadcasts: np.ndarray,
        obs: np.ndarray,
    ) -> None:
        self.trajectory.append(
            {
                "tick": self.step_count,
                "positions": self.positions.copy().tolist(),
                "actions": actions.copy().tolist(),
                "broadcasts": np.round(broadcasts, 3).tolist(),
                "fog_mask": self.fog_mask.astype(np.uint8).copy(),
                "obs_sample": np.round(obs[0], 3).tolist() if len(obs) > 0 else [],
            }
        )

    def get_metrics(self) -> EpisodeMetrics:
        """Compute summary metrics for the completed (or current) episode."""
        open_mask = self.maze.grid != CELL_WALL
        total_open = max(1, int(np.sum(open_mask)))
        uncovered_open = int(np.sum(self.fog_mask & open_mask))
        coverage = float(uncovered_open) / float(total_open)

        unique_visited = int(np.sum(self.visited_mask & open_mask))
        efficiency = float(unique_visited) / float(max(1, self._total_moves + self.config.num_agents))

        # Team overlap / duplicate coverage across agents
        sum_agent_unique = sum(len(s) for s in self._agent_visited)
        if sum_agent_unique > 0 and self.config.num_agents > 1:
            dup_cov = max(0.0, 1.0 - float(unique_visited) / float(sum_agent_unique))
        else:
            dup_cov = 0.0

        b_denom = float(max(1, self._behavior_ticks))
        stay_rate = float(self._stay_ticks) / b_denom
        osc_rate = float(self._oscillation_ticks) / b_denom
        wall_hug_rate = float(self._wall_hug_ticks) / b_denom

        start_d = max(1, int(self.oracle.distance_at(self.maze.start)))
        opt_ratio = (
            float(self.solve_tick) / float(start_d) if self.solved else float("inf")
        )
        mean_spread = self._spread_accum / float(max(1, self.step_count))

        # Normalized final positions sorted by coordinate for permutation-invariant BC
        norm_pos = np.column_stack(
            [
                self.positions[:, 0].astype(np.float32) / max(1.0, float(self.height - 1)),
                self.positions[:, 1].astype(np.float32) / max(1.0, float(self.width - 1)),
            ]
        )
        order = np.lexsort((norm_pos[:, 1], norm_pos[:, 0]))
        sorted_pos = norm_pos[order]

        # Quadrant coverage ratios (4 floats)
        mid_r, mid_c = self.height // 2, self.width // 2
        quads = [
            (slice(0, mid_r), slice(0, mid_c)),
            (slice(0, mid_r), slice(mid_c, self.width)),
            (slice(mid_r, self.height), slice(0, mid_c)),
            (slice(mid_r, self.height), slice(mid_c, self.width)),
        ]
        q_cov = np.zeros(4, dtype=np.float32)
        for q_idx, (rs, cs) in enumerate(quads):
            q_open = int(np.sum(open_mask[rs, cs]))
            q_seen = int(np.sum(self.fog_mask[rs, cs] & open_mask[rs, cs]))
            q_cov[q_idx] = float(q_seen) / float(max(1, q_open))

        return EpisodeMetrics(
            solved=self.solved,
            makespan=self.solve_tick if self.solved else self.config.max_steps,
            max_steps=self.config.max_steps,
            coverage=coverage,
            min_bfs_dist=self.min_bfs_dist,
            start_bfs_dist=start_d,
            wall_hits=self.wall_hits,
            agent_collisions=self.agent_collisions,
            team_spread=float(np.clip(mean_spread, 0.0, 1.0)),
            signal_activity=float(np.clip(self.comm_bus.signal_activity_ratio, 0.0, 1.0)),
            exploration_efficiency=float(np.clip(efficiency, 0.0, 1.0)),
            duplicate_coverage=float(np.clip(dup_cov, 0.0, 1.0)),
            stay_rate=float(np.clip(stay_rate, 0.0, 1.0)),
            oscillation_rate=float(np.clip(osc_rate, 0.0, 1.0)),
            wall_hug_rate=float(np.clip(wall_hug_rate, 0.0, 1.0)),
            optimality_ratio=opt_ratio,
            final_positions=sorted_pos,
            quadrant_coverage=q_cov,
        )


class RandomWalkPolicy:
    """Deterministic seeded random-walk baseline policy."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def act(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = obs.shape[0]
        actions = self.rng.integers(0, 4, size=n, dtype=np.int32)
        broadcasts = np.zeros((n, COMM_DIM), dtype=np.float32)
        return actions, broadcasts


class GreedyCompassPolicy:
    """Greedy baseline that always picks the open neighbor most aligned with the goal compass."""

    # Map wall-sensor index (N=0, S=1, E=2, W=3) to (action, unit_dc, unit_dr)
    _DIRS = [
        (0, ACTION_UP, 0.0, -1.0),
        (1, ACTION_DOWN, 0.0, 1.0),
        (2, ACTION_RIGHT, 1.0, 0.0),
        (3, ACTION_LEFT, -1.0, 0.0),
    ]

    def __init__(self, num_agents: int = 3) -> None:
        self.num_agents = num_agents

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = obs.shape[0]
        actions = np.full(n, ACTION_STAY, dtype=np.int32)
        for i in range(n):
            cos_t = float(obs[i, 13])
            sin_t = float(obs[i, 14])
            best_act = ACTION_STAY
            best_score = -float("inf")
            for w_idx, act, dc, dr in self._DIRS:
                if obs[i, w_idx] < 0.5:
                    score = dc * cos_t + dr * sin_t
                    if score > best_score:
                        best_score = score
                        best_act = act
            actions[i] = best_act
        broadcasts = np.zeros((n, COMM_DIM), dtype=np.float32)
        return actions, broadcasts


class WallFollowerPolicy:
    """Right-hand rule wall-follower baseline policy (with per-agent heading state)."""

    # Headings: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT -> mapped to env ACTION_*
    _HEADING_TO_ACTION = [ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT]
    # Wall sensor index in obs[0:4] is (N=0, S=1, E=2, W=3)
    _HEADING_TO_WALL_IDX = [0, 2, 1, 3]

    def __init__(self, num_agents: int = 3) -> None:
        self.num_agents = num_agents
        self.headings = np.arange(num_agents, dtype=np.int32) % 4

    def reset(self) -> None:
        self.headings = np.arange(self.num_agents, dtype=np.int32) % 4

    def act(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = obs.shape[0]
        actions = np.zeros(n, dtype=np.int32)
        for i in range(n):
            h = int(self.headings[i])
            chosen = h
            for offset in (1, 0, 3, 2):
                cand = (h + offset) % 4
                wall_idx = self._HEADING_TO_WALL_IDX[cand]
                if obs[i, wall_idx] < 0.5:
                    chosen = cand
                    break
            self.headings[i] = chosen
            actions[i] = self._HEADING_TO_ACTION[chosen]
        broadcasts = np.zeros((n, COMM_DIM), dtype=np.float32)
        return actions, broadcasts


class WallFollowerRandomTiePolicy(WallFollowerPolicy):
    """Wall-follower with seeded random initial headings and random tie-breaking at junctions."""

    def __init__(self, num_agents: int = 3, seed: int = 0) -> None:
        super().__init__(num_agents=num_agents)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.headings = self.rng.integers(0, 4, size=self.num_agents, dtype=np.int32)

    def act(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = obs.shape[0]
        actions = np.zeros(n, dtype=np.int32)
        for i in range(n):
            h = int(self.headings[i])
            # Randomly pick right-hand or left-hand preference when multiple open exits exist
            open_cands = []
            for offset in (1, 0, 3, 2):
                cand = (h + offset) % 4
                if obs[i, self._HEADING_TO_WALL_IDX[cand]] < 0.5:
                    open_cands.append(cand)
            if not open_cands:
                chosen = h
            elif len(open_cands) == 1:
                chosen = open_cands[0]
            else:
                # 80% right-hand rule, 20% random non-backtrack open turn
                non_back = [c for c in open_cands if c != (h + 2) % 4]
                if non_back and self.rng.random() < 0.20:
                    chosen = int(non_back[int(self.rng.integers(0, len(non_back)))])
                else:
                    chosen = open_cands[0]
            self.headings[i] = chosen
            actions[i] = self._HEADING_TO_ACTION[chosen]
        broadcasts = np.zeros((n, COMM_DIM), dtype=np.float32)
        return actions, broadcasts


class MultiAgentWallFollowerPolicy(WallFollowerPolicy):
    """Multi-agent wall-follower assigning alternating right-hand and left-hand rules & orientations."""

    def __init__(self, num_agents: int = 3) -> None:
        super().__init__(num_agents=num_agents)
        self.hands = np.array([1 if i % 2 == 0 else -1 for i in range(num_agents)], dtype=np.int32)
        self.reset()

    def reset(self) -> None:
        # Spread initial headings evenly around compass
        self.headings = np.array([(i * 2 + (i // 2)) % 4 for i in range(self.num_agents)], dtype=np.int32)

    def act(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = obs.shape[0]
        actions = np.zeros(n, dtype=np.int32)
        for i in range(n):
            h = int(self.headings[i])
            offsets = (1, 0, 3, 2) if self.hands[i] == 1 else (3, 0, 1, 2)
            chosen = h
            for offset in offsets:
                cand = (h + offset) % 4
                if obs[i, self._HEADING_TO_WALL_IDX[cand]] < 0.5:
                    chosen = cand
                    break
            self.headings[i] = chosen
            actions[i] = self._HEADING_TO_ACTION[chosen]
        broadcasts = np.zeros((n, COMM_DIM), dtype=np.float32)
        return actions, broadcasts
