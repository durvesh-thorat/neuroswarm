"""Objective fitness evaluation and multi-maze swarm rollout harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.env.maze_generator import MazeSpec
from src.env.swarm_env import COMM_DIM, NUM_ACTIONS, EpisodeMetrics, SwarmConfig, SwarmMazeEnv
from src.evolution.neat_core import Genome, build_phenotype


@dataclass(frozen=True)
class FitnessWeights:
    """Weights for objective fitness F_obj (Section 4.3)."""

    w_coverage: float = 6.0
    w_solved: float = 15.0
    w_progress: float = 8.0
    w_collision: float = 0.5


def compute_objective_fitness(metrics: EpisodeMetrics, weights: FitnessWeights | None = None) -> float:
    """Compute scalar objective fitness F_obj from EpisodeMetrics.

    F_obj = w1 * C_cov
          + w2 * I_solved * (1 + (T_max - T_solve) / T_max)
          + w3 * (1 - d_min / d_start)
          - w4 * P_coll

    Note: P_coll is normalized by episode agent-steps into [0, 1+] so wall-hit
    penalties guide clean navigation without creating a "stand-still" local minimum.
    """
    w = weights or FitnessWeights()
    cov_term = w.w_coverage * metrics.coverage
    solve_term = (
        w.w_solved * (1.0 + float(metrics.max_steps - metrics.makespan) / float(max(1, metrics.max_steps)))
        if metrics.solved
        else 0.0
    )
    progress = 1.0 - (float(metrics.min_bfs_dist) / float(max(1, metrics.start_bfs_dist)))
    progress_term = w.w_progress * max(0.0, progress)
    denom = float(max(1, metrics.max_steps * max(1, len(metrics.final_positions))))
    p_coll = float(metrics.wall_hits + 2 * metrics.agent_collisions) / denom
    coll_penalty = w.w_collision * p_coll
    return float(cov_term + solve_term + progress_term - coll_penalty)


def build_behavior_vector(metrics: EpisodeMetrics) -> np.ndarray:
    """Construct behavioral characterization vector b in R^(2N + 4):
    sorted normalized final agent positions + 4 quadrant fog-of-war coverage ratios.
    """
    return np.concatenate(
        [
            metrics.final_positions.astype(np.float32).ravel(),
            metrics.quadrant_coverage.astype(np.float32).ravel(),
        ]
    )


def rollout_genome_on_maze(
    genome: Genome,
    maze: MazeSpec,
    swarm_cfg: SwarmConfig,
) -> tuple[EpisodeMetrics, list[dict[str, Any]]]:
    """Run one deterministic swarm episode for a NEAT genome on a single maze.

    Instantiates one phenotype copy per agent and explicitly calls net.reset()
    before the episode to prevent recurrent state leakage across mazes.
    """
    env = SwarmMazeEnv(maze, swarm_cfg)
    obs = env.reset()

    # Dedicated phenotype per agent so recurrent state is agent-local, reset every episode
    nets = [build_phenotype(genome) for _ in range(swarm_cfg.num_agents)]
    for net in nets:
        net.reset()

    n = swarm_cfg.num_agents
    actions = np.zeros(n, dtype=np.int32)
    broadcasts = np.zeros((n, COMM_DIM), dtype=np.float32)

    for _ in range(swarm_cfg.max_steps):
        for i in range(n):
            out = nets[i].activate(obs[i])
            # Evaluate movement logits (UP, DOWN, LEFT, RIGHT, STAY)
            actions[i] = int(np.argmax(out[:NUM_ACTIONS]))
            broadcasts[i, 0] = float(out[NUM_ACTIONS])
            broadcasts[i, 1] = float(out[NUM_ACTIONS + 1])
        obs, done, _ = env.step(actions, broadcasts)
        if done:
            break

    return env.get_metrics(), env.trajectory


@dataclass
class GenomeEvalResult:
    """Aggregated evaluation result of one genome across a batch of mazes."""

    genome_id: int
    objective_fitness: float
    solve_rate: float
    mean_makespan: float
    mean_coverage: float
    behavior: np.ndarray
    descriptor: tuple[float, float]


def evaluate_genome_batch(task: tuple[Genome, list[MazeSpec], SwarmConfig, FitnessWeights]) -> GenomeEvalResult:
    """Top-level picklable worker function for multiprocessing genome evaluation."""
    genome, mazes, swarm_cfg, weights = task
    fits: list[float] = []
    solved_flags: list[float] = []
    makespans: list[float] = []
    coverages: list[float] = []
    behaviors: list[np.ndarray] = []
    spreads: list[float] = []
    activities: list[float] = []

    for maze in mazes:
        metrics, _ = rollout_genome_on_maze(genome, maze, swarm_cfg)
        fits.append(compute_objective_fitness(metrics, weights))
        solved_flags.append(1.0 if metrics.solved else 0.0)
        makespans.append(float(metrics.makespan))
        coverages.append(float(metrics.coverage))
        behaviors.append(build_behavior_vector(metrics))
        spreads.append(float(metrics.team_spread))
        # Axis 2: signal activity when comm is enabled, else exploration efficiency
        act_metric = metrics.signal_activity if swarm_cfg.comm_enabled else metrics.exploration_efficiency
        activities.append(float(act_metric))

    mean_behavior = np.mean(np.stack(behaviors, axis=0), axis=0).astype(np.float32)
    descriptor = (float(np.mean(spreads)), float(np.mean(activities)))
    return GenomeEvalResult(
        genome_id=genome.genome_id,
        objective_fitness=float(np.mean(fits)),
        solve_rate=float(np.mean(solved_flags)),
        mean_makespan=float(np.mean(makespans)),
        mean_coverage=float(np.mean(coverages)),
        behavior=mean_behavior,
        descriptor=descriptor,
    )
