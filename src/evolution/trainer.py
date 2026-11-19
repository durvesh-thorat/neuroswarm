"""Multiprocessing NEAT training loop with Curriculum Scheduler, Novelty Search & MAP-Elites."""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from src.env.maze_generator import MazeSpec, MazeType, generate_maze
from src.env.swarm_env import SwarmConfig
from src.evolution.fitness import (
    FitnessWeights,
    evaluate_genome_batch,
    rollout_genome_on_maze,
)
from src.evolution.map_elites import GridArchive
from src.evolution.neat_core import (
    Genome,
    NEATConfig,
    NEATPopulation,
    ParallelEvaluator,
    save_genome_json,
)
from src.evolution.novelty import EvolutionMode, NoveltyArchive
from src.telemetry.logger import TelemetryLogger
from src.telemetry.replay import save_replay
from src.utils import seed_everything


@dataclass
class TrainingConfig:
    """End-to-end configuration for a NeuroSwarm evolution run."""

    experiment_id: str = "EXP-1"
    neat_config_path: str = "configs/neat_feedforward.ini"
    evolution_mode: EvolutionMode = "objective"
    comm_enabled: bool = True
    use_map_elites: bool = False
    use_curriculum: bool = True
    goal_cue_mode: Literal["fog", "compass"] = "compass"
    num_agents: int = 3
    vision_radius: int = 2
    comm_radius: float = 4.0
    max_steps: int = 110
    mazes_per_gen: int = 3
    generations: int = 30
    pop_size: int | None = None
    seed: int = 42
    num_workers: int = 0  # 0 = auto CPU count
    output_dir: str = "data/runs"
    db_path: str = "data/telemetry.db"
    fixed_maze_type: MazeType | None = None
    fixed_maze_size: int | None = None


@dataclass
class TrainingResult:
    """Outputs produced by a completed training run."""

    run_id: str
    champion: Genome
    champion_path: Path
    replay_path: Path
    archive_path: Path | None
    best_objective_fitness: float
    final_solve_rate: float
    qd_coverage: float
    qd_score: float


class CurriculumScheduler:
    """Automatically scales maze complexity when rolling solve rate crosses 75%."""

    STAGES: list[tuple[str, int, MazeType]] = [
        ("S1_11x11_perfect", 11, "perfect_dfs"),
        ("S2_15x15_braided", 15, "braided"),
        ("S3_15x15_deceptive", 15, "deceptive"),
    ]

    def __init__(self, enabled: bool = True, threshold: float = 0.75, window: int = 3) -> None:
        self.enabled = enabled
        self.threshold = threshold
        self.stage_idx = 0
        self.recent_solve_rates: deque[float] = deque(maxlen=window)

    @property
    def current_stage(self) -> tuple[str, int, MazeType]:
        return self.STAGES[self.stage_idx]

    def update(self, best_solve_rate: float) -> bool:
        """Advance curriculum stage if rolling solve rate exceeds threshold."""
        if not self.enabled or self.stage_idx >= len(self.STAGES) - 1:
            return False
        self.recent_solve_rates.append(float(best_solve_rate))
        if len(self.recent_solve_rates) == self.recent_solve_rates.maxlen:
            if float(np.mean(self.recent_solve_rates)) >= self.threshold:
                self.stage_idx += 1
                self.recent_solve_rates.clear()
                return True
        return False


def _sample_generation_mazes(
    generation: int,
    seed: int,
    count: int,
    stage_size: int,
    stage_type: MazeType,
    mixed_final_stage: bool = False,
) -> list[MazeSpec]:
    """Generate deterministic rotating mazes for a generation to prevent memorization."""
    mazes: list[MazeSpec] = []
    # Rotate across a pool of 15 deterministic seeds so evolution sees consistent + varied layouts
    base_offset = (seed * 997 + (generation // 3)) % 10000
    mixed_types: list[MazeType] = ["perfect_dfs", "braided", "deceptive"]
    for m_idx in range(count):
        m_type = mixed_types[m_idx % 3] if mixed_final_stage else stage_type
        m_seed = base_offset + m_idx * 31
        mazes.append(generate_maze(size=stage_size, maze_type=m_type, seed=m_seed))
    return mazes


def run_training(cfg: TrainingConfig, verbose: bool = True) -> TrainingResult:
    """Run a full deterministic NEAT neuroevolution experiment and persist telemetry + champion."""
    seed_everything(cfg.seed)
    neat_cfg = NEATConfig.from_ini(cfg.neat_config_path, pop_size_override=cfg.pop_size)
    pop = NEATPopulation(neat_cfg, seed=cfg.seed)
    novelty_archive = NoveltyArchive(k=15, max_size=500, novelty_threshold=0.18, hybrid_alpha=0.45)
    qd_archive = GridArchive(bins=(20, 20)) if cfg.use_map_elites else None
    curriculum = CurriculumScheduler(enabled=cfg.use_curriculum)

    run_id = f"{cfg.experiment_id}_seed{cfg.seed}"
    run_dir = Path(cfg.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = TelemetryLogger(cfg.db_path)
    topology = "feed_forward" if neat_cfg.feed_forward else "recurrent"
    logger.start_run(
        run_id=run_id,
        experiment_id=cfg.experiment_id,
        topology=topology,
        evolution_mode=cfg.evolution_mode,
        comm_enabled=cfg.comm_enabled,
        use_map_elites=cfg.use_map_elites,
        seed=cfg.seed,
        config=asdict(cfg),
    )

    swarm_cfg = SwarmConfig(
        num_agents=cfg.num_agents,
        max_steps=cfg.max_steps,
        vision_radius=cfg.vision_radius,
        comm_radius=cfg.comm_radius,
        comm_enabled=cfg.comm_enabled,
        goal_cue_mode=cfg.goal_cue_mode,
        record_trajectory=False,
    )
    weights = FitnessWeights()
    workers = cfg.num_workers if cfg.num_workers > 0 else max(1, min(os.cpu_count() or 1, 4))

    best_overall: Genome | None = None
    best_overall_obj = -float("inf")
    last_best_solve_rate = 0.0

    with ParallelEvaluator(workers, evaluate_genome_batch) as evaluator:
        for gen in range(cfg.generations):
            t0 = time.perf_counter()

            if cfg.fixed_maze_type is not None and cfg.fixed_maze_size is not None:
                stage_name = f"Fixed_{cfg.fixed_maze_size}x{cfg.fixed_maze_size}_{cfg.fixed_maze_type}"
                stage_size = cfg.fixed_maze_size
                stage_type = cfg.fixed_maze_type
                mixed = False
            else:
                stage_name, stage_size, stage_type = curriculum.current_stage
                # In the final third of training, expose swarms to all 3 maze archetypes each gen
                mixed = gen >= int(cfg.generations * 0.55)
                if mixed:
                    stage_name = "S3_15x15_mixed"
                    stage_size = 15

            mazes = _sample_generation_mazes(
                generation=gen,
                seed=cfg.seed,
                count=cfg.mazes_per_gen,
                stage_size=stage_size,
                stage_type=stage_type,
                mixed_final_stage=mixed,
            )

            tasks = [(g, mazes, swarm_cfg, weights) for g in pop.genomes]
            results = evaluator.evaluate(tasks)

            obj_scores = np.array([r.objective_fitness for r in results], dtype=np.float32)
            behaviors = [r.behavior for r in results]
            nov_scores = novelty_archive.compute_novelty_scores(behaviors)
            sel_scores = novelty_archive.combine_fitness(obj_scores, nov_scores, mode=cfg.evolution_mode)

            if cfg.evolution_mode in ("novelty", "hybrid"):
                novelty_archive.update_archive(behaviors, nov_scores)

            for g, res, sel_f in zip(pop.genomes, results, sel_scores):
                g.objective_fitness = float(res.objective_fitness)
                g.fitness = float(sel_f)
                g.behavior = res.behavior.tolist()
                g.descriptor = res.descriptor
                if qd_archive is not None:
                    qd_archive.add(g, g.objective_fitness, g.descriptor)

            best_idx = int(np.argmax(obj_scores))
            gen_best = pop.genomes[best_idx]
            gen_best_res = results[best_idx]
            last_best_solve_rate = gen_best_res.solve_rate

            # Track stage-aware champion: reset best when entering harder mixed stage
            if gen == int(cfg.generations * 0.55) and cfg.fixed_maze_type is None:
                best_overall_obj = -float("inf")
            if gen_best.objective_fitness >= best_overall_obj or best_overall is None:
                best_overall_obj = gen_best.objective_fitness
                best_overall = gen_best.copy()

            qd_cov = qd_archive.coverage if qd_archive is not None else 0.0
            qd_sc = qd_archive.qd_score if qd_archive is not None else 0.0
            elapsed = time.perf_counter() - t0

            logger.log_generation(
                run_id=run_id,
                generation=gen,
                stage=stage_name,
                max_fitness=float(np.max(obj_scores)),
                mean_fitness=float(np.mean(obj_scores)),
                solve_rate=float(np.mean([r.solve_rate for r in results])),
                mean_makespan=float(np.mean([r.mean_makespan for r in results])),
                mean_coverage=float(np.mean([r.mean_coverage for r in results])),
                species_count=len(pop.species),
                best_nodes=len(gen_best.nodes),
                best_conns=gen_best.num_enabled_connections,
                qd_coverage=qd_cov,
                qd_score=qd_sc,
                elapsed_sec=elapsed,
            )

            if verbose and (gen % 10 == 0 or gen == cfg.generations - 1):
                print(
                    f"[{run_id}] Gen {gen:02d}/{cfg.generations - 1} ({stage_name}) | "
                    f"max_obj={float(np.max(obj_scores)):6.2f} mean_obj={float(np.mean(obj_scores)):6.2f} | "
                    f"best_solve={gen_best_res.solve_rate*100:5.1f}% | sp={len(pop.species)} | "
                    f"qd={qd_cov*100:4.1f}% | {elapsed:.2f}s"
                )

            curriculum.update(gen_best_res.solve_rate)
            if gen < cfg.generations - 1:
                pop.reproduce()

    assert best_overall is not None
    # Export champion genome to JSON
    champ_path = save_genome_json(
        best_overall,
        run_dir / "champion.json",
        metadata={
            "run_id": run_id,
            "experiment_id": cfg.experiment_id,
            "topology": topology,
            "evolution_mode": cfg.evolution_mode,
            "comm_enabled": cfg.comm_enabled,
            "use_map_elites": cfg.use_map_elites,
            "seed": cfg.seed,
            "goal_cue_mode": cfg.goal_cue_mode,
        },
    )
    logger.set_champion_path(run_id, str(champ_path))

    # Record deterministic sample replay of the champion on a 15x15 deceptive maze
    replay_maze = generate_maze(size=15, maze_type="deceptive", seed=9000 + cfg.seed)
    replay_cfg = SwarmConfig(
        num_agents=cfg.num_agents,
        max_steps=cfg.max_steps,
        vision_radius=cfg.vision_radius,
        comm_radius=cfg.comm_radius,
        comm_enabled=cfg.comm_enabled,
        goal_cue_mode=cfg.goal_cue_mode,
        record_trajectory=True,
    )
    rep_metrics, rep_traj = rollout_genome_on_maze(best_overall, replay_maze, replay_cfg)
    replay_path = save_replay(
        run_dir / "best_replay.json",
        replay_maze,
        rep_traj,
        rep_metrics,
        metadata={"run_id": run_id, "experiment_id": cfg.experiment_id, "topology": topology},
    )

    archive_path: Path | None = None
    if qd_archive is not None:
        archive_path = qd_archive.save_summary(run_dir / "map_elites_archive.json")

    return TrainingResult(
        run_id=run_id,
        champion=best_overall,
        champion_path=champ_path,
        replay_path=replay_path,
        archive_path=archive_path,
        best_objective_fitness=best_overall_obj,
        final_solve_rate=last_best_solve_rate,
        qd_coverage=qd_archive.coverage if qd_archive is not None else 0.0,
        qd_score=qd_archive.qd_score if qd_archive is not None else 0.0,
    )
