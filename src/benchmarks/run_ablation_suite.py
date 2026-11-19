"""CLI runner to execute the controlled 4-experiment ablation matrix and OOD benchmark."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from src.benchmarks.ood_evaluator import evaluate_all_champions_ood
from src.evolution.trainer import TrainingConfig, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NeuroSwarm 4-Experiment Ablation Suite & OOD Benchmark")
    parser.add_argument("--config", type=str, default="configs/experiments.yaml")
    parser.add_argument("--generations", type=int, default=None, help="Override generations per run")
    parser.add_argument("--pop-size", type=int, default=None, help="Override population size")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Override seed list (e.g. --seeds 101 202 303)")
    parser.add_argument("--goal-cue-mode", type=str, choices=["compass", "fog"], default=None, help="Override goal cue mode")
    parser.add_argument("--mazes-per-archetype", type=int, default=20, help="Held-out OOD mazes per archetype (default 20 = 80 total)")
    parser.add_argument("--workers", type=int, default=2, help="Multiprocessing worker count")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and only run OOD eval")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)

    defaults = spec["defaults"]
    seeds = args.seeds if args.seeds is not None else defaults["seeds"]
    generations = args.generations if args.generations is not None else int(defaults["generations"])
    pop_size = args.pop_size if args.pop_size is not None else int(defaults["pop_size"])
    goal_cue_mode = args.goal_cue_mode if args.goal_cue_mode is not None else str(defaults.get("goal_cue_mode", "compass"))

    t_start = time.perf_counter()
    if not args.skip_train:
        for exp_id, exp_cfg in spec["experiments"].items():
            print(f"\n=== Running {exp_cfg['name']} ({len(seeds)} seeds x {generations} gens, pop={pop_size}, cue={goal_cue_mode}) ===")
            for seed in seeds:
                tcfg = TrainingConfig(
                    experiment_id=exp_id,
                    neat_config_path=exp_cfg["neat_config"],
                    evolution_mode=exp_cfg["evolution_mode"],
                    comm_enabled=bool(exp_cfg["comm_enabled"]),
                    use_map_elites=bool(exp_cfg["use_map_elites"]),
                    use_curriculum=bool(defaults.get("use_curriculum", True)),
                    goal_cue_mode=goal_cue_mode,  # type: ignore[arg-type]
                    num_agents=int(defaults["num_agents"]),
                    vision_radius=int(defaults["vision_radius"]),
                    comm_radius=float(defaults["comm_radius"]),
                    max_steps=int(defaults["max_steps"]),
                    mazes_per_gen=int(defaults["mazes_per_gen"]),
                    generations=generations,
                    pop_size=pop_size,
                    seed=int(seed),
                    num_workers=args.workers,
                )
                run_training(tcfg, verbose=True)

    print(f"\n=== Running Zero-Shot OOD Evaluation on {args.mazes_per_archetype * 4} Unseen Mazes ===")
    evaluate_all_champions_ood(mazes_per_archetype=args.mazes_per_archetype)
    print("\n" + Path("data/benchmarks/ood_summary.md").read_text(encoding="utf-8"))
    print(f"\nTotal pipeline time: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
