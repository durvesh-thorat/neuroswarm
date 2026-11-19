"""Zero-shot Out-of-Distribution (OOD) evaluator with expanded 80-maze suite,
bootstrap confidence intervals, Mann-Whitney U tests, causal communication controls,
behavioral failure-mode logging, and 5 heuristic baselines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from src.env.maze_generator import (
    MazeSpec,
    MazeType,
    compute_deceptive_difficulty,
    generate_maze,
)
from src.env.oracle import BFSOracle
from src.env.swarm_env import (
    GreedyCompassPolicy,
    MultiAgentWallFollowerPolicy,
    RandomWalkPolicy,
    SwarmConfig,
    SwarmMazeEnv,
    WallFollowerPolicy,
    WallFollowerRandomTiePolicy,
)
from src.evolution.fitness import rollout_genome_on_maze
from src.evolution.neat_core import load_genome_json
from src.telemetry.logger import TelemetryLogger
from src.telemetry.replay import save_replay


@dataclass(frozen=True)
class OODMazeEntry:
    maze_id: str
    maze_type: MazeType
    size: int
    seed: int
    max_steps: int
    maze: MazeSpec
    bfs_optimal: int


def build_frozen_ood_suite(mazes_per_archetype: int = 20) -> list[OODMazeEntry]:
    """Build the fixed, held-out OOD benchmark suite (default: 80 unseen mazes).

    Composition (default mazes_per_archetype=20):
      - 20 unseen 15x15 perfect_dfs
      - 20 unseen 15x15 braided
      - 20 unseen 15x15 deceptive
      - 20 unseen 21x21 scaled-up (rotating perfect_dfs, braided, deceptive)
    """
    specs: list[tuple[str, MazeType, int, int, int]] = []
    for i in range(mazes_per_archetype):
        specs.append((f"ood_15_perfect_{i}", "perfect_dfs", 15, 50000 + i, 150))
    for i in range(mazes_per_archetype):
        specs.append((f"ood_15_braided_{i}", "braided", 15, 50100 + i, 150))
    for i in range(mazes_per_archetype):
        specs.append((f"ood_15_deceptive_{i}", "deceptive", 15, 50200 + i, 150))

    scaled_cycle: list[MazeType] = ["perfect_dfs", "braided", "deceptive"]
    for i in range(mazes_per_archetype):
        mtype = scaled_cycle[i % 3]
        specs.append((f"ood_21_scaled_{i}_{mtype}", mtype, 21, 50300 + i, 240))

    suite: list[OODMazeEntry] = []
    for maze_id, mtype, size, seed, max_steps in specs:
        maze = generate_maze(size=size, maze_type=mtype, seed=seed)
        oracle = BFSOracle.from_maze(maze)
        suite.append(
            OODMazeEntry(
                maze_id=maze_id,
                maze_type=mtype,
                size=size,
                seed=seed,
                max_steps=max_steps,
                maze=maze,
                bfs_optimal=int(oracle.shortest_path_length),
            )
        )
    return suite


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute percentile bootstrap confidence interval for the mean."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    if len(arr) == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    idxs = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = np.mean(arr[idxs], axis=1)
    lo = float(np.percentile(means, 100.0 * alpha / 2.0))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def _build_baseline_policy(name: str, num_agents: int, seed: int) -> Any:
    if name == "BASELINE-Random":
        return RandomWalkPolicy(seed=seed)
    if name == "BASELINE-GreedyCompass":
        return GreedyCompassPolicy(num_agents=num_agents)
    if name == "BASELINE-WallFollower":
        return WallFollowerPolicy(num_agents=num_agents)
    if name == "BASELINE-WallFollower-RandTie":
        return WallFollowerRandomTiePolicy(num_agents=num_agents, seed=seed)
    if name == "BASELINE-WallFollower-MultiHand":
        return MultiAgentWallFollowerPolicy(num_agents=num_agents)
    raise ValueError(f"Unknown baseline policy: {name}")


def _evaluate_heuristic_baseline(
    name: str,
    suite: list[OODMazeEntry],
    logger: TelemetryLogger,
    num_agents: int = 3,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in suite:
        scfg = SwarmConfig(
            num_agents=num_agents,
            max_steps=entry.max_steps,
            comm_enabled=False,
            goal_cue_mode="compass",
        )
        env = SwarmMazeEnv(entry.maze, scfg)
        obs = env.reset()
        policy = _build_baseline_policy(name, num_agents=num_agents, seed=entry.seed)
        policy.reset()
        for _ in range(entry.max_steps):
            acts, comms = policy.act(obs)
            obs, done, _ = env.step(acts, comms)
            if done:
                break
        m = env.get_metrics()
        opt = None if m.optimality_ratio == float("inf") else float(m.optimality_ratio)
        logger.log_ood_result(
            run_id=f"{name}_seed0",
            experiment_id=name,
            seed=0,
            maze_id=entry.maze_id,
            maze_type=entry.maze_type,
            maze_size=entry.size,
            solved=m.solved,
            makespan=m.makespan,
            bfs_optimal=entry.bfs_optimal,
            optimality_ratio=opt,
            coverage=m.coverage,
            wall_hits=m.wall_hits,
            agent_collisions=m.agent_collisions,
            team_spread=m.team_spread,
            signal_activity=m.signal_activity,
        )
        rows.append(
            {
                "run_id": f"{name}_seed0",
                "experiment_id": name,
                "seed": 0,
                "maze_id": entry.maze_id,
                "maze_type": entry.maze_type,
                "maze_size": entry.size,
                "solved": int(m.solved),
                "makespan": m.makespan,
                "bfs_optimal": entry.bfs_optimal,
                "optimality_ratio": opt,
                "coverage": m.coverage,
                "wall_hits": m.wall_hits,
                "agent_collisions": m.agent_collisions,
                "team_spread": m.team_spread,
                "signal_activity": m.signal_activity,
                "duplicate_coverage": m.duplicate_coverage,
                "stay_rate": m.stay_rate,
                "oscillation_rate": m.oscillation_rate,
                "wall_hug_rate": m.wall_hug_rate,
            }
        )
    return rows


def evaluate_all_champions_ood(
    runs_dir: str | Path = "data/runs",
    db_path: str | Path = "data/telemetry.db",
    out_dir: str | Path = "data/benchmarks",
    num_agents: int = 3,
    mazes_per_archetype: int = 20,
    exp_filter: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate all baselines and saved champion.json genomes zero-shot on the held-out
    OOD suite, run causal communication controls (normal vs zero vs shuffle), compute
    bootstrap 95% CIs and Mann-Whitney U tests, and export CSV + Markdown tables.
    """
    runs_path = Path(runs_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    replays_dir = out_path / "ood_replays"
    replays_dir.mkdir(parents=True, exist_ok=True)

    suite = build_frozen_ood_suite(mazes_per_archetype=mazes_per_archetype)
    logger = TelemetryLogger(db_path)

    # 0. Maze difficulty & path-length diagnostics (Item #9 & #10)
    diff_rows: list[dict[str, Any]] = []
    for entry in suite:
        d = compute_deceptive_difficulty(entry.maze, max_steps=entry.max_steps)
        arch = "21x21_scaled" if entry.size == 21 else f"15x15_{entry.maze_type}"
        diff_rows.append(
            {
                "maze_id": entry.maze_id,
                "archetype": arch,
                "maze_size": entry.size,
                "maze_type": entry.maze_type,
                "max_steps": entry.max_steps,
                **d,
            }
        )
    diff_df = pd.DataFrame(diff_rows)
    diff_summary = (
        diff_df.groupby("archetype", as_index=False)
        .agg(
            mazes=("maze_id", "count"),
            mean_bfs_optimal=("bfs_optimal", "mean"),
            mean_detour_ratio=("detour_ratio", "mean"),
            greedy_trapped_pct=("greedy_compass_trapped", lambda s: float(s.mean() * 100.0)),
        )
    )
    diff_summary.to_csv(out_path / "ood_maze_difficulty.csv", index=False)

    all_rows: list[dict[str, Any]] = []
    # 1. All 5 heuristic baselines (Item #17)
    baseline_names = [
        "BASELINE-WallFollower",
        "BASELINE-WallFollower-MultiHand",
        "BASELINE-WallFollower-RandTie",
        "BASELINE-GreedyCompass",
        "BASELINE-Random",
    ]
    for bname in baseline_names:
        all_rows.extend(_evaluate_heuristic_baseline(bname, suite, logger, num_agents=num_agents))

    # 2. Evaluate saved champion.json genomes
    champion_files = sorted(runs_path.glob("*/champion.json"))
    loaded_champions: dict[str, tuple[Any, dict[str, Any]]] = {}

    for champ_file in champion_files:
        genome, meta = load_genome_json(champ_file)
        run_id = str(meta.get("run_id", champ_file.parent.name))
        exp_id = str(meta.get("experiment_id", run_id.split("_seed")[0]))
        if exp_filter is not None and exp_id not in exp_filter:
            continue
        seed = int(meta.get("seed", 0))
        comm_enabled = bool(meta.get("comm_enabled", True))
        goal_cue_mode = str(meta.get("goal_cue_mode", "compass"))
        loaded_champions[run_id] = (genome, meta)

        for entry in suite:
            scfg = SwarmConfig(
                num_agents=num_agents,
                max_steps=entry.max_steps,
                comm_enabled=comm_enabled,
                comm_mode="normal" if comm_enabled else "zero",
                goal_cue_mode=goal_cue_mode,  # type: ignore[arg-type]
                record_trajectory=False,
            )
            m, _ = rollout_genome_on_maze(genome, entry.maze, scfg)
            opt = None if m.optimality_ratio == float("inf") else float(m.optimality_ratio)
            logger.log_ood_result(
                run_id=run_id,
                experiment_id=exp_id,
                seed=seed,
                maze_id=entry.maze_id,
                maze_type=entry.maze_type,
                maze_size=entry.size,
                solved=m.solved,
                makespan=m.makespan,
                bfs_optimal=entry.bfs_optimal,
                optimality_ratio=opt,
                coverage=m.coverage,
                wall_hits=m.wall_hits,
                agent_collisions=m.agent_collisions,
                team_spread=m.team_spread,
                signal_activity=m.signal_activity,
            )
            all_rows.append(
                {
                    "run_id": run_id,
                    "experiment_id": exp_id,
                    "seed": seed,
                    "maze_id": entry.maze_id,
                    "maze_type": entry.maze_type,
                    "maze_size": entry.size,
                    "solved": int(m.solved),
                    "makespan": m.makespan,
                    "bfs_optimal": entry.bfs_optimal,
                    "optimality_ratio": opt,
                    "coverage": m.coverage,
                    "wall_hits": m.wall_hits,
                    "agent_collisions": m.agent_collisions,
                    "team_spread": m.team_spread,
                    "signal_activity": m.signal_activity,
                    "duplicate_coverage": m.duplicate_coverage,
                    "stay_rate": m.stay_rate,
                    "oscillation_rate": m.oscillation_rate,
                    "wall_hug_rate": m.wall_hug_rate,
                }
            )

    raw_df = pd.DataFrame(all_rows)
    raw_df.to_csv(out_path / "ood_raw_results.csv", index=False)

    # 3. Seed-level results table (Item #23)
    seed_agg = (
        raw_df.groupby(["experiment_id", "seed"], as_index=False)
        .agg(
            episodes=("solved", "count"),
            solved_count=("solved", "sum"),
            solve_rate_pct=("solved", lambda s: float(s.mean() * 100.0)),
            mean_makespan=("makespan", "mean"),
            mean_coverage_pct=("coverage", lambda s: float(s.mean() * 100.0)),
            mean_opt_ratio=("optimality_ratio", "mean"),
            mean_signal_pct=("signal_activity", lambda s: float(s.mean() * 100.0)),
            mean_dup_cov_pct=("duplicate_coverage", lambda s: float(s.mean() * 100.0)),
            mean_stay_pct=("stay_rate", lambda s: float(s.mean() * 100.0)),
            mean_osc_pct=("oscillation_rate", lambda s: float(s.mean() * 100.0)),
            mean_wall_hug_pct=("wall_hug_rate", lambda s: float(s.mean() * 100.0)),
        )
    )
    seed_agg.to_csv(out_path / "ood_seed_level.csv", index=False)

    # 4. Main summary table with bootstrap 95% CIs & solved counts (Items #1, #5, #15)
    discovered_exps = sorted(e for e in raw_df["experiment_id"].unique() if e not in baseline_names)
    exp_order = baseline_names + discovered_exps

    summary_rows: list[dict[str, Any]] = []
    behav_rows: list[dict[str, Any]] = []
    for exp_id in exp_order:
        ep_sub = raw_df[raw_df["experiment_id"] == exp_id]
        sd_sub = seed_agg[seed_agg["experiment_id"] == exp_id]
        if ep_sub.empty:
            continue
        solved_arr = ep_sub["solved"].to_numpy(dtype=np.float64) * 100.0
        ci_lo, ci_hi = bootstrap_ci(solved_arr)
        n_solved = int(ep_sub["solved"].sum())
        n_total = int(len(ep_sub))
        opt_vals = ep_sub["optimality_ratio"].dropna().to_numpy(dtype=np.float64)
        opt_mean = float(np.mean(opt_vals)) if len(opt_vals) > 0 else float("nan")

        summary_rows.append(
            {
                "Experiment": exp_id,
                "Seeds": int(len(sd_sub)),
                "Episodes": n_total,
                "Solved_Episodes": n_solved,
                "Solve_Rate_Mean": float(np.mean(solved_arr)),
                "Solve_Rate_CI95_Low": ci_lo,
                "Solve_Rate_CI95_High": ci_hi,
                "Makespan_Mean": float(ep_sub["makespan"].mean()),
                "Opt_Ratio_Mean": opt_mean,
                "Coverage_Mean": float(ep_sub["coverage"].mean() * 100.0),
                "Signal_Activity": float(ep_sub["signal_activity"].mean() * 100.0),
            }
        )
        behav_rows.append(
            {
                "Experiment": exp_id,
                "Episodes": n_total,
                "Duplicate_Coverage_Pct": float(ep_sub["duplicate_coverage"].mean() * 100.0),
                "Stay_Rate_Pct": float(ep_sub["stay_rate"].mean() * 100.0),
                "Oscillation_Rate_Pct": float(ep_sub["oscillation_rate"].mean() * 100.0),
                "Wall_Hug_Rate_Pct": float(ep_sub["wall_hug_rate"].mean() * 100.0),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_path / "ood_summary.csv", index=False)
    pd.DataFrame(behav_rows).to_csv(out_path / "ood_behavior_profile.csv", index=False)

    # 5. Pairwise Mann-Whitney U significance tests over episode outcomes (Item #16)
    sig_rows: list[dict[str, Any]] = []
    evolved_present = [e for e in ("EXP-1", "EXP-2", "EXP-3", "EXP-4") if e in raw_df["experiment_id"].values]
    for e1, e2 in combinations(evolved_present, 2):
        a = raw_df[raw_df["experiment_id"] == e1]["solved"].to_numpy(dtype=np.float64)
        b = raw_df[raw_df["experiment_id"] == e2]["solved"].to_numpy(dtype=np.float64)
        stat, pval = mannwhitneyu(a, b, alternative="two-sided")
        sig_rows.append(
            {
                "Exp_A": e1,
                "Exp_B": e2,
                "Rate_A_Pct": float(np.mean(a) * 100.0),
                "Rate_B_Pct": float(np.mean(b) * 100.0),
                "Diff_Pct_Pts": float((np.mean(b) - np.mean(a)) * 100.0),
                "U_Stat": float(stat),
                "p_value": float(pval),
                "Significant_p05": bool(pval < 0.05),
            }
        )
    pd.DataFrame(sig_rows).to_csv(out_path / "ood_significance.csv", index=False)

    # 6. Causal Communication Control Ablation: normal vs zero vs shuffle (Item #21)
    comm_rows: list[dict[str, Any]] = []
    for run_id, (genome, meta) in sorted(loaded_champions.items()):
        if not bool(meta.get("comm_enabled", False)):
            continue
        exp_id = str(meta.get("experiment_id", run_id.split("_seed")[0]))
        seed = int(meta.get("seed", 0))
        goal_cue_mode = str(meta.get("goal_cue_mode", "compass"))
        for mode in ("normal", "zero", "shuffle"):
            solved_cnt = 0
            cov_sum = 0.0
            for entry in suite:
                scfg = SwarmConfig(
                    num_agents=num_agents,
                    max_steps=entry.max_steps,
                    comm_enabled=(mode != "zero"),
                    comm_mode=mode,  # type: ignore[arg-type]
                    comm_shuffle_seed=entry.seed + seed,
                    goal_cue_mode=goal_cue_mode,  # type: ignore[arg-type]
                    record_trajectory=False,
                )
                m, _ = rollout_genome_on_maze(genome, entry.maze, scfg)
                solved_cnt += int(m.solved)
                cov_sum += float(m.coverage)
            comm_rows.append(
                {
                    "experiment_id": exp_id,
                    "run_id": run_id,
                    "seed": seed,
                    "comm_mode": mode,
                    "episodes": len(suite),
                    "solved_count": solved_cnt,
                    "solve_rate_pct": float(solved_cnt) / float(len(suite)) * 100.0,
                    "mean_coverage_pct": cov_sum / float(len(suite)) * 100.0,
                }
            )
    if comm_rows:
        comm_df = pd.DataFrame(comm_rows)
        comm_summary = (
            comm_df.groupby(["experiment_id", "comm_mode"], as_index=False)
            .agg(
                episodes=("episodes", "sum"),
                solved_count=("solved_count", "sum"),
                solve_rate_pct=("solve_rate_pct", "mean"),
                mean_coverage_pct=("mean_coverage_pct", "mean"),
            )
        )
        comm_summary.to_csv(out_path / "ood_comm_ablation.csv", index=False)

    # 7. Breakdown by maze archetype
    raw_df["archetype"] = np.where(
        raw_df["maze_size"] == 21,
        "21x21_scaled",
        "15x15_" + raw_df["maze_type"].astype(str),
    )
    by_arch = (
        raw_df.groupby(["experiment_id", "archetype"], as_index=False)
        .agg(
            episodes=("solved", "count"),
            solved_count=("solved", "sum"),
            solve_rate_pct=("solved", lambda s: float(s.mean() * 100.0)),
            mean_makespan=("makespan", "mean"),
            mean_coverage_pct=("coverage", lambda s: float(s.mean() * 100.0)),
        )
    )
    by_arch.to_csv(out_path / "ood_by_archetype.csv", index=False)

    # 8. Export clean Markdown summary table with 95% bootstrap CIs & solved episode counts
    md_lines = [
        "| Experiment | Seeds | Solved / Total | OOD Solve Rate (95% CI) | Mean Makespan | Optimality Ratio (n solved) | Coverage (%) | Signal (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, r in summary_df.iterrows():
        n_sol = int(r["Solved_Episodes"])
        n_tot = int(r["Episodes"])
        opt_str = f"{r['Opt_Ratio_Mean']:.2f} (n={n_sol})" if not np.isnan(r["Opt_Ratio_Mean"]) else "N/A (n=0)"
        md_lines.append(
            f"| **{r['Experiment']}** | {int(r['Seeds'])} | {n_sol}/{n_tot} | "
            f"{r['Solve_Rate_Mean']:.1f}% [{r['Solve_Rate_CI95_Low']:.1f}, {r['Solve_Rate_CI95_High']:.1f}] | "
            f"{r['Makespan_Mean']:.1f} | {opt_str} | "
            f"{r['Coverage_Mean']:.1f}% | {r['Signal_Activity']:.1f}% |"
        )
    (out_path / "ood_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # 9. Record showcase OOD replays including Wall-Follower vs EXP-4 comparison (Item #27)
    showcase_meta = _export_showcase_replays(raw_df, suite, loaded_champions, replays_dir, num_agents)
    with open(out_path / "showcase_index.json", "w", encoding="utf-8") as fh:
        json.dump(showcase_meta, fh, indent=2)

    return summary_df, by_arch, showcase_meta


def _export_showcase_replays(
    raw_df: pd.DataFrame,
    suite: list[OODMazeEntry],
    loaded_champions: dict[str, tuple[Any, dict[str, Any]]],
    replays_dir: Path,
    num_agents: int,
) -> dict[str, Any]:
    """Select best seed per experiment and record trajectory replays including Wall-Follower."""
    best_run_per_exp: dict[str, str] = {}
    for exp_id in ("EXP-1", "EXP-2", "EXP-3", "EXP-4"):
        sub = raw_df[raw_df["experiment_id"] == exp_id]
        if sub.empty:
            continue
        run_scores = sub.groupby("run_id").agg(solve=("solved", "mean"), makespan=("makespan", "mean"))
        run_scores = run_scores.sort_values(by=["solve", "makespan"], ascending=[False, True])
        best_run_per_exp[exp_id] = str(run_scores.index[0])

    suite_map = {e.maze_id: e for e in suite}
    selected_maze_ids: list[str] = []

    if "EXP-1" in best_run_per_exp and "EXP-4" in best_run_per_exp:
        r1 = raw_df[raw_df["run_id"] == best_run_per_exp["EXP-1"]].set_index("maze_id")
        r4 = raw_df[raw_df["run_id"] == best_run_per_exp["EXP-4"]].set_index("maze_id")
        for prefix in ("ood_15_deceptive_", "ood_15_braided_", "ood_15_perfect_"):
            cands = [mid for mid in suite_map if mid.startswith(prefix)]
            best_mid = max(
                cands,
                key=lambda mid: (
                    int(r4.loc[mid, "solved"]) - int(r1.loc[mid, "solved"]),
                    float(r1.loc[mid, "makespan"]) - float(r4.loc[mid, "makespan"]),
                ),
            )
            selected_maze_ids.append(best_mid)
    else:
        selected_maze_ids = ["ood_15_deceptive_0", "ood_15_braided_0", "ood_15_perfect_0"]

    exported: dict[str, Any] = {
        "best_runs": best_run_per_exp,
        "mazes": selected_maze_ids,
        "replays": {},
    }

    for mid in selected_maze_ids:
        entry = suite_map[mid]
        exported["replays"][mid] = {}
        # Export Wall-Follower replay on the exact same maze (Item #27)
        wf_cfg = SwarmConfig(num_agents=num_agents, max_steps=entry.max_steps, comm_enabled=False, goal_cue_mode="compass", record_trajectory=True)
        wf_env = SwarmMazeEnv(entry.maze, wf_cfg)
        obs = wf_env.reset()
        wf_pol = WallFollowerPolicy(num_agents=num_agents)
        wf_pol.reset()
        for _ in range(entry.max_steps):
            acts, comms = wf_pol.act(obs)
            obs, done, _ = wf_env.step(acts, comms)
            if done:
                break
        wf_path = replays_dir / f"{mid}__BASELINE-WallFollower.json"
        save_replay(
            wf_path,
            entry.maze,
            wf_env.trajectory,
            wf_env.get_metrics(),
            metadata={"run_id": "BASELINE-WallFollower_seed0", "experiment_id": "BASELINE-WallFollower", "maze_id": mid, "comm_enabled": False},
        )
        exported["replays"][mid]["BASELINE-WallFollower"] = str(wf_path)

        for exp_id, run_id in best_run_per_exp.items():
            genome, meta = loaded_champions[run_id]
            scfg = SwarmConfig(
                num_agents=num_agents,
                max_steps=entry.max_steps,
                comm_enabled=bool(meta.get("comm_enabled", True)),
                goal_cue_mode=str(meta.get("goal_cue_mode", "compass")),  # type: ignore[arg-type]
                record_trajectory=True,
            )
            metrics, traj = rollout_genome_on_maze(genome, entry.maze, scfg)
            rpath = replays_dir / f"{mid}__{exp_id}.json"
            save_replay(
                rpath,
                entry.maze,
                traj,
                metrics,
                metadata={
                    "run_id": run_id,
                    "experiment_id": exp_id,
                    "maze_id": mid,
                    "topology": meta.get("topology", "unknown"),
                    "comm_enabled": bool(meta.get("comm_enabled", True)),
                },
            )
            exported["replays"][mid][exp_id] = str(rpath)

    return exported
