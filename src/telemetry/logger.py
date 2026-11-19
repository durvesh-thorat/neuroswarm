"""SQLite experiment telemetry logger for runs, generation curves, and OOD benchmarks."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TelemetryLogger:
    """Structured SQLite persistence for NeuroSwarm training and benchmark telemetry."""

    def __init__(self, db_path: str | Path = "data/telemetry.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    topology TEXT NOT NULL,
                    evolution_mode TEXT NOT NULL,
                    comm_enabled INTEGER NOT NULL,
                    use_map_elites INTEGER NOT NULL,
                    seed INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    champion_path TEXT
                );

                CREATE TABLE IF NOT EXISTS generations (
                    run_id TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    stage TEXT NOT NULL,
                    max_fitness REAL NOT NULL,
                    mean_fitness REAL NOT NULL,
                    solve_rate REAL NOT NULL,
                    mean_makespan REAL NOT NULL,
                    mean_coverage REAL NOT NULL,
                    species_count INTEGER NOT NULL,
                    best_nodes INTEGER NOT NULL,
                    best_conns INTEGER NOT NULL,
                    qd_coverage REAL NOT NULL,
                    qd_score REAL NOT NULL,
                    elapsed_sec REAL NOT NULL,
                    PRIMARY KEY (run_id, generation),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS ood_benchmarks (
                    run_id TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    maze_id TEXT NOT NULL,
                    maze_type TEXT NOT NULL,
                    maze_size INTEGER NOT NULL,
                    solved INTEGER NOT NULL,
                    makespan INTEGER NOT NULL,
                    bfs_optimal INTEGER NOT NULL,
                    optimality_ratio REAL,
                    coverage REAL NOT NULL,
                    wall_hits INTEGER NOT NULL,
                    agent_collisions INTEGER NOT NULL,
                    team_spread REAL NOT NULL,
                    signal_activity REAL NOT NULL,
                    PRIMARY KEY (run_id, maze_id)
                );
                """
            )

    def start_run(
        self,
        run_id: str,
        experiment_id: str,
        topology: str,
        evolution_mode: str,
        comm_enabled: bool,
        use_map_elites: bool,
        seed: int,
        config: dict[str, Any],
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                (run_id, experiment_id, topology, evolution_mode, comm_enabled, use_map_elites, seed, created_at, config_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    experiment_id,
                    topology,
                    evolution_mode,
                    int(comm_enabled),
                    int(use_map_elites),
                    int(seed),
                    now,
                    json.dumps(config),
                ),
            )

    def set_champion_path(self, run_id: str, champion_path: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE runs SET champion_path = ? WHERE run_id = ?", (champion_path, run_id))

    def log_generation(
        self,
        run_id: str,
        generation: int,
        stage: str,
        max_fitness: float,
        mean_fitness: float,
        solve_rate: float,
        mean_makespan: float,
        mean_coverage: float,
        species_count: int,
        best_nodes: int,
        best_conns: int,
        qd_coverage: float,
        qd_score: float,
        elapsed_sec: float,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO generations
                (run_id, generation, stage, max_fitness, mean_fitness, solve_rate,
                 mean_makespan, mean_coverage, species_count, best_nodes, best_conns,
                 qd_coverage, qd_score, elapsed_sec)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    int(generation),
                    stage,
                    float(max_fitness),
                    float(mean_fitness),
                    float(solve_rate),
                    float(mean_makespan),
                    float(mean_coverage),
                    int(species_count),
                    int(best_nodes),
                    int(best_conns),
                    float(qd_coverage),
                    float(qd_score),
                    float(elapsed_sec),
                ),
            )

    def log_ood_result(
        self,
        run_id: str,
        experiment_id: str,
        seed: int,
        maze_id: str,
        maze_type: str,
        maze_size: int,
        solved: bool,
        makespan: int,
        bfs_optimal: int,
        optimality_ratio: float | None,
        coverage: float,
        wall_hits: int,
        agent_collisions: int,
        team_spread: float,
        signal_activity: float,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO ood_benchmarks
                (run_id, experiment_id, seed, maze_id, maze_type, maze_size, solved,
                 makespan, bfs_optimal, optimality_ratio, coverage, wall_hits,
                 agent_collisions, team_spread, signal_activity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    experiment_id,
                    int(seed),
                    maze_id,
                    maze_type,
                    int(maze_size),
                    int(solved),
                    int(makespan),
                    int(bfs_optimal),
                    None if (optimality_ratio is None or optimality_ratio == float("inf")) else float(optimality_ratio),
                    float(coverage),
                    int(wall_hits),
                    int(agent_collisions),
                    float(team_spread),
                    float(signal_activity),
                ),
            )

    def fetch_generations(self, run_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if run_id is not None:
                rows = conn.execute(
                    "SELECT * FROM generations WHERE run_id = ? ORDER BY generation ASC",
                    (run_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM generations ORDER BY run_id, generation ASC"
                ).fetchall()
        return [dict(r) for r in rows]

    def fetch_ood_benchmarks(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM ood_benchmarks ORDER BY experiment_id, seed, maze_id"
            ).fetchall()
        return [dict(r) for r in rows]
