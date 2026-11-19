"""Smoke tests for NEAT evolution, recurrent state reset, Novelty Search, MAP-Elites, and JSON export."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.env.maze_generator import generate_maze
from src.env.swarm_env import SwarmConfig
from src.evolution.fitness import rollout_genome_on_maze
from src.evolution.map_elites import GridArchive
from src.evolution.neat_core import (
    NEATConfig,
    RecurrentNetwork,
    build_phenotype,
    create_random_genome,
    InnovationTracker,
    load_genome_json,
    save_genome_json,
)
from src.evolution.novelty import NoveltyArchive
from src.evolution.trainer import TrainingConfig, run_training


class TestEvolution(unittest.TestCase):
    def test_recurrent_network_explicit_reset(self) -> None:
        cfg = NEATConfig.from_ini("configs/neat_recurrent.ini")
        rng = np.random.default_rng(11)
        tracker = InnovationTracker(cfg.num_outputs + cfg.num_hidden)
        genome = create_random_genome(1, cfg, tracker, rng)
        net = build_phenotype(genome)
        self.assertIsInstance(net, RecurrentNetwork)

        x = np.ones(cfg.num_inputs, dtype=np.float32) * 0.7
        net.reset()
        out1 = net.activate(x)
        _ = net.activate(x)  # advance internal recurrent state
        net.reset()
        out2 = net.activate(x)
        np.testing.assert_allclose(out1, out2, atol=1e-6)

    def test_novelty_archive_knn_and_hybrid(self) -> None:
        archive = NoveltyArchive(k=3, max_size=50, novelty_threshold=0.1)
        behaviors = [np.array([i * 0.1, 0.0, 0.5, 0.5], dtype=np.float32) for i in range(10)]
        scores = archive.compute_novelty_scores(behaviors)
        self.assertEqual(scores.shape, (10,))
        self.assertTrue(np.all(scores > 0.0))
        added = archive.update_archive(behaviors, scores)
        self.assertGreater(added, 0)

        obj = np.linspace(0.0, 12.0, 10, dtype=np.float32)
        hybrid = archive.combine_fitness(obj, scores, mode="hybrid")
        self.assertEqual(hybrid.shape, (10,))

    def test_map_elites_grid_archive(self) -> None:
        cfg = NEATConfig.from_ini("configs/neat_feedforward.ini")
        rng = np.random.default_rng(5)
        tracker = InnovationTracker(cfg.num_outputs + cfg.num_hidden)
        g1 = create_random_genome(1, cfg, tracker, rng)
        g2 = create_random_genome(2, cfg, tracker, rng)

        archive = GridArchive(bins=(20, 20))
        self.assertTrue(archive.add(g1, 5.0, (0.25, 0.75)))
        self.assertFalse(archive.add(g2, 3.0, (0.25, 0.75)))  # lower fitness rejected
        self.assertTrue(archive.add(g2, 8.5, (0.25, 0.75)))   # higher fitness replaces
        self.assertTrue(archive.add(g1, 4.0, (0.80, 0.10)))   # distinct niche inserted
        self.assertEqual(len(archive.cells), 2)
        self.assertAlmostEqual(archive.qd_score, 12.5, places=4)
        mat = archive.to_heatmap_matrix()
        self.assertEqual(mat.shape, (20, 20))

    def test_3gen_evolution_and_json_export_determinism(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tcfg = TrainingConfig(
                experiment_id="SMOKE",
                neat_config_path="configs/neat_recurrent.ini",
                evolution_mode="hybrid",
                comm_enabled=True,
                use_map_elites=True,
                generations=3,
                pop_size=12,
                mazes_per_gen=2,
                max_steps=40,
                seed=77,
                num_workers=2,
                output_dir=str(Path(tmp) / "runs"),
                db_path=str(Path(tmp) / "telemetry.db"),
            )
            res = run_training(tcfg, verbose=False)
            self.assertTrue(res.champion_path.exists())
            self.assertTrue(res.replay_path.exists())
            self.assertIsNotNone(res.archive_path)
            self.assertTrue(res.archive_path.exists())
            self.assertGreater(res.qd_coverage, 0.0)

            # Reload JSON champion and confirm deterministic replay score
            reloaded, meta = load_genome_json(res.champion_path)
            self.assertEqual(meta["experiment_id"], "SMOKE")
            maze = generate_maze(size=11, maze_type="deceptive", seed=123)
            scfg = SwarmConfig(num_agents=3, max_steps=50, comm_enabled=True, goal_cue_mode="compass")
            m1, _ = rollout_genome_on_maze(res.champion, maze, scfg)
            m2, _ = rollout_genome_on_maze(reloaded, maze, scfg)
            self.assertEqual(m1.solved, m2.solved)
            self.assertEqual(m1.makespan, m2.makespan)
            self.assertAlmostEqual(m1.coverage, m2.coverage, places=6)


if __name__ == "__main__":
    unittest.main()
