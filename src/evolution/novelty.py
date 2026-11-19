"""Behavioral characterization and k-NN Novelty Search archive using scipy.spatial.cKDTree."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.spatial import cKDTree

EvolutionMode = Literal["objective", "novelty", "hybrid"]


class NoveltyArchive:
    """Bounded behavioral archive with fast k-NN sparsity scoring."""

    def __init__(
        self,
        k: int = 15,
        max_size: int = 500,
        novelty_threshold: float = 0.20,
        hybrid_alpha: float = 0.50,
    ) -> None:
        self.k = int(k)
        self.max_size = int(max_size)
        self.novelty_threshold = float(novelty_threshold)
        self.hybrid_alpha = float(hybrid_alpha)
        self.archive: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self.archive)

    def compute_novelty_scores(self, pop_behaviors: list[np.ndarray]) -> np.ndarray:
        """Compute mean Euclidean distance to k nearest neighbors in (population U archive)."""
        if not pop_behaviors:
            return np.zeros(0, dtype=np.float32)

        pop_arr = np.stack(pop_behaviors, axis=0).astype(np.float64)
        if self.archive:
            arch_arr = np.stack(self.archive, axis=0).astype(np.float64)
            combined = np.vstack([pop_arr, arch_arr])
        else:
            combined = pop_arr

        k_eff = min(self.k + 1, len(combined))  # +1 because query includes self at distance 0
        if k_eff <= 1:
            return np.ones(len(pop_arr), dtype=np.float32)

        tree = cKDTree(combined)
        dists, _ = tree.query(pop_arr, k=k_eff)
        if dists.ndim == 1:
            dists = dists[:, None]
        # Exclude self-match (column 0)
        mean_dists = np.mean(dists[:, 1:], axis=1)
        return mean_dists.astype(np.float32)

    def update_archive(self, pop_behaviors: list[np.ndarray], novelty_scores: np.ndarray) -> int:
        """Insert sufficiently novel behaviors into the bounded archive."""
        added = 0
        for b, score in zip(pop_behaviors, novelty_scores):
            if float(score) >= self.novelty_threshold or len(self.archive) < self.k:
                self.archive.append(b.astype(np.float32).copy())
                added += 1
        if len(self.archive) > self.max_size:
            # Keep most recent bounded window
            self.archive = self.archive[-self.max_size :]
        return added

    def combine_fitness(
        self,
        objective_scores: np.ndarray,
        novelty_scores: np.ndarray,
        mode: EvolutionMode = "hybrid",
    ) -> np.ndarray:
        """Return selection fitness according to objective, novelty, or hybrid mode."""
        obj = np.asarray(objective_scores, dtype=np.float64)
        nov = np.asarray(novelty_scores, dtype=np.float64)
        if mode == "objective":
            return obj.astype(np.float32)
        if mode == "novelty":
            return nov.astype(np.float32)

        # Hybrid: min-max normalize both signals to [0, 1] and blend, plus keep solve bonus
        obj_span = max(1e-6, float(np.max(obj) - np.min(obj)))
        nov_span = max(1e-6, float(np.max(nov) - np.min(nov)))
        obj_norm = (obj - float(np.min(obj))) / obj_span
        nov_norm = (nov - float(np.min(nov))) / nov_span
        blended = (1.0 - self.hybrid_alpha) * obj_norm + self.hybrid_alpha * nov_norm
        # Preserve raw objective bonus for solved genomes so fast solvers refine
        solve_boost = np.where(obj >= 10.0, 0.5 + 0.05 * (obj - 10.0), 0.0)
        return (blended + solve_boost).astype(np.float32)
