"""Custom 2D MAP-Elites Quality Diversity (QD) GridArchive for variable-topology NEAT genomes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.evolution.neat_core import Genome


@dataclass
class ArchiveCell:
    row: int
    col: int
    fitness: float
    descriptor: tuple[float, float]
    genome: Genome


class GridArchive:
    """2D behavioral grid indexing variable-topology NEAT genomes.

    Axis 0 (rows): Team Spread / Dispersion in [0, 1].
    Axis 1 (cols): Signal Activity (or Exploration Efficiency) in [0, 1].
    """

    def __init__(self, bins: tuple[int, int] = (20, 20)) -> None:
        self.rows = int(bins[0])
        self.cols = int(bins[1])
        self.cells: dict[tuple[int, int], ArchiveCell] = {}

    def _to_index(self, descriptor: tuple[float, float]) -> tuple[int, int]:
        d0 = float(np.clip(descriptor[0], 0.0, 0.999999))
        d1 = float(np.clip(descriptor[1], 0.0, 0.999999))
        r = int(d0 * self.rows)
        c = int(d1 * self.cols)
        return (r, c)

    def add(self, genome: Genome, fitness: float, descriptor: tuple[float, float]) -> bool:
        """Insert genome if its niche is empty or if it improves on the niche's current elite."""
        key = self._to_index(descriptor)
        existing = self.cells.get(key)
        if existing is None or fitness > existing.fitness:
            self.cells[key] = ArchiveCell(
                row=key[0],
                col=key[1],
                fitness=float(fitness),
                descriptor=(float(descriptor[0]), float(descriptor[1])),
                genome=genome.copy(),
            )
            return True
        return False

    @property
    def coverage(self) -> float:
        """Fraction of total archive cells populated."""
        return float(len(self.cells)) / float(self.rows * self.cols)

    @property
    def qd_score(self) -> float:
        """Sum of positive elite fitness values across all populated cells."""
        return float(sum(max(0.0, cell.fitness) for cell in self.cells.values()))

    def best_elite(self) -> ArchiveCell | None:
        if not self.cells:
            return None
        return max(self.cells.values(), key=lambda c: c.fitness)

    def to_heatmap_matrix(self) -> np.ndarray:
        """Export a (rows, cols) float32 matrix with NaN for empty cells."""
        mat = np.full((self.rows, self.cols), np.nan, dtype=np.float32)
        for (r, c), cell in self.cells.items():
            mat[r, c] = cell.fitness
        return mat

    def save_summary(self, path: str | Path) -> Path:
        """Persist archive metadata and 2D heatmap matrix to JSON."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        mat = self.to_heatmap_matrix()
        payload: dict[str, Any] = {
            "rows": self.rows,
            "cols": self.cols,
            "populated_cells": len(self.cells),
            "coverage": self.coverage,
            "qd_score": self.qd_score,
            "heatmap": [[None if np.isnan(v) else round(float(v), 4) for v in row] for row in mat],
        }
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        return out
