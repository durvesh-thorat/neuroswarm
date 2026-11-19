"""Local-radius multi-agent communication bus with test-time ablation modes."""

from __future__ import annotations

from typing import Literal

import numpy as np

CommMode = Literal["normal", "zero", "shuffle"]


class CommBus:
    """Manages local-radius numeric signal broadcasting between swarm agents.

    Signals emitted at tick t are received by teammates within broadcast radius
    on tick t+1. Supports three modes for causal communication verification:
      - "normal": standard local-radius mean broadcast delivery
      - "zero": received signals are strictly zeroed out
      - "shuffle": received signals are resampled/permuted from the historical
        non-zero broadcast pool to preserve marginal magnitude distribution
        while destroying teammate-specific temporal/spatial information
    """

    def __init__(
        self,
        num_agents: int,
        comm_dim: int = 2,
        default_radius: float = 4.0,
        comm_enabled: bool = True,
        agent_radii: np.ndarray | None = None,
        activity_threshold: float = 0.25,
        comm_mode: CommMode = "normal",
        shuffle_seed: int = 0,
    ) -> None:
        self.num_agents = num_agents
        self.comm_dim = comm_dim
        self.default_radius = float(default_radius)
        self.comm_enabled = bool(comm_enabled)
        self.comm_mode: CommMode = comm_mode if self.comm_enabled else "zero"
        self.shuffle_seed = int(shuffle_seed)
        self.activity_threshold = float(activity_threshold)
        if agent_radii is not None:
            self.agent_radii = np.asarray(agent_radii, dtype=np.float32)
        else:
            self.agent_radii = np.full(num_agents, self.default_radius, dtype=np.float32)

        self.last_broadcasts = np.zeros((num_agents, comm_dim), dtype=np.float32)
        self._history_pool: list[np.ndarray] = []
        self._rng = np.random.default_rng(self.shuffle_seed)
        self.total_active_broadcasts: int = 0
        self.total_ticks: int = 0

    def reset(self) -> None:
        self.last_broadcasts.fill(0.0)
        self._history_pool.clear()
        self._rng = np.random.default_rng(self.shuffle_seed)
        self.total_active_broadcasts = 0
        self.total_ticks = 0

    def broadcast(self, signals: np.ndarray) -> np.ndarray:
        """Record emitted signals (clipped to [-1, 1]) for delivery on the next tick."""
        clipped = np.clip(np.asarray(signals, dtype=np.float32), -1.0, 1.0)
        if not self.comm_enabled or self.comm_mode == "zero":
            self.last_broadcasts.fill(0.0)
            self.total_ticks += 1
            return self.last_broadcasts.copy()

        self.last_broadcasts = clipped
        if self.comm_mode == "shuffle":
            for row in clipped:
                self._history_pool.append(row.copy())
        active_agents = int(np.sum(np.max(np.abs(clipped), axis=1) >= self.activity_threshold))
        self.total_active_broadcasts += active_agents
        self.total_ticks += 1
        return clipped

    def receive(self, positions: np.ndarray) -> np.ndarray:
        """Compute aggregated received signal [s0, s1] for each agent from in-range teammates."""
        received = np.zeros((self.num_agents, self.comm_dim), dtype=np.float32)
        if not self.comm_enabled or self.comm_mode == "zero" or self.num_agents <= 1:
            return received

        # Pairwise Euclidean distances: shape (N, N)
        diff = positions[:, None, :].astype(np.float32) - positions[None, :, :].astype(np.float32)
        dists = np.sqrt(np.sum(diff * diff, axis=-1))

        for i in range(self.num_agents):
            in_range = (dists[i] <= self.agent_radii) & (np.arange(self.num_agents) != i)
            count = int(np.sum(in_range))
            if count > 0:
                if self.comm_mode == "shuffle" and self._history_pool:
                    idxs = self._rng.integers(0, len(self._history_pool), size=count)
                    sampled = np.stack([self._history_pool[int(k)] for k in idxs], axis=0)
                    # Also randomly flip channel order/signs across channels within observed pool
                    received[i] = np.mean(sampled, axis=0)
                else:
                    received[i] = np.mean(self.last_broadcasts[in_range], axis=0)
        return received

    @property
    def signal_activity_ratio(self) -> float:
        """Fraction of agent-ticks with non-zero active communication broadcasts."""
        denom = max(1, self.total_ticks * self.num_agents)
        return float(self.total_active_broadcasts) / float(denom)
