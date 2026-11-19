"""Pure-Python/NumPy NEAT neuroevolution core (compatible with NEAT .ini configs & JSON export).

Implements variable-topology neuroevolution with:
- Innovation-tracked connections and node genes
- Structural mutations (add/delete node, add/delete connection) & weight/bias mutations
- Compatibility-distance speciation with elitism and stagnation control
- Fast FeedForwardNetwork (topological sort) and RecurrentNetwork (with explicit .reset())
- Multiprocessing ParallelEvaluator
- Human-readable JSON genome serialization & deserialization
"""

from __future__ import annotations

import configparser
import copy
import json
import math
import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass
class NEATConfig:
    """Parsed NEAT hyperparameters from a standard NEAT .ini file."""

    pop_size: int = 60
    num_inputs: int = 15
    num_hidden: int = 2
    num_outputs: int = 7
    feed_forward: bool = True
    initial_conn_prob: float = 0.5

    bias_init_mean: float = 0.0
    bias_init_stdev: float = 0.5
    bias_min_value: float = -5.0
    bias_max_value: float = 5.0
    bias_mutate_power: float = 0.35
    bias_mutate_rate: float = 0.6
    bias_replace_rate: float = 0.08

    weight_init_mean: float = 0.0
    weight_init_stdev: float = 0.8
    weight_min_value: float = -5.0
    weight_max_value: float = 5.0
    weight_mutate_power: float = 0.45
    weight_mutate_rate: float = 0.8
    weight_replace_rate: float = 0.1

    conn_add_prob: float = 0.35
    conn_delete_prob: float = 0.10
    node_add_prob: float = 0.15
    node_delete_prob: float = 0.05

    compat_disjoint_coeff: float = 1.0
    compat_weight_coeff: float = 0.5
    compat_threshold: float = 3.0

    max_stagnation: int = 15
    species_elitism: int = 2
    elitism: int = 2
    survival_threshold: float = 0.25

    @classmethod
    def from_ini(cls, path: str | Path, pop_size_override: int | None = None) -> "NEATConfig":
        parser = configparser.ConfigParser()
        with open(path, "r", encoding="utf-8") as fh:
            parser.read_file(fh)

        neat_sec = parser["NEAT"]
        gen_sec = parser["DefaultGenome"]
        sp_sec = parser["DefaultSpeciesSet"]
        stag_sec = parser["DefaultStagnation"]
        rep_sec = parser["DefaultReproduction"]

        init_conn = gen_sec.get("initial_connection", "partial_direct 0.5").split()
        conn_prob = float(init_conn[1]) if len(init_conn) > 1 else 1.0

        return cls(
            pop_size=pop_size_override if pop_size_override is not None else neat_sec.getint("pop_size", 60),
            num_inputs=gen_sec.getint("num_inputs", 15),
            num_hidden=gen_sec.getint("num_hidden", 2),
            num_outputs=gen_sec.getint("num_outputs", 7),
            feed_forward=gen_sec.getboolean("feed_forward", True),
            initial_conn_prob=conn_prob,
            bias_init_mean=gen_sec.getfloat("bias_init_mean", 0.0),
            bias_init_stdev=gen_sec.getfloat("bias_init_stdev", 0.5),
            bias_min_value=gen_sec.getfloat("bias_min_value", -5.0),
            bias_max_value=gen_sec.getfloat("bias_max_value", 5.0),
            bias_mutate_power=gen_sec.getfloat("bias_mutate_power", 0.35),
            bias_mutate_rate=gen_sec.getfloat("bias_mutate_rate", 0.6),
            bias_replace_rate=gen_sec.getfloat("bias_replace_rate", 0.08),
            weight_init_mean=gen_sec.getfloat("weight_init_mean", 0.0),
            weight_init_stdev=gen_sec.getfloat("weight_init_stdev", 0.8),
            weight_min_value=gen_sec.getfloat("weight_min_value", -5.0),
            weight_max_value=gen_sec.getfloat("weight_max_value", 5.0),
            weight_mutate_power=gen_sec.getfloat("weight_mutate_power", 0.45),
            weight_mutate_rate=gen_sec.getfloat("weight_mutate_rate", 0.8),
            weight_replace_rate=gen_sec.getfloat("weight_replace_rate", 0.1),
            conn_add_prob=gen_sec.getfloat("conn_add_prob", 0.35),
            conn_delete_prob=gen_sec.getfloat("conn_delete_prob", 0.10),
            node_add_prob=gen_sec.getfloat("node_add_prob", 0.15),
            node_delete_prob=gen_sec.getfloat("node_delete_prob", 0.05),
            compat_disjoint_coeff=gen_sec.getfloat("compatibility_disjoint_coefficient", 1.0),
            compat_weight_coeff=gen_sec.getfloat("compatibility_weight_coefficient", 0.5),
            compat_threshold=sp_sec.getfloat("compatibility_threshold", 3.0),
            max_stagnation=stag_sec.getint("max_stagnation", 15),
            species_elitism=stag_sec.getint("species_elitism", 2),
            elitism=rep_sec.getint("elitism", 2),
            survival_threshold=rep_sec.getfloat("survival_threshold", 0.25),
        )


@dataclass
class NodeGene:
    key: int
    bias: float
    activation: str = "tanh"


@dataclass
class ConnectionGene:
    in_node: int
    out_node: int
    weight: float
    enabled: bool
    innovation: int

    @property
    def key(self) -> tuple[int, int]:
        return (self.in_node, self.out_node)


class InnovationTracker:
    """Tracks structural innovation numbers and new hidden node IDs across generations."""

    def __init__(self, start_node_id: int) -> None:
        self.next_node_id = start_node_id
        self.next_innovation = 1
        self._conn_innovations: dict[tuple[int, int], int] = {}

    def get_innovation(self, in_node: int, out_node: int) -> int:
        key = (in_node, out_node)
        if key not in self._conn_innovations:
            self._conn_innovations[key] = self.next_innovation
            self.next_innovation += 1
        return self._conn_innovations[key]

    def new_node_id(self) -> int:
        nid = self.next_node_id
        self.next_node_id += 1
        return nid


@dataclass
class Genome:
    """Variable-topology NEAT genome."""

    genome_id: int
    num_inputs: int
    num_outputs: int
    feed_forward: bool
    nodes: dict[int, NodeGene] = field(default_factory=dict)
    connections: dict[tuple[int, int], ConnectionGene] = field(default_factory=dict)
    fitness: float = 0.0
    objective_fitness: float = 0.0
    behavior: list[float] = field(default_factory=list)
    descriptor: tuple[float, float] = (0.0, 0.0)

    @property
    def input_keys(self) -> list[int]:
        return [-i - 1 for i in range(self.num_inputs)]

    @property
    def output_keys(self) -> list[int]:
        return list(range(self.num_outputs))

    @property
    def hidden_keys(self) -> list[int]:
        return [k for k in self.nodes if k >= self.num_outputs]

    @property
    def num_enabled_connections(self) -> int:
        return sum(1 for c in self.connections.values() if c.enabled)

    def copy(self, new_id: int | None = None) -> "Genome":
        g = copy.deepcopy(self)
        if new_id is not None:
            g.genome_id = new_id
        return g

    def to_dict(self) -> dict[str, Any]:
        """Serialize genome to human-readable JSON-compatible dict."""
        return {
            "format": "neuroswarm-neat-json-v2",
            "genome_id": self.genome_id,
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "feed_forward": self.feed_forward,
            "fitness": float(self.fitness),
            "objective_fitness": float(self.objective_fitness),
            "descriptor": [float(self.descriptor[0]), float(self.descriptor[1])],
            "nodes": [
                {
                    "key": n.key,
                    "bias": round(float(n.bias), 6),
                    "activation": n.activation,
                }
                for _, n in sorted(self.nodes.items())
            ],
            "connections": [
                {
                    "in_node": c.in_node,
                    "out_node": c.out_node,
                    "weight": round(float(c.weight), 6),
                    "enabled": bool(c.enabled),
                    "innovation": int(c.innovation),
                }
                for _, c in sorted(self.connections.items())
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Genome":
        nodes = {
            int(n["key"]): NodeGene(
                key=int(n["key"]),
                bias=float(n["bias"]),
                activation=str(n.get("activation", "tanh")),
            )
            for n in data["nodes"]
        }
        connections = {
            (int(c["in_node"]), int(c["out_node"])): ConnectionGene(
                in_node=int(c["in_node"]),
                out_node=int(c["out_node"]),
                weight=float(c["weight"]),
                enabled=bool(c["enabled"]),
                innovation=int(c["innovation"]),
            )
            for c in data["connections"]
        }
        desc = data.get("descriptor", [0.0, 0.0])
        return cls(
            genome_id=int(data.get("genome_id", 0)),
            num_inputs=int(data["num_inputs"]),
            num_outputs=int(data["num_outputs"]),
            feed_forward=bool(data["feed_forward"]),
            nodes=nodes,
            connections=connections,
            fitness=float(data.get("fitness", 0.0)),
            objective_fitness=float(data.get("objective_fitness", 0.0)),
            descriptor=(float(desc[0]), float(desc[1])),
        )


def save_genome_json(genome: Genome, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
    """Export a NEAT genome to human-readable JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = genome.to_dict()
    if metadata:
        payload["metadata"] = metadata
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return out_path


def load_genome_json(path: str | Path) -> tuple[Genome, dict[str, Any]]:
    """Load a NEAT genome and optional metadata from JSON."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    meta = dict(data.get("metadata", {}))
    return Genome.from_dict(data), meta


def _creates_cycle(connections: dict[tuple[int, int], ConnectionGene], in_node: int, out_node: int) -> bool:
    """Return True if adding edge (in_node -> out_node) creates a directed cycle."""
    if in_node == out_node:
        return True
    visited = {out_node}
    stack = [out_node]
    while stack:
        cur = stack.pop()
        for (src, dst), gene in connections.items():
            if not gene.enabled:
                continue
            if src == cur:
                if dst == in_node:
                    return True
                if dst not in visited:
                    visited.add(dst)
                    stack.append(dst)
    return False


def create_random_genome(
    genome_id: int,
    cfg: NEATConfig,
    tracker: InnovationTracker,
    rng: np.random.Generator,
) -> Genome:
    """Initialize a genome with output nodes, optional initial hidden nodes, and partial connections."""
    g = Genome(
        genome_id=genome_id,
        num_inputs=cfg.num_inputs,
        num_outputs=cfg.num_outputs,
        feed_forward=cfg.feed_forward,
    )
    for out_k in g.output_keys:
        bias = float(np.clip(rng.normal(cfg.bias_init_mean, cfg.bias_init_stdev), cfg.bias_min_value, cfg.bias_max_value))
        g.nodes[out_k] = NodeGene(key=out_k, bias=bias)

    hidden_ids: list[int] = []
    for h_idx in range(cfg.num_hidden):
        hk = cfg.num_outputs + h_idx
        if hk >= tracker.next_node_id:
            tracker.next_node_id = hk + 1
        bias = float(np.clip(rng.normal(cfg.bias_init_mean, cfg.bias_init_stdev), cfg.bias_min_value, cfg.bias_max_value))
        g.nodes[hk] = NodeGene(key=hk, bias=bias)
        hidden_ids.append(hk)

    # Connect inputs -> hidden and hidden -> outputs, plus direct input -> output
    sources_l1 = g.input_keys
    targets_l1 = hidden_ids + g.output_keys
    for src in sources_l1:
        for dst in targets_l1:
            if rng.random() < cfg.initial_conn_prob:
                w = float(np.clip(rng.normal(cfg.weight_init_mean, cfg.weight_init_stdev), cfg.weight_min_value, cfg.weight_max_value))
                innov = tracker.get_innovation(src, dst)
                g.connections[(src, dst)] = ConnectionGene(src, dst, w, True, innov)

    for h in hidden_ids:
        for dst in g.output_keys:
            if rng.random() < max(0.6, cfg.initial_conn_prob):
                w = float(np.clip(rng.normal(cfg.weight_init_mean, cfg.weight_init_stdev), cfg.weight_min_value, cfg.weight_max_value))
                innov = tracker.get_innovation(h, dst)
                g.connections[(h, dst)] = ConnectionGene(h, dst, w, True, innov)
        # In recurrent mode, seed initial self-memory loop with moderate probability
        if not cfg.feed_forward and rng.random() < 0.5:
            w = float(np.clip(rng.normal(0.0, 0.5), cfg.weight_min_value, cfg.weight_max_value))
            innov = tracker.get_innovation(h, h)
            g.connections[(h, h)] = ConnectionGene(h, h, w, True, innov)

    return g


def mutate_genome(
    genome: Genome,
    cfg: NEATConfig,
    tracker: InnovationTracker,
    rng: np.random.Generator,
) -> None:
    """Apply NEAT structural and parameter mutations in-place."""
    # 1. Node bias mutations
    for node in genome.nodes.values():
        r = rng.random()
        if r < cfg.bias_mutate_rate:
            node.bias = float(np.clip(node.bias + rng.normal(0.0, cfg.bias_mutate_power), cfg.bias_min_value, cfg.bias_max_value))
        elif r < cfg.bias_mutate_rate + cfg.bias_replace_rate:
            node.bias = float(np.clip(rng.normal(cfg.bias_init_mean, cfg.bias_init_stdev), cfg.bias_min_value, cfg.bias_max_value))

    # 2. Connection weight mutations
    for conn in genome.connections.values():
        r = rng.random()
        if r < cfg.weight_mutate_rate:
            conn.weight = float(np.clip(conn.weight + rng.normal(0.0, cfg.weight_mutate_power), cfg.weight_min_value, cfg.weight_max_value))
        elif r < cfg.weight_mutate_rate + cfg.weight_replace_rate:
            conn.weight = float(np.clip(rng.normal(cfg.weight_init_mean, cfg.weight_init_stdev), cfg.weight_min_value, cfg.weight_max_value))

    # 3. Structural mutation: add node (split an enabled connection)
    if rng.random() < cfg.node_add_prob:
        enabled_conns = [c for c in genome.connections.values() if c.enabled]
        if enabled_conns:
            chosen = enabled_conns[int(rng.integers(0, len(enabled_conns)))]
            chosen.enabled = False
            new_nid = tracker.new_node_id()
            genome.nodes[new_nid] = NodeGene(key=new_nid, bias=0.0)
            in_innov = tracker.get_innovation(chosen.in_node, new_nid)
            out_innov = tracker.get_innovation(new_nid, chosen.out_node)
            genome.connections[(chosen.in_node, new_nid)] = ConnectionGene(chosen.in_node, new_nid, 1.0, True, in_innov)
            genome.connections[(new_nid, chosen.out_node)] = ConnectionGene(new_nid, chosen.out_node, chosen.weight, True, out_innov)

    # 4. Structural mutation: add connection
    if rng.random() < cfg.conn_add_prob:
        possible_in = genome.input_keys + list(genome.nodes.keys())
        possible_out = list(genome.nodes.keys())
        for _ in range(10):
            src = possible_in[int(rng.integers(0, len(possible_in)))]
            dst = possible_out[int(rng.integers(0, len(possible_out)))]
            if (src, dst) in genome.connections:
                continue
            if genome.feed_forward and _creates_cycle(genome.connections, src, dst):
                continue
            w = float(np.clip(rng.normal(cfg.weight_init_mean, cfg.weight_init_stdev), cfg.weight_min_value, cfg.weight_max_value))
            innov = tracker.get_innovation(src, dst)
            genome.connections[(src, dst)] = ConnectionGene(src, dst, w, True, innov)
            break

    # 5. Structural mutation: delete connection
    if rng.random() < cfg.conn_delete_prob and len(genome.connections) > 2:
        keys = list(genome.connections.keys())
        del_key = keys[int(rng.integers(0, len(keys)))]
        del genome.connections[del_key]

    # 6. Structural mutation: delete hidden node
    if rng.random() < cfg.node_delete_prob:
        hidden = genome.hidden_keys
        if hidden:
            del_nid = hidden[int(rng.integers(0, len(hidden)))]
            del genome.nodes[del_nid]
            to_drop = [k for k in genome.connections if k[0] == del_nid or k[1] == del_nid]
            for k in to_drop:
                del genome.connections[k]


def crossover_genomes(
    parent1: Genome,
    parent2: Genome,
    child_id: int,
    rng: np.random.Generator,
) -> Genome:
    """NEAT crossover aligning genes by key/innovation, inheriting excess/disjoint from fitter parent."""
    if parent2.fitness > parent1.fitness:
        parent1, parent2 = parent2, parent1

    child = Genome(
        genome_id=child_id,
        num_inputs=parent1.num_inputs,
        num_outputs=parent1.num_outputs,
        feed_forward=parent1.feed_forward,
    )
    for k, n1 in parent1.nodes.items():
        if k in parent2.nodes and rng.random() < 0.5:
            child.nodes[k] = copy.deepcopy(parent2.nodes[k])
        else:
            child.nodes[k] = copy.deepcopy(n1)

    for k, c1 in parent1.connections.items():
        if k in parent2.connections and rng.random() < 0.5:
            child.connections[k] = copy.deepcopy(parent2.connections[k])
        else:
            child.connections[k] = copy.deepcopy(c1)

    return child


def genome_distance(g1: Genome, g2: Genome, cfg: NEATConfig) -> float:
    """Compute NEAT compatibility distance between two variable-topology genomes."""
    c1 = g1.connections
    c2 = g2.connections
    if not c1 and not c2:
        return 0.0
    keys1 = set(c1.keys())
    keys2 = set(c2.keys())
    shared = keys1 & keys2
    disjoint = len(keys1 ^ keys2)
    max_genes = max(len(c1), len(c2), 1)
    weight_diff = 0.0
    if shared:
        weight_diff = sum(abs(c1[k].weight - c2[k].weight) for k in shared) / len(shared)
    return (cfg.compat_disjoint_coeff * disjoint) / max_genes + cfg.compat_weight_coeff * weight_diff


class FeedForwardNetwork:
    """Compiled acyclic phenotype network with topological evaluation order."""

    def __init__(
        self,
        input_keys: list[int],
        output_keys: list[int],
        node_evals: list[tuple[int, float, list[tuple[int, float]]]],
    ) -> None:
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.node_evals = node_evals

    def reset(self) -> None:
        """No-op for stateless feed-forward networks (uniform API with RecurrentNetwork)."""

    def activate(self, inputs: np.ndarray | list[float]) -> np.ndarray:
        values: dict[int, float] = {}
        for k, v in zip(self.input_keys, inputs):
            values[k] = float(v)
        for node_key, bias, links in self.node_evals:
            s = bias
            for src, w in links:
                s += values.get(src, 0.0) * w
            values[node_key] = math.tanh(s)
        return np.array([values.get(k, 0.0) for k in self.output_keys], dtype=np.float32)

    @classmethod
    def create(cls, genome: Genome) -> "FeedForwardNetwork":
        incoming: dict[int, list[tuple[int, float]]] = {k: [] for k in genome.nodes}
        for c in genome.connections.values():
            if c.enabled and c.out_node in incoming:
                incoming[c.out_node].append((c.in_node, c.weight))

        # Topological sort (Kahn's algorithm) over non-input nodes
        input_set = set(genome.input_keys)
        in_degree: dict[int, int] = {k: 0 for k in genome.nodes}
        adj: dict[int, list[int]] = {k: [] for k in genome.nodes}
        for dst, links in incoming.items():
            for src, _ in links:
                if src not in input_set and src in adj:
                    adj[src].append(dst)
                    in_degree[dst] += 1

        ready = [k for k, deg in in_degree.items() if deg == 0]
        order: list[int] = []
        while ready:
            cur = ready.pop()
            order.append(cur)
            for nxt in adj[cur]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    ready.append(nxt)
        # Fallback for any remaining nodes
        for k in genome.nodes:
            if k not in order:
                order.append(k)

        node_evals = [(k, genome.nodes[k].bias, incoming[k]) for k in order]
        return cls(genome.input_keys, genome.output_keys, node_evals)


class RecurrentNetwork:
    """Compiled recurrent phenotype network with persistent hidden activations and explicit .reset()."""

    def __init__(
        self,
        input_keys: list[int],
        output_keys: list[int],
        node_evals: list[tuple[int, float, list[tuple[int, float]]]],
    ) -> None:
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.node_evals = node_evals
        self.values: dict[int, float] = {k: 0.0 for k, _, _ in node_evals}

    def reset(self) -> None:
        """Explicitly zero all recurrent node activations before each maze episode."""
        for k in self.values:
            self.values[k] = 0.0

    def activate(self, inputs: np.ndarray | list[float]) -> np.ndarray:
        prev = dict(self.values)
        for k, v in zip(self.input_keys, inputs):
            prev[k] = float(v)
        new_vals: dict[int, float] = {}
        for node_key, bias, links in self.node_evals:
            s = bias
            for src, w in links:
                s += prev.get(src, 0.0) * w
            new_vals[node_key] = math.tanh(s)
        self.values = new_vals
        return np.array([self.values.get(k, 0.0) for k in self.output_keys], dtype=np.float32)

    @classmethod
    def create(cls, genome: Genome) -> "RecurrentNetwork":
        incoming: dict[int, list[tuple[int, float]]] = {k: [] for k in genome.nodes}
        for c in genome.connections.values():
            if c.enabled and c.out_node in incoming:
                incoming[c.out_node].append((c.in_node, c.weight))
        node_evals = [(k, genome.nodes[k].bias, incoming[k]) for k in sorted(genome.nodes.keys())]
        return cls(genome.input_keys, genome.output_keys, node_evals)


def build_phenotype(genome: Genome) -> FeedForwardNetwork | RecurrentNetwork:
    """Compile a NEAT genome into its FeedForward or Recurrent phenotype."""
    if genome.feed_forward:
        return FeedForwardNetwork.create(genome)
    return RecurrentNetwork.create(genome)


@dataclass
class Species:
    species_id: int
    representative: Genome
    members: list[Genome] = field(default_factory=list)
    best_fitness: float = -float("inf")
    stagnation: int = 0


class NEATPopulation:
    """Manages speciation, fitness-proportionate reproduction, elitism, and structural evolution."""

    def __init__(self, cfg: NEATConfig, seed: int = 0) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.tracker = InnovationTracker(start_node_id=cfg.num_outputs + cfg.num_hidden)
        self._next_genome_id = 1
        self._next_species_id = 1
        self.species: list[Species] = []
        self.genomes: list[Genome] = [
            create_random_genome(self._alloc_genome_id(), cfg, self.tracker, self.rng)
            for _ in range(cfg.pop_size)
        ]
        self._speciate()

    def _alloc_genome_id(self) -> int:
        gid = self._next_genome_id
        self._next_genome_id += 1
        return gid

    def _speciate(self) -> None:
        for sp in self.species:
            sp.members = []
        for g in self.genomes:
            placed = False
            for sp in self.species:
                if genome_distance(g, sp.representative, self.cfg) < self.cfg.compat_threshold:
                    sp.members.append(g)
                    placed = True
                    break
            if not placed:
                sp = Species(species_id=self._next_species_id, representative=g.copy(), members=[g])
                self._next_species_id += 1
                self.species.append(sp)
        self.species = [sp for sp in self.species if sp.members]
        for sp in self.species:
            sp.representative = sp.members[int(self.rng.integers(0, len(sp.members)))].copy()

    def reproduce(self) -> None:
        """Advance population by one generation using speciation, elitism, crossover, and mutation."""
        # Update stagnation & sort members by fitness descending
        for sp in self.species:
            sp.members.sort(key=lambda g: g.fitness, reverse=True)
            sp_best = sp.members[0].fitness
            if sp_best > sp.best_fitness:
                sp.best_fitness = sp_best
                sp.stagnation = 0
            else:
                sp.stagnation += 1

        # Protect top species_elitism species from stagnation culling
        self.species.sort(key=lambda s: s.best_fitness, reverse=True)
        surviving_species: list[Species] = []
        for idx, sp in enumerate(self.species):
            if idx < self.cfg.species_elitism or sp.stagnation <= self.cfg.max_stagnation:
                surviving_species.append(sp)
        if not surviving_species:
            surviving_species = self.species[: max(1, self.cfg.species_elitism)]
        self.species = surviving_species

        # Compute adjusted fitness shares
        min_fit = min(g.fitness for sp in self.species for g in sp.members)
        shift = abs(min_fit) + 1e-3 if min_fit <= 0 else 0.0
        sp_means = [
            float(np.mean([g.fitness + shift for g in sp.members])) for sp in self.species
        ]
        total_mean = max(1e-6, sum(sp_means))

        next_gen: list[Genome] = []
        # Allocate offspring per species
        spawn_counts = [
            max(1, int(round(self.cfg.pop_size * (m / total_mean)))) for m in sp_means
        ]
        # Adjust exact total to cfg.pop_size
        diff = self.cfg.pop_size - sum(spawn_counts)
        if diff > 0:
            spawn_counts[0] += diff
        elif diff < 0:
            for i in range(len(spawn_counts) - 1, -1, -1):
                take = min(-diff, max(0, spawn_counts[i] - 1))
                spawn_counts[i] -= take
                diff += take
                if diff == 0:
                    break

        for sp, count in zip(self.species, spawn_counts):
            if count <= 0:
                continue
            # Elitism: copy top genomes directly
            n_elite = min(self.cfg.elitism, len(sp.members), count)
            for i in range(n_elite):
                next_gen.append(sp.members[i].copy(new_id=self._alloc_genome_id()))

            cutoff = max(1, int(math.ceil(len(sp.members) * self.cfg.survival_threshold)))
            parents = sp.members[:cutoff]
            for _ in range(count - n_elite):
                p1 = parents[int(self.rng.integers(0, len(parents)))]
                p2 = parents[int(self.rng.integers(0, len(parents)))]
                child = crossover_genomes(p1, p2, self._alloc_genome_id(), self.rng)
                mutate_genome(child, self.cfg, self.tracker, self.rng)
                next_gen.append(child)

        self.genomes = next_gen[: self.cfg.pop_size]
        self._speciate()


class ParallelEvaluator:
    """Multiprocessing genome evaluator compatible with CPU-bound headless rollouts."""

    def __init__(self, num_workers: int, eval_fn: Callable[[Any], Any]) -> None:
        self.num_workers = max(1, int(num_workers))
        self.eval_fn = eval_fn
        self._pool = mp.Pool(processes=self.num_workers) if self.num_workers > 1 else None

    def evaluate(self, tasks: list[Any]) -> list[Any]:
        if self._pool is None:
            return [self.eval_fn(t) for t in tasks]
        return self._pool.map(self.eval_fn, tasks)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self) -> "ParallelEvaluator":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
