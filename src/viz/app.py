"""Streamlit Interactive Replay & Ablation Analytics Lab for NeuroSwarm Maze Lab.

Launch locally with:
    streamlit run src/viz/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.evolution.neat_core import load_genome_json
from src.telemetry.logger import TelemetryLogger
from src.telemetry.replay import load_replay
from src.viz.generate_assets import _draw_maze_frame
from src.viz.topology_plot import plot_genome_topology

BENCH_DIR = Path("data/benchmarks")
RUNS_DIR = Path("data/runs")
DB_PATH = Path("data/telemetry.db")

st.set_page_config(page_title="NeuroSwarm Maze Lab", layout="wide", page_icon="🧠")
st.title("NeuroSwarm Maze Lab — Pure Neuroevolution Replay & Ablation Inspector")
st.caption("Zero LLMs · Variable-Topology NEAT · Fog-of-War Multi-Agent Swarms · Novelty Search & MAP-Elites")

if not (BENCH_DIR / "showcase_index.json").exists():
    st.warning("No benchmark data found. Run `python -m src.benchmarks.run_ablation_suite` first.")
    st.stop()

showcase = json.loads((BENCH_DIR / "showcase_index.json").read_text(encoding="utf-8"))
summary_df = pd.read_csv(BENCH_DIR / "ood_summary.csv")
by_arch_df = pd.read_csv(BENCH_DIR / "ood_by_archetype.csv")

tab_replay, tab_compare, tab_analytics, tab_topology = st.tabs(
    ["Episode Replay Player", "Side-by-Side Comparison", "Ablation Analytics & MAP-Elites", "Neural Topology"]
)

with tab_replay:
    col_sel1, col_sel2 = st.columns(2)
    maze_id = col_sel1.selectbox("Unseen OOD Maze", showcase["mazes"], key="single_maze")
    exp_id = col_sel2.selectbox("Experiment Champion", list(showcase["replays"][maze_id].keys()), index=3)
    replay = load_replay(showcase["replays"][maze_id][exp_id])
    max_t = len(replay["frames"]) - 1
    tick = st.slider("Simulation Tick", 0, max_t, 0, key="single_tick")

    c_map, c_info = st.columns([2, 1])
    with c_map:
        fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=120)
        fig.patch.set_facecolor("#0f172a")
        _draw_maze_frame(ax, replay, tick, f"{exp_id} on {maze_id}")
        st.pyplot(fig)
        plt.close(fig)
    with c_info:
        st.subheader("Episode Telemetry")
        m = replay["metrics"]
        st.metric("Solved", "Yes" if m["solved"] else "No")
        st.metric("Makespan (ticks)", f"{m['makespan']} / {m['max_steps']}")
        st.metric("Fog-of-War Coverage", f"{m['coverage'] * 100:.1f}%")
        st.metric("Signal Activity", f"{m['signal_activity'] * 100:.1f}%")
        st.write("**Agent 0 Live Sensors (15D):**")
        obs = replay["frames"][tick].get("obs_sample", [])
        if obs:
            st.bar_chart(pd.DataFrame({"Sensor Value": obs}))

with tab_compare:
    maze_cmp = st.selectbox("Unseen OOD Maze (Side-by-Side)", showcase["mazes"], key="cmp_maze")
    c_left, c_right = st.columns(2)
    exp_left = c_left.selectbox("Left Swarm", list(showcase["replays"][maze_cmp].keys()), index=0)
    exp_right = c_right.selectbox("Right Swarm", list(showcase["replays"][maze_cmp].keys()), index=3)
    rep_l = load_replay(showcase["replays"][maze_cmp][exp_left])
    rep_r = load_replay(showcase["replays"][maze_cmp][exp_right])
    max_cmp = max(len(rep_l["frames"]), len(rep_r["frames"])) - 1
    tick_cmp = st.slider("Synchronized Tick", 0, max_cmp, 0, key="cmp_tick")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.8), dpi=120)
    fig.patch.set_facecolor("#0f172a")
    _draw_maze_frame(ax1, rep_l, tick_cmp, f"{exp_left} ({maze_cmp})")
    _draw_maze_frame(ax2, rep_r, tick_cmp, f"{exp_right} ({maze_cmp})")
    st.pyplot(fig)
    plt.close(fig)

with tab_analytics:
    st.subheader("Zero-Shot 20-Maze OOD Ablation Table")
    st.dataframe(summary_df, use_container_width=True)
    st.subheader("Performance Breakdown by Maze Archetype")
    st.dataframe(by_arch_df, use_container_width=True)
    if DB_PATH.exists():
        logger = TelemetryLogger(DB_PATH)
        gen_df = pd.DataFrame(logger.fetch_generations())
        gen_df["experiment_id"] = gen_df["run_id"].str.split("_seed").str[0]
        pivot = gen_df.groupby(["generation", "experiment_id"])["max_fitness"].mean().unstack()
        st.subheader("Generation Fitness Curves (Mean Max Objective Fitness)")
        st.line_chart(pivot)

with tab_topology:
    champ_files = sorted(RUNS_DIR.glob("*/champion.json"))
    choice = st.selectbox("Select Saved JSON Champion", [p.parent.name for p in champ_files])
    g, meta = load_genome_json(RUNS_DIR / choice / "champion.json")
    fig = plot_genome_topology(g, title=f"{choice} ({meta.get('topology', 'NEAT')})")
    st.pyplot(fig)
    plt.close(fig)
