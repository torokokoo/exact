"""Configuration dataclass for SNN-EXAMM evolution runs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    """All hyperparameters for an SNN-EXAMM evolution run."""

    # ── RL Environment ──────────────────────────────────────────────────
    env_name: str = "CartPole-v1"
    n_eval_episodes: int = 5
    t_sim: int = 15
    max_steps_per_episode: int = 500

    # ── Evolution ───────────────────────────────────────────────────────
    number_islands: int = 10
    island_size: int = 10
    max_genomes: int = 2000
    number_threads: int = 4

    mutation_rate: float = 0.3
    intra_island_crossover_rate: float = 0.7
    more_fit_crossover_rate: float = 0.5
    less_fit_crossover_rate: float = 0.5
    num_mutations: int = 1

    # ── Extinction ──────────────────────────────────────────────────────
    extinction_event_generation_number: int = 0  # 0 = disabled
    islands_to_exterminate: int = 1
    repopulation_method: str = "bestGenome"

    # ── Weight Initialization ───────────────────────────────────────────
    weight_initialize: str = "xavier"
    mutated_component_weight: str = "lamarckian"
    weight_inheritance: str = "lamarckian"

    # ── Network Topology ────────────────────────────────────────────────
    min_recurrent_depth: int = 1
    max_recurrent_depth: int = 10

    # ── Output ──────────────────────────────────────────────────────────
    output_directory: str = "snn_output"
    log_interval: int = 10  # print stats every N genomes
