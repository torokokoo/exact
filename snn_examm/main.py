"""Main evolution loop for SNN-EXAMM.

Mirrors multithreaded/examm_mt.cxx: N worker threads, each doing
generate_genome() -> evaluate() -> insert_genome() with a shared lock.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import threading
import time
from typing import Optional

from .config import Config
from .evolution.engine import EvolverEngine
from .evaluation.rl_evaluator import RLEvaluator

logger = logging.getLogger(__name__)


def worker_thread(
    thread_id: int,
    engine: EvolverEngine,
    evaluator: RLEvaluator,
    lock: threading.Lock,
    config: Config,
) -> None:
    """Worker thread: generate -> evaluate -> insert loop.

    Direct port of examm_thread() from examm_mt.cxx:43-68.
    """
    while True:
        # Generate genome (under lock)
        with lock:
            genome = engine.generate_genome()

        if genome is None:
            break  # Search complete

        # Evaluate in RL environment (no lock needed — fully parallel)
        fitness = evaluator.evaluate(genome)
        genome.fitness = fitness
        genome.snapshot_best_parameters()

        # Insert back into population (under lock)
        with lock:
            insert_pos = engine.insert_genome(genome)

            stats = engine.get_stats()
            evaluated = stats["evaluated"]
            if evaluated % config.log_interval == 0:
                best = stats["global_best_fitness"]
                best_str = f"{best:.2f}" if best is not None else "N/A"
                logger.info(
                    f"[Thread {thread_id}] Evaluated: {evaluated}, "
                    f"Global best: {best_str}, "
                    f"This genome: {fitness:.2f} (island {genome.group_id}, pos {insert_pos})"
                )


def run_evolution(config: Config) -> Optional[dict]:
    """Run the full SNN-EXAMM evolution loop.

    Returns the final stats dictionary.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Create evaluator to get env dimensions
    evaluator = RLEvaluator(
        env_name=config.env_name,
        n_episodes=config.n_eval_episodes,
        t_sim=config.t_sim,
        max_steps_per_episode=config.max_steps_per_episode,
    )

    logger.info(
        f"Environment: {config.env_name} "
        f"(obs={evaluator.observation_size}, actions={evaluator.n_actions})"
    )

    # Create evolution engine
    engine = EvolverEngine(
        n_inputs=evaluator.observation_size,
        n_outputs=evaluator.n_actions,
        number_islands=config.number_islands,
        island_size=config.island_size,
        max_genomes=config.max_genomes,
        mutation_rate=config.mutation_rate,
        intra_island_crossover_rate=config.intra_island_crossover_rate,
        more_fit_crossover_rate=config.more_fit_crossover_rate,
        less_fit_crossover_rate=config.less_fit_crossover_rate,
        extinction_event_generation_number=config.extinction_event_generation_number,
        islands_to_exterminate=config.islands_to_exterminate,
        num_mutations=config.num_mutations,
        repopulation_method=config.repopulation_method,
        weight_initialize=config.weight_initialize,
        mutated_component_weight=config.mutated_component_weight,
        weight_inheritance=config.weight_inheritance,
        min_recurrent_depth=config.min_recurrent_depth,
        max_recurrent_depth=config.max_recurrent_depth,
    )

    logger.info(
        f"Starting evolution: {config.max_genomes} genomes, "
        f"{config.number_islands} islands x {config.island_size}, "
        f"{config.number_threads} threads"
    )
    logger.info(
        f"Weights: init={config.weight_initialize}, "
        f"mutated={config.mutated_component_weight}, "
        f"inheritance={config.weight_inheritance}"
    )

    lock = threading.Lock()
    start_time = time.time()

    # Launch worker threads
    threads = []
    for i in range(config.number_threads):
        t = threading.Thread(
            target=worker_thread,
            args=(i, engine, evaluator, lock, config),
            daemon=True,
        )
        threads.append(t)
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    elapsed = time.time() - start_time
    stats = engine.get_stats()

    logger.info(f"Evolution complete in {elapsed:.1f}s")
    logger.info(f"Evaluated: {stats['evaluated']} genomes")
    logger.info(f"Global best fitness: {stats['global_best_fitness']:.2f}")

    # Save results
    os.makedirs(config.output_directory, exist_ok=True)
    stats_path = os.path.join(config.output_directory, "evolution_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    best_genome = engine.get_best_genome()
    if best_genome:
        logger.info(
            f"Best genome: {best_genome.get_enabled_node_count()} hidden nodes, "
            f"{best_genome.get_enabled_edge_count()} edges, "
            f"{best_genome.get_enabled_recurrent_edge_count()} recurrent edges"
        )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="SNN-EXAMM: Neuroevolution of SNNs for RL")

    parser.add_argument("--env_name", type=str, default="CartPole-v1")
    parser.add_argument("--n_eval_episodes", type=int, default=5)
    parser.add_argument("--t_sim", type=int, default=15)
    parser.add_argument("--max_steps_per_episode", type=int, default=500)

    parser.add_argument("--number_islands", type=int, default=10)
    parser.add_argument("--island_size", type=int, default=10)
    parser.add_argument("--max_genomes", type=int, default=2000)
    parser.add_argument("--number_threads", type=int, default=4)

    parser.add_argument("--mutation_rate", type=float, default=0.3)
    parser.add_argument("--intra_island_crossover_rate", type=float, default=0.7)
    parser.add_argument("--more_fit_crossover_rate", type=float, default=0.5)
    parser.add_argument("--less_fit_crossover_rate", type=float, default=0.5)
    parser.add_argument("--num_mutations", type=int, default=1)

    parser.add_argument("--extinction_event_generation_number", type=int, default=0)
    parser.add_argument("--islands_to_exterminate", type=int, default=1)
    parser.add_argument("--repopulation_method", type=str, default="bestGenome")

    parser.add_argument("--weight_initialize", type=str, default="xavier")
    parser.add_argument("--mutated_component_weight", type=str, default="lamarckian")
    parser.add_argument("--weight_inheritance", type=str, default="lamarckian")

    parser.add_argument("--min_recurrent_depth", type=int, default=1)
    parser.add_argument("--max_recurrent_depth", type=int, default=10)

    parser.add_argument("--output_directory", type=str, default="snn_output")
    parser.add_argument("--log_interval", type=int, default=10)

    args = parser.parse_args()
    config = Config(**vars(args))
    run_evolution(config)


if __name__ == "__main__":
    main()
