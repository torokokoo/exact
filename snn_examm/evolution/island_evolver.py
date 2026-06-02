"""Multi-island evolution coordinator.

Port of examm/island_speciation_strategy.{hxx,cxx}.
Manages multiple islands, round-robin genome generation, extinction events,
and repopulation.
"""

from __future__ import annotations

import logging
import random
from typing import Callable, Optional

from ..genome.snn_genome import SNNGenome
from .island import Island, INITIALIZING, FILLED, REPOPULATING

logger = logging.getLogger(__name__)


class IslandEvolver:
    """Multi-island coordinator for island-based speciation.

    Port of IslandSpeciationStrategy.
    """

    def __init__(
        self,
        number_islands: int,
        island_size: int,
        seed_genome: SNNGenome,
        mutation_rate: float = 0.3,
        intra_island_crossover_rate: float = 0.7,
        inter_island_crossover_rate: float = 1.0,
        extinction_event_generation_number: int = 0,
        islands_to_exterminate: int = 1,
        num_mutations: int = 1,
        repopulation_method: str = "bestGenome",
        repeat_extinction: bool = True,
        max_genomes: int = -1,
    ):
        self.number_islands = number_islands
        self.island_size = island_size
        self.seed_genome = seed_genome
        self.mutation_rate = mutation_rate
        self.intra_island_crossover_rate = intra_island_crossover_rate
        self.inter_island_crossover_rate = inter_island_crossover_rate
        self.extinction_event_generation_number = extinction_event_generation_number
        self.islands_to_exterminate = islands_to_exterminate
        self.num_mutations = num_mutations
        self.repopulation_method = repopulation_method
        self.repeat_extinction = repeat_extinction
        self.max_genomes = max_genomes

        self.islands = [Island(i, island_size) for i in range(number_islands)]
        self.generation_island = 0  # round-robin counter
        self.generated_genomes = 0
        self.evaluated_genomes = 0

        self.global_best_genome: Optional[SNNGenome] = None

    def get_best_genome(self) -> Optional[SNNGenome]:
        return self.global_best_genome

    def _rank_islands(self) -> list[int]:
        """Rank islands from worst (index 0) to best (last index) by best fitness.

        Port of IslandSpeciationStrategy::rank_islands (island_speciation_strategy.cxx:261-293).
        Note: EXAMM sorts worst-first for MSE (ascending). For RL reward (descending),
        worst island = lowest best_fitness.
        """
        eligible = []
        for i in range(self.number_islands):
            if self.repeat_extinction or self.islands[i].erase_again_num == 0:
                eligible.append(i)

        # Sort by best fitness ascending (worst first)
        eligible.sort(key=lambda i: self.islands[i].get_best_fitness())
        return eligible

    def _repopulate(self) -> None:
        """Check if extinction event should fire and handle it.

        Port of IslandSpeciationStrategy::repopulate (island_speciation_strategy.cxx:232-259).
        """
        if self.extinction_event_generation_number <= 0:
            return

        have_room = (
            self.evaluated_genomes > 1
            and self.evaluated_genomes % self.extinction_event_generation_number == 0
            and (self.max_genomes <= 0 or self.max_genomes - self.evaluated_genomes >= self.extinction_event_generation_number)
        )

        if not have_room:
            return

        # Update global best before extinction
        best = self._find_global_best()
        if best:
            self.global_best_genome = best.copy()

        ranked = self._rank_islands()
        for i in range(min(self.islands_to_exterminate, len(ranked))):
            island_idx = ranked[i]
            logger.info(f"Extinction: erasing island {island_idx}")
            self.islands[island_idx].erase_island()

    def _find_global_best(self) -> Optional[SNNGenome]:
        """Find the best genome across all islands."""
        best = None
        for island in self.islands:
            gb = island.get_best_genome()
            if gb and (best is None or gb.fitness > best.fitness):
                best = gb
        return best

    def _number_filled_islands(self) -> int:
        return sum(1 for island in self.islands if island.is_full())

    def _get_other_filled_island(self, exclude: int) -> Optional[int]:
        """Get a random filled island that isn't the excluded one."""
        others = [i for i in range(self.number_islands)
                  if i != exclude and self.islands[i].is_full()]
        if not others:
            return None
        return random.choice(others)

    def generate_genome(
        self,
        mutate_fn: Callable[[int, SNNGenome], None],
        crossover_fn: Callable[[SNNGenome, SNNGenome], SNNGenome],
    ) -> Optional[SNNGenome]:
        """Generate the next genome via the round-robin island strategy.

        Port of IslandSpeciationStrategy::generate_genome (island_speciation_strategy.cxx:374-410).

        Args:
            mutate_fn: callable(num_mutations, genome) that mutates the genome in place.
            crossover_fn: callable(parent1, parent2) -> child genome.

        Returns:
            A new SNNGenome ready for evaluation, or None if search is complete.
        """
        if 0 < self.max_genomes <= self.generated_genomes:
            return None

        island = self.islands[self.generation_island]
        new_genome: Optional[SNNGenome] = None

        if island.is_initializing():
            new_genome = self._generate_for_initializing(island, mutate_fn)
        elif island.is_full():
            new_genome = self._generate_for_filled(island, mutate_fn, crossover_fn)
        elif island.is_repopulating():
            new_genome = self._generate_for_repopulating(island, mutate_fn, crossover_fn)

        if new_genome is None:
            # Shouldn't happen, but recurse with safety
            logger.warning(f"Island {self.generation_island}: genome is None, retrying")
            self.generation_island = (self.generation_island + 1) % self.number_islands
            return self.generate_genome(mutate_fn, crossover_fn)

        self.generated_genomes += 1
        new_genome.generation_id = self.generated_genomes
        new_genome.group_id = self.generation_island

        # For initializing islands, insert immediately (with initial fitness)
        if island.is_initializing():
            copy = new_genome.copy()
            self.insert_genome(copy)

        # Advance round-robin
        self.generation_island = (self.generation_island + 1) % self.number_islands

        return new_genome

    def _generate_for_initializing(
        self,
        island: Island,
        mutate_fn: Callable[[int, SNNGenome], None],
    ) -> SNNGenome:
        """Generate genome for an initializing island.

        Port of IslandSpeciationStrategy::generate_for_initializing_island
        (island_speciation_strategy.cxx:295-329).
        """
        if island.size() == 0:
            new_genome = self.seed_genome.copy()
            new_genome.initialize_randomly()
        else:
            new_genome = island.copy_random_genome()
            if new_genome is None:
                new_genome = self.seed_genome.copy()
                new_genome.initialize_randomly()
            mutate_fn(self.num_mutations, new_genome)

        new_genome.fitness = float("-inf")
        return new_genome

    def _generate_for_filled(
        self,
        island: Island,
        mutate_fn: Callable[[int, SNNGenome], None],
        crossover_fn: Callable[[SNNGenome, SNNGenome], SNNGenome],
    ) -> SNNGenome:
        """Generate genome for a filled island.

        Port of IslandSpeciationStrategy::generate_for_filled_island
        (island_speciation_strategy.cxx:443+).
        """
        r = random.random()

        if not island.is_full() or r < self.mutation_rate:
            # Mutation
            genome = island.copy_random_genome()
            if genome is None:
                genome = self.seed_genome.copy()
                genome.initialize_randomly()
            mutate_fn(self.num_mutations, genome)
            return genome

        elif r < self.intra_island_crossover_rate or self._number_filled_islands() <= 1:
            # Intra-island crossover
            p1, p2 = island.copy_two_random_genomes()
            if p1 is None or p2 is None:
                genome = island.copy_random_genome()
                if genome is None:
                    genome = self.seed_genome.copy()
                    genome.initialize_randomly()
                mutate_fn(self.num_mutations, genome)
                return genome
            return crossover_fn(p1, p2)

        else:
            # Inter-island crossover
            other_idx = self._get_other_filled_island(self.generation_island)
            if other_idx is None:
                # Fall back to intra-island
                p1, p2 = island.copy_two_random_genomes()
                if p1 is None or p2 is None:
                    genome = island.copy_random_genome()
                    if genome is None:
                        genome = self.seed_genome.copy()
                        genome.initialize_randomly()
                    mutate_fn(self.num_mutations, genome)
                    return genome
                return crossover_fn(p1, p2)

            p1 = island.copy_random_genome()
            p2 = self.islands[other_idx].copy_random_genome()
            if p1 is None or p2 is None:
                genome = island.copy_random_genome() or self.seed_genome.copy()
                mutate_fn(self.num_mutations, genome)
                return genome
            return crossover_fn(p1, p2)

    def _generate_for_repopulating(
        self,
        island: Island,
        mutate_fn: Callable[[int, SNNGenome], None],
        crossover_fn: Callable[[SNNGenome, SNNGenome], SNNGenome],
    ) -> SNNGenome:
        """Generate genome for a repopulating island.

        Port of IslandSpeciationStrategy::generate_for_repopulating_island
        (island_speciation_strategy.cxx:331-372).
        """
        method = self.repopulation_method.lower()

        if method == "bestgenome":
            if self.global_best_genome:
                new_genome = self.global_best_genome.copy()
            else:
                new_genome = self.seed_genome.copy()
                new_genome.initialize_randomly()
            mutate_fn(self.num_mutations, new_genome)
            return new_genome

        elif method in ("bestparents", "randomparents"):
            # Get parents from surviving islands
            surviving = [i for i in range(self.number_islands)
                         if self.islands[i].is_full()]
            if len(surviving) < 2:
                new_genome = (self.global_best_genome or self.seed_genome).copy()
                mutate_fn(self.num_mutations, new_genome)
                return new_genome

            i1, i2 = random.sample(surviving, 2)
            if method == "bestparents":
                p1 = self.islands[i1].get_best_genome()
                p2 = self.islands[i2].get_best_genome()
            else:
                p1 = self.islands[i1].copy_random_genome()
                p2 = self.islands[i2].copy_random_genome()

            if p1 and p2:
                child = crossover_fn(p1.copy(), p2.copy())
                mutate_fn(self.num_mutations, child)
                return child

            new_genome = (self.global_best_genome or self.seed_genome).copy()
            mutate_fn(self.num_mutations, new_genome)
            return new_genome

        elif method == "bestisland":
            best_island = max(
                (isl for isl in self.islands if isl.is_full()),
                key=lambda isl: isl.get_best_fitness(),
                default=None,
            )
            if best_island:
                for g in best_island.get_genomes():
                    copy = g.copy()
                    mutate_fn(1, copy)
                    island.insert_genome(copy)

            return self._generate_for_filled(island, mutate_fn, crossover_fn)

        else:
            new_genome = (self.global_best_genome or self.seed_genome).copy()
            mutate_fn(self.num_mutations, new_genome)
            return new_genome

    def insert_genome(self, genome: SNNGenome) -> int:
        """Insert an evaluated genome into its assigned island.

        Port of IslandSpeciationStrategy::insert_genome
        (island_speciation_strategy.cxx:170-212).

        Returns 0 if new global best, -1 if not inserted, >0 otherwise.
        """
        self._repopulate()

        new_global_best = False
        if self.global_best_genome is None:
            self.global_best_genome = genome.copy()
            new_global_best = True
        elif genome.fitness > self.global_best_genome.fitness:
            self.global_best_genome = genome.copy()
            new_global_best = True

        self.evaluated_genomes += 1

        island_idx = genome.group_id
        if island_idx < 0 or island_idx >= len(self.islands):
            logger.error(f"Invalid island index: {island_idx}")
            return -1

        insert_position = self.islands[island_idx].insert_genome(genome)

        if insert_position == 0:
            return 0 if new_global_best else 1
        return insert_position

    def get_stats(self) -> dict:
        """Return evolution statistics."""
        stats = {
            "generated": self.generated_genomes,
            "evaluated": self.evaluated_genomes,
            "global_best_fitness": self.global_best_genome.fitness if self.global_best_genome else None,
            "islands": [],
        }
        for island in self.islands:
            stats["islands"].append({
                "id": island.id,
                "size": island.size(),
                "status": island.status,
                "best_fitness": island.get_best_fitness() if island.genomes else None,
            })
        return stats
