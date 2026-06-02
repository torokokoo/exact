"""Island class for island-based speciation.

Port of examm/island.hxx and examm/island.cxx.
Maintains a sorted population of SNNGenome objects.
Key difference from EXAMM: fitness is maximized (higher reward = better),
not minimized (lower MSE = better).
"""

from __future__ import annotations

import random
from bisect import insort
from typing import Optional

from ..genome.snn_genome import SNNGenome

INITIALIZING = 0
FILLED = 1
REPOPULATING = 2


class Island:
    """A single island maintaining a fitness-sorted population.

    Genomes are sorted best (index 0, highest fitness) to worst (last index).
    """

    def __init__(self, island_id: int, max_size: int):
        self.id = island_id
        self.max_size = max_size
        self.genomes: list[SNNGenome] = []
        self.status = INITIALIZING
        self.erased_generation_id = -1
        self.latest_generation_id = -1
        self.erased = False
        self.erase_again_num = 0

        # Structural deduplication
        self._structure_map: dict[str, list[SNNGenome]] = {}

    def get_best_fitness(self) -> float:
        if not self.genomes:
            return float("-inf")
        return self.genomes[0].fitness

    def get_worst_fitness(self) -> float:
        if not self.genomes:
            return float("-inf")
        return self.genomes[-1].fitness

    def get_best_genome(self) -> Optional[SNNGenome]:
        if not self.genomes:
            return None
        return self.genomes[0]

    def size(self) -> int:
        return len(self.genomes)

    def is_full(self) -> bool:
        return len(self.genomes) >= self.max_size

    def is_initializing(self) -> bool:
        return self.status == INITIALIZING

    def is_repopulating(self) -> bool:
        return self.status == REPOPULATING

    def copy_random_genome(self) -> Optional[SNNGenome]:
        """Select a random genome and return a copy."""
        if not self.genomes:
            return None
        return random.choice(self.genomes).copy()

    def copy_two_random_genomes(self) -> tuple[Optional[SNNGenome], Optional[SNNGenome]]:
        """Select two different random genomes and return copies."""
        if len(self.genomes) < 2:
            return None, None
        g1, g2 = random.sample(self.genomes, 2)
        return g1.copy(), g2.copy()

    def insert_genome(self, genome: SNNGenome) -> int:
        """Insert a genome maintaining descending fitness order.

        Port of Island::insert_genome (island.cxx:428-645).
        Returns -1 if not inserted, 0 if new island best, >0 otherwise.
        """
        # Reject genomes from before island was erased
        if genome.generation_id <= self.erased_generation_id:
            return -1

        # Reject if island is full and genome is worse than worst
        if self.is_full() and genome.fitness <= self.get_worst_fitness():
            return -1

        # Structural duplicate check
        struct_hash = genome.get_structural_hash()
        if struct_hash in self._structure_map:
            for existing in self._structure_map[struct_hash]:
                if existing.fitness >= genome.fitness:
                    return -1  # duplicate with better or equal fitness exists
                else:
                    # New genome is better, remove old one
                    self.genomes.remove(existing)
                    self._structure_map[struct_hash].remove(existing)
                    break

        # Find insertion position (descending order: best first)
        insert_pos = 0
        for i, g in enumerate(self.genomes):
            if genome.fitness > g.fitness:
                insert_pos = i
                break
            insert_pos = i + 1

        if insert_pos >= self.max_size:
            return -1

        self.genomes.insert(insert_pos, genome)

        # Track in structure map
        if struct_hash not in self._structure_map:
            self._structure_map[struct_hash] = []
        self._structure_map[struct_hash].append(genome)

        # Evict worst if over capacity
        if len(self.genomes) > self.max_size:
            evicted = self.genomes.pop()
            evicted_hash = evicted.get_structural_hash()
            if evicted_hash in self._structure_map:
                if evicted in self._structure_map[evicted_hash]:
                    self._structure_map[evicted_hash].remove(evicted)

        # Update status
        if self.status == INITIALIZING and self.is_full():
            self.status = FILLED
        elif self.status == REPOPULATING and self.is_full():
            self.status = FILLED

        return insert_pos

    def erase_island(self) -> None:
        """Erase all genomes for extinction event.

        Port of Island::erase_island (island.cxx).
        """
        if self.genomes:
            self.erased_generation_id = max(g.generation_id for g in self.genomes)
        self.genomes.clear()
        self._structure_map.clear()
        self.erased = True
        self.erase_again_num += 1
        self.status = REPOPULATING

    def been_erased(self) -> bool:
        return self.erased

    def get_genomes(self) -> list[SNNGenome]:
        return self.genomes

    def __repr__(self) -> str:
        status_names = {INITIALIZING: "INIT", FILLED: "FULL", REPOPULATING: "REPOP"}
        best = f"{self.get_best_fitness():.4f}" if self.genomes else "N/A"
        return (
            f"Island(id={self.id}, size={self.size()}/{self.max_size}, "
            f"status={status_names.get(self.status, '?')}, best={best})"
        )
