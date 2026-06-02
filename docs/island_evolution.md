# Island-Based Evolutionary Algorithm — Inner Workings

## Data Structures

### Island (`examm/island.hxx`)
```
Island {
    id, max_size
    vector<RNN_Genome*> genomes          ← sorted best→worst by fitness
    unordered_map<string, vector<RNN_Genome*>> structure_map  ← dedup by structural hash
    status: INITIALIZING | FILLED | REPOPULATING
    unordered_map<int32_t, RNN_Genome*> evaluating_genomes    ← for SWEET (in-flight genomes)
    erased_generation_id                 ← prevents re-insertion of old genomes after extinction
}
```

### EXAMM controller (`examm/examm.hxx`)
Holds `SpeciationStrategy*` (island or NEAT), global innovation counters (`edge_innovation_count`, `node_innovation_count`), mutation rates, and per-operation statistics.

---

## High-Level Flow

```
main()
  ├── load time-series data
  ├── create EXAMM + islands
  └── spawn N worker threads, each running:

        while true:
            LOCK examm_mutex
            genome = examm->generate_genome()       ← picks island, creates child
            UNLOCK examm_mutex

            genome->backpropagate_stochastic(...)    ← train (NO lock, fully parallel)

            LOCK examm_mutex
            examm->insert_genome(genome)             ← update island populations
            UNLOCK examm_mutex
```

Generation and insertion are the only serialized sections. Training is fully parallel across threads.

---

## Island State Machine

Each island has three states that control how new genomes are created:

```
INITIALIZING  ─── island.size() reaches max_size ──►  FILLED
                                                           │
REPOPULATING ◄────── extinction event erases island ───────┘
     │
     └── (re-fills, then transitions back to FILLED)
```

---

## `generate_genome()` — Detailed Flow

Called while holding the mutex. Iterates islands in **round-robin** order.

```
generate_genome():
    island = islands[generation_island]

    if island.INITIALIZING:
        ── generate_for_initializing_island()
           Create a minimal genome via single mutation of the seed,
           then immediately insert a copy into the island.

    if island.FILLED:
        ── generate_for_filled_island()
           r = random [0,1)
           r < mutation_rate          → copy one genome from island, mutate it
           r < intra_island_xover     → crossover two parents from SAME island
           else                       → crossover two parents from DIFFERENT islands

    if island.REPOPULATING:
        ── generate_for_repopulating_island()
           depends on --repopulation_method (see Extinction section)

    genome.generation_id = ++generated_genomes
    genome.group_id = generation_island

    generation_island = (generation_island + 1) % number_islands
    return genome
```

---

## Parent Selection Strategies

All strategies pick parents from the `genomes` vector (sorted best→worst, index 0 = best).

### 1. Random (default)
```
p1 = genomes[ floor(size * rand) ]
p2 = genomes[ floor((size-1) * rand) ]   ← different index from p1
```
Uniform probability — no fitness bias beyond inclusion in island.

### 2. Harada (frequency-based, `--is_harada_selection`)
Reduces evaluation-time bias by deprioritizing over-explored genomes:
```
sort genomes by search_frequency (ascending)
pool_size = floor(island_size * harada_selection_ratio)
pick p1, p2 from genomes[0 .. pool_size-1]
p1.search_frequency += 1
p2.search_frequency += 1
```
Genomes selected as parents get their frequency incremented; future selections
will prefer genomes with lower frequency.

### 3. SWEET (Selection While Evaluating, `--is_sweet`)
Includes in-flight (not-yet-returned) genomes as crossover candidates:
```
eval_vec = list of evaluating_genomes

if eval_vec empty:
    fallback → pick both parents from island
elif eval_vec.size >= 2:
    pick both parents from eval_vec
else (1 in-flight):
    p1 = island genome, p2 = eval_vec[0]

parent1 = smaller genome (by weight count)   ← crossover convention
parent2 = larger genome
```
SWEET and Harada can be combined.

---

## Mutation Operations (`examm/examm.cxx::mutate()`)

`mutate(max_mutations, genome)` applies up to `max_mutations` randomly selected operations:

| Operation | What it does |
|-----------|-------------|
| `clone` | No structural change; carries parent weights as-is |
| `add_edge` | Add forward edge between existing nodes |
| `add_recurrent_edge` | Add recurrent edge with random time-skip [min_depth, max_depth] |
| `enable_edge` / `disable_edge` | Toggle an existing edge |
| `split_edge(NodeType)` | Replace one edge with node + 2 edges |
| `add_node(NodeType)` | Insert a node with incoming/outgoing edges |
| `enable_node` / `disable_node` | Toggle an existing node |
| `split_node(NodeType)` | Split one node into two |
| `merge_node(NodeType)` | Collapse two adjacent nodes into one |

`NodeType` is sampled from `possible_node_types` (simple, LSTM, GRU, MGU, UGRNN, delta, jordan, elman for EXAMM; sum, multiply, sin, cos, tanh, sigmoid, inverse for EXA-GP).

### Phased Evolution (`--growth_phase_genomes` / `--reduction_phase_genomes`)
When both are set, mutations cycle between two phases:
```
cycle_length = growth_phase_genomes + reduction_phase_genomes

GROWTH phase  (first growth_phase_genomes genomes of cycle):
    add_*/enable_* rates = 1,   disable_*/merge_* rates = 0

REDUCTION phase (remaining genomes):
    add_*/enable_* rates = 0,   disable_*/merge_* rates = 1
```

---

## Crossover (`examm/examm.cxx::crossover()`)

Genomes are aligned by **innovation number** (global counter incremented each time a new node or edge type is created anywhere in the search).

```
sort p1.edges and p2.edges by innovation_number

merge loop:
    if same innovation → MATCHING: blend weights, include in child
    if p1 only         → DISJOINT/EXCESS from fitter parent:
                         include with probability more_fit_crossover_rate
    if p2 only         → DISJOINT/EXCESS from less fit parent:
                         include with probability less_fit_crossover_rate

Same logic repeated for recurrent edges.

child = new RNN_Genome(merged nodes, merged edges, merged recurrent edges)
```

Weight blending for matching edges:
```
t = Uniform[-0.5, 1.5]
child_weight = t * (p2_weight - p1_weight) + p1_weight
```
Values of `t` outside [0,1] produce extrapolation beyond both parents.

---

## `insert_genome()` — Detailed Flow

```
insert_genome(genome):
    1. if genome.generation_id <= island.erased_generation_id → discard
    2. if island.FILLED and genome.fitness > island.worst_fitness → discard
    3. compute structural_hash(genome)
    4. check structure_map for exact duplicates:
       - if duplicate found and old is better → discard new
       - if duplicate found and new is better → remove old, proceed
    5. binary-search sorted `genomes` for insertion position
    6. if insert_position >= max_size → discard (would be at tail)
    7. insert copy at position; update structure_map
    8. if genomes.size() > max_size → delete genomes.back() (worst)
    9. return insert_position  (0 = new global best for this island)
```

---

## Extinction & Repopulation

Triggered inside `insert_genome()` every `extinction_event_generation_number` insertions (when `> 0`).

### Ranking islands
Islands ranked by their **best genome's fitness** (ascending = worst first). Islands recently erased are excluded unless `--repeat_extinction` is set.

### Erasure
```
for i in 0..islands_to_exterminate:
    worst_island = rank[i]
    worst_island.genomes.clear()
    worst_island.structure_map.clear()
    worst_island.status = REPOPULATING
    worst_island.erased_generation_id = current_generation_id
```

`erased_generation_id` ensures that any genome that was generated from this island before the erasure (and is still being trained by a worker) will be discarded on insertion.

### Repopulation methods (`--repopulation_method`)

| Method | How new genomes are created |
|--------|----------------------------|
| `bestParents` | Crossover of best genomes from non-erased islands |
| `randomParents` | Crossover of random genomes from non-erased islands |
| `bestGenome` | Mutate the global best genome |
| `bestIsland` | Copy every genome from the best island, each mutated once |

---

## Termination

The search ends when either condition is met (checked before each `generate_genome()`):
- `evaluated_genomes >= max_genomes` (if `max_genomes > 0`)
- elapsed wall-clock time >= `max_wallclock_seconds` (if `> 0`)

`generate_genome()` returns `NULL` → worker thread exits.

---

## Output Files

Written to `--output_directory`:

| File | Contents |
|------|----------|
| `fitness_log.csv` | One row per inserted genome: time, fitness, island, network size, generated_by |
| `global_best_genome_<id>.{bin,txt,gv}` | Best genome seen so far, updated on each improvement |
| `rnn_genome_<id>.json` | Full genome JSON (only with `--generate_visualization_json`) |
| `completed` | Empty sentinel; only created on clean exit |
