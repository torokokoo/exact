# EXAMM: Evolutionary eXploration of Augmenting Memory Models
## Presentation Summary

---

## 1. What Is EXAMM?

**ELI5**: Imagine you're trying to design the best possible brain for predicting tomorrow's weather. Instead of designing it by hand, you let evolution do the work: start with simple brains, train them a bit, combine the best ones, mutate them, and repeat — thousands of times. That's EXAMM. It evolves recurrent neural networks (RNNs) that get better at predicting time series data over generations.

**Technical summary**: EXAMM is a neuroevolution framework that evolves RNN topologies and weights simultaneously. It uses island-based population management, NEAT-style crossover (innovation number alignment), and Lamarckian weight inheritance to efficiently explore the space of memory-augmented neural architectures.

---

## 2. RNN Nodes and Genome Representation

### 2.1 What Is a Genome?

**ELI5**: A genome is a blueprint for a brain. It describes which neurons exist, how they're wired together, and what values (weights) each connection holds. Two types of connections exist: forward edges (normal signal flow) and recurrent edges (signals that loop back in time).

**Technical**: An `RNN_Genome` holds:
- `vector<RNN_Node_Interface*> nodes` — one per neuron
- `vector<RNN_Edge*> edges` — feedforward connections
- `vector<RNN_Recurrent_Edge*> recurrent_edges` — time-delayed connections (depth 1–10)
- `vector<double> best_parameters` — flat weight vector, best snapshot during training
- `string structural_hash` — identity fingerprint for deduplication

```cpp
// rnn/rnn_genome.hxx:253
void get_mu_sigma(const vector<double>& p, double& mu, double& sigma);

// rnn/rnn_genome.cxx:1492
void RNN_Genome::assign_reachability() {
    // Marks every node/edge as forward_reachable and/or backward_reachable.
    // Only nodes reachable from inputs AND reaching outputs are active.
    // Computes structural_hash from active innovation numbers.
}
```

---

### 2.2 Node Types

**ELI5**: Not all neurons are the same. Some are simple (just do tanh), others are complex gated units (LSTM, GRU) that can selectively remember or forget things from the past.

| Node | Gates | Weights | Memory Mechanism |
|------|-------|---------|-----------------|
| Simple (`RNN_Node`) | None | 1 (bias) | `tanh(input + bias)` |
| LSTM | Input, Forget, Output + Cell | 11 | Explicit cell state, long-range memory |
| GRU | Update, Reset | 9 | Gated hidden state blending |
| MGU | Forget only | 6 | Simplified GRU (1 gate) |
| UGRNN | Update | 6 | Linear interpolation between candidate and previous |
| Delta | Adaptive | 6 | Frequency-based; `alpha`, `beta1`, `beta2` offsets |

**LSTM forward pass** (the full computation every time step):
```
// rnn/rnn_nodes.md — LSTM_Node forward
o[t] = sigmoid(output_gate_weight*x + output_gate_update_weight*c_prev + output_gate_bias)
i[t] = sigmoid(input_gate_weight*x  + input_gate_update_weight*c_prev  + input_gate_bias)
f[t] = sigmoid(forget_gate_weight*x + forget_gate_update_weight*c_prev + forget_gate_bias + 1.0)
g[t] = tanh(cell_weight*x + cell_bias)

cell[t]   = f[t]*c_prev + i[t]*g[t]
output[t] = o[t] * cell[t]
```
Note: `+1.0` bias trick on the forget gate helps preserve gradients at the start of training.

---

### 2.3 How Nodes Fire — The Synchronization Protocol

**ELI5**: Each neuron waits until it has received signals from ALL its input connections before computing. This is like waiting for all ingredients before cooking. No explicit scheduling needed — neurons trigger themselves.

**Technical**: Nodes use counters instead of a topological sort:
```cpp
// rnn/rnn_node_interface.hxx:101
vector<int32_t> inputs_fired;   // incremented when an edge delivers a value
vector<int32_t> outputs_fired;  // incremented when a downstream edge delivers a gradient

// Forward: fires computation when inputs_fired[t] == total_inputs
// Backward: fires backprop  when outputs_fired[t] == total_outputs
```

---

### 2.4 Recurrent Edges — Memory Across Time

**ELI5**: Normal edges carry information forward through layers. Recurrent edges carry information *back in time* — a neuron at time step `t` can send its output to another neuron at step `t+3`, for example. This is how the network "remembers" what happened before.

```
// rnn/rnn_nodes.md — RNN_Recurrent_Edge
Forward: input_node->output_values[t]  →  output_node at time t + recurrent_depth
Backward: error at time t  →  flows back to input_node at time t - recurrent_depth
```

Recurrent depth is configurable: `1..max_recurrent_depth` (default up to 10).

---

### 2.5 Backpropagation Through Time (BPTT)

**ELI5**: Training is like giving the brain a test, seeing how wrong it was, and then nudging all the connection strengths in the direction that reduces the error — going backwards through time to assign blame.

**Technical flow** (`rnn/rnn_genome.cxx:1181`):
```
backpropagate_stochastic():
  1. set_weights(initial_parameters)         ← start from birth weights
  2. evaluate on validation → initial_fitness_before_bp
  3. for epoch in 0..bp_iterations:
       a. shuffle training series
       b. for each series:
            forward_pass → calculate_error_mse → backward_pass → get_gradients
            weight_update->update(params, gradient)   ← Adam, RMSProp, etc.
       c. evaluate on validation
       d. if best so far: snapshot best_parameters
  4. set_weights(best_parameters)            ← restore best checkpoint
```

Loss function:
```
MSE = (1/T) * Σ_t (predicted[t] - expected[t])²
delta[t] = (predicted[t] - expected[t]) * (2/T)   ← injected into output nodes
```

---

## 3. Island-Based Evolution

### 3.1 The Big Picture

**ELI5**: Instead of having one big pool of genomes competing, EXAMM splits the population into isolated islands — like the Galapagos. Each island evolves independently, preserving diversity. Occasionally, the weakest islands get wiped out (extinction), and their territory is repopulated using the best survivors from other islands.

**Technical**: EXAMM uses multiple `Island` objects, each holding up to `max_size` genomes sorted best→worst. The main loop:

```cpp
// multithreaded/examm_mt.cxx:28,45-65
mutex examm_mutex;

// Worker thread body:
examm_mutex.lock();
RNN_Genome* genome = examm->generate_genome();   // pick island, create child
examm_mutex.unlock();

genome->backpropagate_stochastic(...);            // train — FULLY PARALLEL, no lock

examm_mutex.lock();
examm->insert_genome(genome);                     // update island population
examm_mutex.unlock();
```

Generate and insert are serialized; training is parallel across all worker threads.

---

### 3.2 Island State Machine

```
INITIALIZING ──── reaches max_size ────► FILLED
                                            │
REPOPULATING ◄─── extinction event ─────────┘
     │
     └──── re-fills ────► FILLED
```

- **INITIALIZING**: Single mutations of the seed genome fill the island from scratch.
- **FILLED**: Normal evolution — mutation and crossover.
- **REPOPULATING**: Island was erased; new genomes come from survivors on other islands.

```cpp
// examm/island.hxx:27,40,50
int32_t erased_generation_id;          // stale genome guard
int32_t status;                        // INITIALIZING=0, FILLED=1, REPOPULATING=2
const static int32_t REPOPULATING = 2;

// examm/island.cxx:430
if (genome->get_generation_id() <= erased_generation_id) {
    // discard — genome was generated before the island was wiped
}
```

---

### 3.3 Generating a Child Genome

**ELI5**: To make a new candidate brain, EXAMM either:
- **Mutates** one parent (adds a neuron, removes a connection, etc.)
- **Crosses over** two parents (merge both blueprints, keeping the best parts)

Parent selection cycles round-robin across islands.

**Mutation operations**:
| Operation | Effect |
|-----------|--------|
| `clone` | Copy parent weights unchanged |
| `add_edge` | New feedforward connection |
| `add_recurrent_edge` | New time-delayed connection |
| `add_node(Type)` | Insert a new neuron + 2 edges |
| `split_edge(Type)` | Replace one edge with node + 2 edges |
| `enable/disable_edge` | Toggle existing connection |
| `split_node / merge_node` | Structural reshaping |

**Crossover by innovation number**:
```
// examm/examm.cxx — crossover()
Edges aligned by global innovation_number:

MATCHING (both parents have it):
    t = Uniform[-0.5, 1.5]
    child_weight = t*(p2_weight - p1_weight) + p1_weight   ← allows extrapolation

DISJOINT (fitter parent only):    include with probability more_fit_crossover_rate
DISJOINT (weaker parent only):    include with probability less_fit_crossover_rate
```

---

### 3.4 Inserting a Genome — Structural Deduplication

**ELI5**: Before accepting a new genome into the island, EXAMM checks its "fingerprint" (structural hash). If an identical topology already exists in the island and it's better, the new one is discarded — no point keeping worse duplicates.

```cpp
// examm/island.cxx:430 — insert_genome()
// Step 1: discard if generated before island erasure
// Step 2: discard if worse than island's worst member
// Step 3: check structural_hash for exact topology duplicates
// Step 4: binary-search insert position (sorted by fitness)
// Step 5: evict the worst genome if island is over capacity
```

Structural hash:
```
structural_hash = node_hash + "_" + edge_hash + "_" + recurrent_edge_hash
// Each hash = sum of innovation_numbers of reachable+enabled components
// rnn/rnn_genome.cxx:1492
```

---

### 3.5 Parent Selection Strategies

Three strategies are available:

**Random** (default): uniform pick from island genomes.

**Harada** (frequency-based, `--is_harada_selection`):
```
// Prefer underexplored genomes
sort genomes by search_frequency ascending
pick from top floor(island_size * harada_ratio) genomes
selected_genome.search_frequency += 1
```

**SWEET** (Selection While Evaluating, `--is_sweet`):
```
// Include in-flight (still-training) genomes as crossover candidates
eval_vec = currently_evaluating_genomes
if |eval_vec| >= 2: pick both parents from eval_vec
elif |eval_vec| == 1: one parent from island, one from eval_vec
else: both from island
```

---

### 3.6 Extinction Events

**ELI5**: Every N generations, the worst-performing islands get completely wiped out. Their space is then repopulated using survivors from the better islands. This forces the search to not get stuck in local optima.

Triggered every `extinction_event_generation_number` insertions (`examm/island_speciation_strategy.cxx:247`):
```
Rank islands by best genome fitness (ascending = worst first)
For worst K islands:
    island.genomes.clear()
    island.status = REPOPULATING
    island.erased_generation_id = current_generation_id   ← guards against stale genomes
```

Repopulation methods (`--repopulation_method`):
| Method | Source |
|--------|--------|
| `bestParents` | Crossover of best genomes from surviving islands |
| `randomParents` | Random crossover from surviving islands |
| `bestGenome` | Mutate the global best |
| `bestIsland` | Copy + mutate every genome from the best island |

---

## 4. Lamarckian Weight Initialization

### 4.1 The Biological Analogy

**ELI5**: In regular evolution, a giraffe that spends its life stretching its neck can't pass a "longer neck" to its children — only genes matter. But Lamarck (a pre-Darwin biologist) believed acquired traits *could* be inherited. EXAMM uses this idea: a parent genome that trained its weights successfully shares those weight statistics with its children. New neurons and connections in the child are initialized not randomly, but based on what the parent learned.

**Technical**: When a child genome has new structural components (new nodes, new edges) not present in any parent, those components are initialized by sampling from:
```
N(μ_parent, σ_parent)
```
where μ and σ are computed from the parent's `best_parameters` (the trained weight vector).

---

### 4.2 Three Independent Weight Strategies

```cpp
// weights/weight_rules.hxx:9
enum WeightType { RANDOM=0, XAVIER=1, KAIMING=2, LAMARCKIAN=3, GP=4, NONE=-1 };

// weights/weight_rules.cxx:8-9 — defaults
weight_inheritance        = LAMARCKIAN;   // how child inherits from parent
mutated_components_weight = LAMARCKIAN;   // how new parts added by mutation are initialized
// weight_initialize = XAVIER (initial population, no parents)
```

| Flag | Controls |
|------|---------|
| `--weight_initialize` | Initial population (no parents) |
| `--weight_inheritance` | Crossover child's inherited weights |
| `--mutated_component_weight` | New parts added by mutation |

**Constraint**: `weight_initialize` cannot be `LAMARCKIAN` (no parents exist yet).

---

### 4.3 Computing μ and σ

**ELI5**: Take all the trained weights of the parent (flattened into a list), compute their average and spread. That's it — simple statistics become the "genetic memory" passed to the next generation.

```cpp
// rnn/rnn_genome.cxx:1714
void RNN_Genome::get_mu_sigma(const vector<double>& p, double& mu, double& sigma) {
    // mu = mean(p),  clamped element-wise to [-10, 10] before summing
    mu /= p.size();

    // sigma = sample std dev
    sigma /= (p.size() - 1);
    sigma = sqrt(sigma);
}
// rnn/rnn_genome.hxx:253
```

The flat `params` vector contains (in order): all `RNN_Edge` weights → all `RNN_Recurrent_Edge` weights → all node internal weights.

---

### 4.4 Initialization Methods Compared

| Method | Formula | Use Case |
|--------|---------|---------|
| **Random** | `Uniform[-1, 1]` | Baseline |
| **Xavier** | `sqrt(6/(fan_in+fan_out)) * Uniform[-1,1]` | Stable variance across layers |
| **Kaiming** | `sqrt(2/fan_in) * Normal(0,1)` | ReLU-like activations |
| **Lamarckian** | `bound(Normal(μ_parent, σ_parent))` | Inherit parent's weight scale |
| **GP** | `1.0` for all weights | EXA-GP identity bootstrap |

Lamarckian example — GRU node:
```cpp
// rnn/gru_node.cxx:34
void GRU_Node::initialize_lamarckian(
    minstd_rand0& generator, NormalDistribution& normal_distribution, double mu, double sigma
) {
    zw = bound(normal_distribution.random(generator, mu, sigma));
    zu = bound(normal_distribution.random(generator, mu, sigma));
    z_bias = bound(normal_distribution.random(generator, mu, sigma));
    rw = bound(normal_distribution.random(generator, mu, sigma));
    // ... all 9 GRU weights sampled from N(mu, sigma)
}
```

---

### 4.5 Lamarckian Flow in Each Scenario

**Mutation of an existing genome**:
```
// examm/examm.cxx:561,637,643
mu, sigma = get_mu_sigma(parent.best_parameters)

new_node.initialize_lamarckian(generator, dist, mu, sigma)
new_edge.weight = bound(Normal(mu, sigma))
// Pre-existing weights on the cloned parent are NEVER touched
```

**Crossover of two parents**:
```
// examm/examm.cxx:1020-1032
p1_mu, p1_sigma = get_mu_sigma(p1.best_parameters)
p2_mu, p2_sigma = get_mu_sigma(p2.best_parameters)

MATCHING edges   → blended: t*p2_w + (1-t)*p1_w,  t ~ Uniform[-0.5, 1.5]
DISJOINT edges   → direct copy from source parent
NEW components   → N(p1_mu, p1_sigma)   [fitter parent's statistics]
// examm/examm.cxx:1244,1259
```

**Summary table**:

| Situation | Matched weights | Unmatched weights | New components |
|-----------|----------------|-------------------|----------------|
| Initial population | — | — | Xavier/Kaiming/Random |
| Mutation (Lamarckian) | unchanged (parent copy) | unchanged | `N(μ_parent, σ_parent)` |
| Crossover (Lamarckian) | blended from both parents | direct from source | `N(μ_p1, σ_p1)` |
| Crossover (non-Lamarckian) | overwritten | overwritten | `weight_initialize` strategy |

---

## 5. Putting It All Together — One Generation

```
1. Worker thread calls generate_genome()
   ├── Island INITIALIZING → mutate seed genome once, Xavier init
   ├── Island FILLED       → mutate or crossover parents
   └── Island REPOPULATING → pull genomes from surviving islands

2. Genome receives Lamarckian weights for any new structural parts

3. backpropagate_stochastic() runs:
   ├── BPTT for bp_iterations epochs
   ├── Adam/RMSProp/etc. updates weights
   └── Snapshots best_parameters on validation improvement

4. insert_genome():
   ├── Check erased_generation_id (discard if stale)
   ├── Check structural_hash (discard if duplicate + worse)
   ├── Insert sorted by fitness, evict worst if over capacity
   └── Check extinction trigger → erase worst islands if due

5. Next generation: child's best_parameters become μ/σ source
   for ITS children's new structural components
```

---

## 6. Key Files for Code Snippets

| Topic | File | Key Lines |
|-------|------|-----------|
| Node synchronization protocol | `rnn/rnn_node_interface.hxx` | 101–102 |
| LSTM forward/backward | `rnn/lstm_node.cxx` | full file |
| GRU Lamarckian init | `rnn/gru_node.cxx` | 34–48 |
| Genome backprop loop | `rnn/rnn_genome.cxx` | 1181+ |
| μ/σ computation | `rnn/rnn_genome.cxx` | 1714–1755 |
| assign_reachability / structural hash | `rnn/rnn_genome.cxx` | 1492+ |
| Weight type enum & defaults | `weights/weight_rules.hxx`, `weight_rules.cxx` | 9, 8–9 |
| Island state machine & erasure | `examm/island.hxx`, `island.cxx` | 27–50, 430, 674 |
| Main evolution loop (mutex) | `multithreaded/examm_mt.cxx` | 28, 45–65 |
| Crossover (μ/σ from parents) | `examm/examm.cxx` | 1020–1032, 1244–1259 |
| Weight validity check | `examm/examm.cxx` | 1287–1298 |
| Extinction trigger | `examm/island_speciation_strategy.cxx` | 247 |
