# RNN Genome — Inner Workings

## `RNN_Genome` vs `RNN`

| Class | Role |
|-------|------|
| `RNN_Genome` | Evolutionary unit: owns topology + full training state (fitness, weight snapshots, genealogy, serialization) |
| `RNN` | Stateless execution wrapper: owns topology and buffers for a single forward/backward pass; created fresh per training series |

`RNN_Genome::backpropagate*()` internally creates `RNN` objects from its topology for each training series, runs them, and aggregates gradients back.

---

## Key Fields

### Identity and evolution metadata
```cpp
int32_t generation_id          // assigned by EXAMM on creation, monotonically increasing
int32_t group_id               // island ID this genome was generated from
double  search_frequency       // Harada: incremented each time this genome is selected as parent
vector<int32_t> parent_ids     // IDs of parent genomes
map<string, int32_t> generated_by_map  // e.g. {"add_node(LSTM)": 1, "crossover": 1}
```

### Weight snapshots
```cpp
vector<double> initial_parameters  // weights at genome birth (before backprop)
vector<double> best_parameters     // weights achieving best_validation_mse
```

### Fitness
```cpp
double best_validation_mse         // primary fitness metric (lower = better)
double best_validation_mae
double initial_fitness_before_bp   // validation MSE before any training
int64_t bp_time_milliseconds       // training wall-clock time
```

### Network structure
```cpp
vector<RNN_Node_Interface*> nodes
vector<RNN_Edge*> edges
vector<RNN_Recurrent_Edge*> recurrent_edges
string structural_hash             // computed by assign_reachability()
```

### Training configuration
```cpp
int32_t bp_iterations
string  backprop_iterations_type   // "const" | "random" | "scaled"
bool    use_dropout
double  dropout_probability
WeightRules* weight_rules
```

---

## `get_fitness()`

Returns `best_validation_mse`. Lower is better. EXAMM sorts island populations by this value ascending.

---

## Structural Hash

Computed in `assign_reachability()`:
```
structural_hash = to_string(node_hash) + "_" + to_string(edge_hash) + "_" + to_string(recurrent_edge_hash)

node_hash           = sum of innovation_numbers of all reachable+enabled nodes
edge_hash           = sum of innovation_numbers of all reachable+enabled edges
recurrent_edge_hash = sum of innovation_numbers of all reachable+enabled recurrent edges
```

Used by `Island::insert_genome()` to detect topologically identical genomes and keep only the fitter copy.

`assign_reachability()` also sets `forward_reachable` and `backward_reachable` on every node and edge — only nodes/edges reachable from inputs **and** reaching outputs participate in computation.

---

## Backpropagation Flow

### `backpropagate_stochastic(training_inputs, training_outputs, validation_inputs, validation_outputs, weight_update)`

```
1. set_weights(initial_parameters)       ← start from birth weights
2. evaluate on validation → initial_fitness_before_bp
3. for epoch in 0..bp_iterations:
     a. shuffle training series order
     b. for each training series s:
          create RNN from genome topology
          rnn->reset(series_length)
          rnn->set_weights(current_parameters)
          rnn->forward_pass(inputs[s])
          rnn->calculate_error_mse(outputs[s], mse_s, deltas_s)
          rnn->backward_pass(deltas_s)
          rnn->get_gradients(gradient)
          weight_update->update(current_parameters, gradient)    ← in-place weight update
     c. evaluate on validation data
     d. if validation_mse < best_validation_mse:
          best_validation_mse = validation_mse
          best_parameters = current_parameters
4. set_weights(best_parameters)
5. record bp_time_milliseconds
```

### `get_analytic_gradient()` (multi-threaded, used by non-stochastic variant)
- Spawns one thread per training series
- Each thread creates its own `RNN`, runs forward + backward, produces a gradient vector
- Gradients are summed across all series, then divided by the number of series

---

## Mutation Operations

All mutations modify the genome's node/edge vectors in-place. Innovation counters are passed by reference from EXAMM to ensure global uniqueness.

| Operation | What changes |
|-----------|-------------|
| `add_edge(mu, sigma, innovation)` | New `RNN_Edge` between two existing nodes; weight initialized per weight rules |
| `add_recurrent_edge(mu, sigma, depth_dist, innovation)` | New `RNN_Recurrent_Edge` with random depth |
| `enable_edge()` / `disable_edge()` | Toggles `enabled` flag on a random edge |
| `add_node(mu, sigma, type, depth_dist, edge_innov, node_innov)` | Inserts a node; adds two edges to connect it to the graph |
| `enable_node()` / `disable_node()` | Toggles `enabled` flag on a random hidden node |
| `split_edge(mu, sigma, type, ...)` | Replaces one edge with a node + two edges |
| `split_node(mu, sigma, type, ...)` | Duplicates a node, splitting its input connections |
| `merge_node(mu, sigma, type, ...)` | Collapses two connected nodes into one |

After any mutation that adds structural components, `assign_reachability()` is called to update reachability flags and recompute the structural hash.

### Node type selection during mutation
`possible_node_types` is sampled uniformly. EXAMM mode: `SIMPLE, JORDAN, ELMAN, UGRNN, MGU, GRU, DELTA, LSTM`. EXA-GP mode: `SIN, COS, SUM, MULTIPLY, INVERSE, TANH, SIGMOID` (or `*_GP` variants for EXA-GP-MIN).

---

## Serialization

`write_to_stream(ostream&)` / `read_from_stream(istream&)` implement binary round-trip serialization. Written in this order:

1. `generation_id`, `group_id`
2. `bp_iterations`, `use_dropout`, `dropout_probability`
3. `weight_rules` (init/inheritance/mutation strategies)
4. `generator` state (RNG)
5. Fitness metrics: `initial_fitness_before_bp`, `best_validation_mse`, `best_validation_mae`, `bp_time_milliseconds`
6. `parent_ids`, `generated_by_map`
7. `best_parameters`, `initial_parameters`
8. `input_parameter_names`, `output_parameter_names`
9. All nodes (each node serializes itself, including type tag)
10. All edges (innovation numbers + weight + enabled flag)
11. All recurrent edges (same + `recurrent_depth`)
12. Normalization bounds: `normalize_type`, `normalize_mins/maxs/avgs/std_devs`

On `read_from_stream()`, `assign_reachability()` is called at the end to reconstruct `structural_hash` and reachability flags (not stored; derived).

`write_to_array()` / `read_from_array()` wrap the same logic for MPI transmission (serializes to `char*`).

---

## `GenomeProperty` — Training Configuration

Parsed from CLI arguments and applied to new genomes before backpropagation:

```cpp
GenomeProperty* gp = generate_genome_property_from_arguments(arguments, time_series_sets);
gp->set_genome_properties(genome);
```

### `backprop_iterations_type` modes

| Mode | Behavior |
|------|---------|
| `"const"` | `bp_iterations` epochs for every genome |
| `"random"` | `Uniform[bp_min, bp_max]` epochs per genome |
| `"scaled"` | `min(bp_max, floor((bp_slope * generated_genomes)^bp_exponent) + bp_min)` — ramps up as more genomes are evaluated |

CLI: `--backprop_iterations_type`, `--bp_iterations`, `--bp_min`, `--bp_max`, `--bp_slope`, `--bp_exponent`

---

## `generate_nn.cxx` — Network Construction

Builds initial genome topologies:

```cpp
RNN_Genome* create_lstm(input_parameter_names, hidden_layers, hidden_nodes,
                         output_parameter_names, max_recurrent_depth, weight_rules)
// same for: create_gru, create_mgu, create_ugrnn, create_delta,
//           create_ff, create_elman, create_jordan,
//           create_sin, create_cos, create_sum, create_multiply, create_tanh,
//           create_sigmoid, create_inverse, create_dnas_nn
```

The template `create_nn()` function:
1. Creates `INPUT_LAYER` nodes for each input parameter name
2. Creates `number_hidden_layers × number_hidden_nodes` hidden nodes via `make_node` callback
3. Connects every node in layer `l` to every node in layer `l+1` with `RNN_Edge`
4. Adds `RNN_Recurrent_Edge` at depths `1..max_recurrent_depth` from every hidden/output node back to every node in earlier layers
5. Creates `OUTPUT_LAYER` nodes, fully connected from last hidden layer
6. Returns `RNN_Genome` with this topology and weight rules

`get_seed_genome()` either loads a pre-trained genome from `--genome_file` (transfer learning) or calls `create_ff()` with a single hidden node to produce a minimal starting point for evolution.

---

## `mse.cxx` — Loss Functions

```
MSE per output i:
    mse_i = (1/T) * sum_t[(predicted[i][t] - expected[i][t])^2]
    total_mse = sum_i[mse_i]

Backprop delta (injected into output node error_values):
    delta[i][t] = (predicted[i][t] - expected[i][t]) * (2 / T)
```

MAE uses absolute values and sign-based derivatives. Both are computed by the `RNN` class methods `calculate_error_mse()` / `calculate_error_mae()` after each forward pass.
