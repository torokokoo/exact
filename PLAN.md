# Plan: Add SNN (Spiking Neural Network) Support to EXACT C++

## Context

EXACT currently evolves RNN architectures (LSTM, GRU, MGU, UGRNN, Delta-RNN, etc.) for time series forecasting using BPTT (backpropagation through time). The goal is to add Spiking Neural Network (SNN) support — specifically LIF (Leaky Integrate-and-Fire) neurons — directly into the C++ codebase, reusing the existing genome representation, island evolution, Lamarckian weight initialization, mutation/crossover, and optimizer infrastructure.

**Key insight:** Contrary to the common belief that "backprop is incompatible with SNN," modern SNN training uses **surrogate gradient descent** — the forward pass uses the true Heaviside spike function, while the backward pass substitutes a smooth differentiable surrogate derivative. This makes LIF nodes fully compatible with EXACT's existing BPTT pipeline, optimizers (Adam, etc.), and weight management. No new training algorithm is needed — just a new node type.

**What stays the same:** RNN_Genome class, mutation/crossover operators, island evolution (EXAMM), Lamarckian weight init (mu/sigma), WeightRules, optimizers, serialization format, multithreaded/MPI executables, time series data loading.

**What's new:** LIF_Node class (3 evolvable parameters), surrogate gradient in backward pass, type registration, gradient test.

A Python prototype already exists at `snn_examm/` (LIF nodes, synapses, evolution engine for RL). This C++ implementation ports the LIF neuron model while integrating with EXACT's BPTT training rather than RL-only evaluation.

---

## Checkpoint 1: LIF Node C++ Classes

**Goal:** Implement the LIF spiking neuron as a new `RNN_Node_Interface` subclass.

### Files to Create

#### `rnn/lif_node.hxx`
- Class `LIF_Node : public RNN_Node_Interface`
- Follow the UGRNN_Node pattern (`rnn/ugrnn_node.hxx`)
- Private members:
  - **Evolvable parameters (3 weights):** `double v_thresh, beta, bias`
  - **Per-timestep state:** `vector<double> membrane_potential, spike_output`
  - **Per-timestep gradient accumulators:** `vector<double> d_v_thresh, d_beta, d_bias`
  - **Temporal gradient propagation:** `vector<double> d_membrane` (like UGRNN's `d_h_prev`)
- `#define NUMBER_LIF_WEIGHTS 3`
- All virtual method signatures from `RNN_Node_Interface`

#### `rnn/lif_node.cxx`
- **Constructor:** Set `node_type = LIF_NODE`, call base constructor
- **Forward pass (`input_fired`):**
  ```
  When inputs_fired[time] == total_inputs:
    x = input_values[time]
    v_prev = (time > 0 && spike_output[time-1] <= 0.5) ? membrane_potential[time-1] : 0.0
    v = beta * v_prev + x + bias
    membrane_potential[time] = v
    spike_output[time] = (v >= v_thresh) ? 1.0 : 0.0
    output_values[time] = spike_output[time]
  ```
- **Backward pass (`try_update_deltas`):**
  ```
  When outputs_fired[time] == total_outputs:
    Surrogate gradient: sg = 1.0 / (1.0 + 25.0 * |v - v_thresh|)^2
    d_h = error_values[time] + (time < series_length-1 ? d_membrane[time+1] : 0)
    d_v = d_h * sg
    d_bias[time] = d_v
    d_beta[time] = d_v * v_prev
    d_v_thresh[time] = -d_h * sg
    reset_mask = (time > 0 && spike_output[time-1] > 0.5) ? 0.0 : 1.0
    d_membrane[time] = d_v * beta * reset_mask
    d_input[time] = d_v
  ```
- **`error_fired` / `output_fired`:** Follow UGRNN pattern exactly (accumulate error, call `try_update_deltas`)
- **Weight management:** `get_number_weights()` returns 3; `get/set_weights` serialize v_thresh, beta, bias with `bound()`
- **`get_gradients`:** Sum d_v_thresh[i], d_beta[i], d_bias[i] over all timesteps
- **`reset`:** `.assign(series_length, 0.0)` for all vectors (follow UGRNN pattern)
- **Initialization:** Follow UGRNN pattern exactly — all 4 methods (lamarckian, xavier, kaiming, uniform_random) apply to all 3 parameters uniformly. Do NOT constrain parameter ranges — let evolution/training find good values.
- **`copy`:** Deep copy all member fields (follow UGRNN pattern)
- **`write_to_stream`:** Call `RNN_Node_Interface::write_to_stream(out)`

### Reference Files
- `rnn/ugrnn_node.hxx` / `rnn/ugrnn_node.cxx` — closest pattern (gated cell, 6 weights, temporal state)
- `snn_examm/genome/lif_node.py` — Python LIF dynamics reference

### Verification
- Files compile in isolation (no undefined symbols from interface)
- All pure virtual methods implemented

---

## Checkpoint 2: Build System & Type Registration

**Goal:** Register LIF_NODE in the type system, factory function, deserialization, and CMake.

### Files to Modify

#### `rnn/rnn_node_interface.hxx` (line ~61)
- Add: `#define LIF_NODE 28` (after `INPUT_NODE_GP 27`)

#### `rnn/rnn_node_interface.cxx` (lines 15-49)
- Add `"lif"` to `NODE_TYPES[]` array at index 28
- Add `{"lif", LIF_NODE}` to `string_to_node_type` map

#### `rnn/generate_nn.hxx` (line ~39)
- Add: `#include "rnn/lif_node.hxx"`
- Add: `#define create_lif(...) create_memory_cell_nn<LIF_Node>(__VA_ARGS__)`

#### `rnn/generate_nn.cxx` (line ~75, in `create_hidden_node` switch)
- Add: `case LIF_NODE: return new LIF_Node(++innovation_counter, HIDDEN_LAYER, depth);`

#### `rnn/rnn_genome.cxx`
- **`read_node_from_stream`** (~line 3412, before the `else` fatal error): Add `else if (node_type == LIF_NODE) { node = new LIF_Node(innovation_number, layer_type, depth); }`
- **`#include`** at top: Add `#include "rnn/lif_node.hxx"`

#### `rnn/CMakeLists.txt` (line 1)
- Add `lif_node.cxx` to `add_library(examm_nn ...)` source list

### Verification
- Full project builds with `cmake .. && make`
- `--possible_node_types lif` is parseable (string_to_node_type resolves it)
- `create_lif(inputs, 1, 1, outputs, 1, weight_rules)` creates a valid genome

---

## Checkpoint 3: Gradient Test

**Goal:** Verify the surrogate gradient computation is mathematically correct.

### Files to Create

#### `rnn_tests/test_lif_gradients.cxx`
- Follow `rnn_tests/test_ugrnn_gradients.cxx` pattern exactly
- Include `rnn/lif_node.hxx`
- Test configurations: 1x1, 1x1x1, 1x2x1, 2x2x2, 2x2x2, 3x3x3, 3x4x3
- For each config, test with `max_recurrent_depth` from 1 to 5
- Use `create_lif(...)` to generate test genomes

### Files to Modify

#### `rnn_tests/CMakeLists.txt`
- Add:
  ```
  add_executable(test_lif_gradients test_lif_gradients.cxx gradient_test.cxx)
  target_link_libraries(test_lif_gradients examm_strategy exact_common exact_time_series exact_weights examm_nn ${MYSQL_LIBRARIES} pthread)
  ```

### Important Note on Gradient Testing
The surrogate gradient is an **approximation** — the analytic gradient (BPTT with surrogate) and empirical gradient (finite difference with true Heaviside forward pass) will inherently disagree near the spike threshold. This is expected and correct behavior for surrogate gradient methods.

**Strategy:** The gradient test validates that the gradient computation is internally consistent. Near-threshold disagreements are documented rather than treated as failures. The gradient test still catches real bugs (wrong signs, missing terms, off-by-one in temporal indexing).

### Verification
- `./rnn_tests/test_lif_gradients` runs without crashes
- Gradients agree in regions far from the spike threshold
- Near-threshold disagreements are within expected surrogate approximation bounds

---

## Checkpoint 4: EXAMM Integration

**Goal:** Ensure LIF nodes work correctly within EXAMM's evolutionary loop.

### What Already Works (no changes needed)
- **`examm/examm.cxx` mutation:** `set_possible_node_types()` already converts strings via `node_type_from_string()` — adding `"lif"` to the map (Checkpoint 2) makes `--possible_node_types lif` work automatically
- **Mutation operators:** `add_node`, `split_edge`, `split_node` all call `create_hidden_node(node_type, ...)` which has the `LIF_NODE` case from Checkpoint 2
- **Crossover:** Uses `node->copy()` which is implemented in Checkpoint 1
- **Lamarckian init:** `get_mu_sigma()` computes from flat weight vector; `initialize_lamarckian()` applies N(mu, sigma) — both work generically with LIF's 3 parameters
- **Serialization:** `write_to_stream` / `read_node_from_stream` handled in Checkpoints 1-2

### What to Verify
- `--possible_node_types lif` produces genomes with LIF hidden nodes
- Mutations (add_node, split_edge) correctly create LIF nodes
- Crossover between two LIF genomes produces a valid child
- Mixed mode `--possible_node_types lif,lstm` works (LIF and LSTM coexist)
- Lamarckian weight inheritance: parent's mu/sigma from 3 LIF params initializes child LIF nodes correctly
- Genome serialization roundtrip (write to .bin, read back) preserves LIF node parameters

---

## Checkpoint 5: End-to-End Run

**Goal:** Run SNN-EXAMM on a real time series dataset using the existing `examm_mt` executable.

### No New Executable Needed
The existing `multithreaded/examm_mt` already supports arbitrary node types via `--possible_node_types`. After Checkpoints 1-4:
```bash
./build/multithreaded/examm_mt --possible_node_types lif \
    --training_filenames ../datasets/2018_coal/burner_0.csv \
    --validation_filenames ../datasets/2018_coal/burner_1.csv \
    --time_offset 1 \
    --input_parameter_names "Main_Coverage,Main_Damper_Position" \
    --output_parameter_names "Main_Flame_Intensity" \
    --number_islands 10 --island_size 10 --max_genomes 500 \
    --number_threads 4 --bp_iterations 5 \
    --output_directory ../test_output/snn_run
```

### Verification
- Full build succeeds with no warnings on LIF files
- Short evolution run (500 genomes) completes without crashes
- Fitness improves over generations (validation MSE decreases)
- Output files generated: `fitness_log.csv`, `global_best_genome_*.bin`
- Best genome can be loaded back and evaluated

---

## Checkpoint 6: Run Script & Documentation

**Goal:** Create a convenience run script and update documentation.

### Files to Create

#### `scripts/base_run/coal_snn_mt.sh`
- Shell script with sensible SNN defaults for the coal dataset
- Documents all relevant CLI arguments
- Follows the pattern of existing `scripts/base_run/coal_mt.sh`

### Verification
- Run script executes successfully
- Mixed node type run (`--possible_node_types lif,lstm`) works
- All gradient tests pass: `./build/rnn_tests/test_lif_gradients`

---

## Summary: All File Changes

| File | Action | Checkpoint |
|------|--------|-----------|
| `rnn/lif_node.hxx` | **Create** | 1 |
| `rnn/lif_node.cxx` | **Create** | 1 |
| `rnn/rnn_node_interface.hxx` | Modify (add `#define LIF_NODE 28`) | 2 |
| `rnn/rnn_node_interface.cxx` | Modify (add "lif" to maps) | 2 |
| `rnn/generate_nn.hxx` | Modify (add include + create_lif macro) | 2 |
| `rnn/generate_nn.cxx` | Modify (add case LIF_NODE) | 2 |
| `rnn/rnn_genome.cxx` | Modify (add LIF to read_node_from_stream + include) | 2 |
| `rnn/CMakeLists.txt` | Modify (add lif_node.cxx) | 2 |
| `rnn_tests/test_lif_gradients.cxx` | **Create** | 3 |
| `rnn_tests/CMakeLists.txt` | Modify (add test executable) | 3 |
| `scripts/base_run/coal_snn_mt.sh` | **Create** | 6 |

**New files:** 4 | **Modified files:** 7 | **Total:** 11
