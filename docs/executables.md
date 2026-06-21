# Executables — Inner Workings

## Multithreaded (`multithreaded/`)

### `examm_mt` — Main Evolution Entry Point

**Initialization sequence:**
```
1. Parse CLI arguments
2. Log::initialize(arguments)
3. TimeSeriesSets::generate_from_arguments()       ← load + normalize CSVs
4. get_train_validation_data()                     ← export to 3D vectors
5. WeightUpdate::generate_from_arguments()
6. WeightRules::generate_from_arguments()
7. get_seed_genome()                               ← minimal FF genome or loaded .bin
8. generate_examm_from_arguments()                 ← creates EXAMM + islands
9. spawn N threads (--number_threads)
10. join all threads
11. write final results, delete EXAMM
```

**Per-thread loop:**
```cpp
void examm_thread(int id) {
    while (true) {
        examm_mutex.lock();
        RNN_Genome* genome = examm->generate_genome();   // NULL = done
        examm_mutex.unlock();
        if (!genome) break;

        Log::set_id("genome_" + generation_id + "_thread_" + id);
        genome->backpropagate_stochastic(training_inputs, training_outputs,
                                          validation_inputs, validation_outputs,
                                          weight_update);
        Log::release_id(...);

        examm_mutex.lock();
        examm->insert_genome(genome);
        examm_mutex.unlock();
        delete genome;
    }
}
```

The mutex is only held during `generate_genome()` and `insert_genome()`. All backpropagation (the expensive part) is fully parallel.

---

### `examm_mt_single_series` — Leave-One-Out Cross-Validation

Runs `examm_mt` logic inside a double loop:

```
for each time series i (as validation):
    training_indexes = all other series
    test_indexes = [i]
    for repeat k in 0..repeats:
        create EXAMM, run evolution (same thread pool logic)
        evaluate best_genome on training and validation
        write best_genome to: output_directory/output_filename_slice_i_repeat_k.{bin,gv}
        append MSE/MAE to: overall_results.txt
```

---

### `snn_rl_mt` — SNN-EXAMM for RL

Evolves LIF-based SNN policies with island evolution and Lamarckian weight inheritance. Genome evaluation is external RL reward instead of time-series BPTT; reward is stored as `fitness = -average_reward` so existing EXAMM lower-is-better selection remains unchanged.

The Week 7 supported prototype is CartPole. Pendulum and SPSA local search are experimental extensions. Use `rl_fitness_log.csv` as the official RL log; `fitness_log.csv` is still emitted by EXAMM for compatibility but should not be used as the RL analysis source. See [`docs/rl_tools_snn.md`](rl_tools_snn.md) for the rl-tools integration details, command-line options, output files, and the Pendulum SAC baseline.

---

## MPI (`mpi/`)

### `examm_mpi` — Distributed Master-Worker

**MPI rank roles:**
- **Rank 0**: master — owns EXAMM, manages population
- **Ranks 1…N-1**: workers — train genomes

**Message tags:**
```cpp
WORK_REQUEST_TAG  = 1   // worker → master: "I need a genome"
GENOME_LENGTH_TAG = 2   // master/worker → other: "genome is N bytes"
GENOME_TAG        = 3   // master/worker → other: genome data chunks
TERMINATE_TAG     = 4   // master → worker: "search is done, exit"
```

**Master loop:**
```
while workers_alive > 0:
    MPI_Probe(ANY_SOURCE, ANY_TAG)

    if WORK_REQUEST_TAG:
        genome = examm->generate_genome()
        if genome == NULL:
            send TERMINATE_TAG to worker
            workers_alive--
        else:
            pool[genome.generation_id] = genome.copy()    ← track in-flight
            send genome in 32KB chunks to worker

    if GENOME_LENGTH_TAG (worker returning trained genome):
        receive genome in 32KB chunks
        remove from pool
        examm->insert_genome(genome)
```

**Worker loop:**
```
while true:
    send WORK_REQUEST_TAG to rank 0
    MPI_Probe(rank 0, ANY_TAG)

    if TERMINATE_TAG: break

    if GENOME_LENGTH_TAG:
        receive genome in chunks
        genome->backpropagate_stochastic(...)
        send trained genome back to rank 0 in chunks
        delete genome
```

**Genome chunking:** Serialized genomes are sent in **32KB chunks** (`32 * 1024` bytes) to stay within MPI message size limits across all implementations. Sender computes `ceil(total_bytes / chunk_size)` and sends that many `MPI_Send` calls; receiver reassembles.

**Serialization:** Uses `genome->write_to_array(char** array, int32_t& length)` and `new RNN_Genome(char* array, length)`.

---

### `examm_mpi_multi` — K-Fold Cross-Validation with MPI

Extends `examm_mpi` with an outer fold loop, similar to `examm_mt_single_series`:

```
for slice i = 0 to num_series step fold_size:
    test_indexes = [i, i+1, ..., i+fold_size-1]
    training_indexes = all others
    for repeat k in 0..repeats:
        all ranks: run master() or worker() as appropriate
        MPI_Barrier()                        ← synchronize before next repeat
        master writes results to hierarchical directories
```

Output structure: `output_directory/slice_i/repeat_k_best.{bin,gv}` and `runtimes.csv`.

---

## `rnn_examples/`

### `evaluate_rnn`
```
--genome_file, --testing_filenames, --time_offset, --output_directory

1. Load genome from .bin
2. Apply genome's stored normalization to test CSVs
3. Evaluate: MSE = genome->get_mse(best_parameters, inputs, outputs)
             MAE = genome->get_mae(...)
4. genome->write_predictions() → CSV files in output_directory
```

### `train_rnn`
Trains a single hand-specified architecture (no evolution):
```
--rnn_type (lstm|gru|delta|mgu|ugrnn|ff|jordan|elman|dnas)
--num_hidden_layers, --max_recurrent_depth, --bp_iterations

1. create_<rnn_type>(inputs, hidden_layers, ..., weight_rules)
2. genome->backpropagate() or backpropagate_stochastic()
3. Log final MSE/MAE on training and test sets
```
Useful for establishing a non-evolved baseline.

### `finetune_rnn`
Continue training a pre-evolved genome on new data:
```
--genome_file, --finetune_iterations (default 100)

1. Load genome, load new time series, apply genome's normalization
2. Evaluate before fine-tuning → log initial MSE/MAE
3. genome->set_weights(initial_parameters); genome->backpropagate_stochastic()
4. Evaluate after → log improvement
5. Save fine-tuned genome to output_directory/finetuned_genome.bin
```

### Other tools in `rnn_examples/`
| Binary | Purpose |
|--------|---------|
| `evaluate_rnns_multi_offset` | Evaluate same genome at multiple time offsets |
| `rnn_statistics` | Print network statistics (node counts, edge counts, etc.) |
| `analyze_global_bests` | Analyze best genomes across multiple runs |
| `analyze_experimental_setups` | Compare results across experimental configurations |
| `rnn_heatmap` | Generate correlation heatmap of time series |

---

## `rnn_tests/` — Gradient Validation

### How gradient checking works

Each test binary creates one or more `RNN_Genome` instances and calls `gradient_test()`:

```
gradient_test(name, genome, inputs, outputs):
    for 10 regression iterations:
        analytic  = genome->get_analytic_gradient(params, inputs, outputs)
        empirical = genome->get_empirical_gradient(params, inputs, outputs)
                    // finite difference: (MSE(w+ε) - MSE(w-ε)) / (2ε)

        for each weight j:
            if |analytic[j] - empirical[j]| > 1e-9:
                FAIL: log weight index and both values
```

One test binary per node type:

| Binary | Node tested |
|--------|------------|
| `test_feed_forward_gradients` | Simple feed-forward (RNN_Node) |
| `test_elman_gradients` | Elman RNN |
| `test_jordan_gradients` | Jordan RNN |
| `test_lstm_gradients` | LSTM_Node |
| `test_gru_gradients` | GRU_Node |
| `test_mgu_gradients` | MGU_Node |
| `test_ugrnn_gradients` | UGRNN_Node |
| `test_delta_gradients` | Delta_Node |
| `test_sin_gradients` | SIN_Node |
| `test_cos_gradients` | COS_Node |
| `test_tanh_gradients` | TANH_Node |
| `test_sigmoid_gradients` | SIGMOID_Node |
| `test_inverse_gradients` | INVERSE_Node |
| `test_sum_gradients` | SUM_Node |
| `test_multiply_gradients` | MULTIPLY_Node |
| `test_*_gp_gradients` | GP variants of the above |
| `test_dnas_gradients` | DNAS compound node |
| `test_enarc_gradients` | ENARC compound node |
| `test_enas_dag_gradients` | ENAS DAG compound node |
| `test_random_dag_gradients` | Random DAG compound node |
| `test_get_equations` | EXA-GP symbolic equation extraction |
| `test_node_to_binary` | Node serialization round-trip |

**Each test covers multiple topology variants:** 1–3 inputs, 0–2 hidden layers (varying width), 1–3 outputs, and recurrent depths 1–5.

**Note on GP gradient tests:** GP node tests will fail unless the condition that zeros out `d_weight` on edges is disabled, because GP nodes intentionally do not train edge weights.
