# Common Utilities — Inner Workings

## `common/arguments.hxx` — Argument Parsing

Template-based parser for `--flag value` style CLI arguments.

```cpp
argument_exists(args, "--flag")                             // bool: flag present?
get_argument(args, "--flag", required, result)              // T: single value after flag
get_argument_vector(args, "--flag", required, results)      // vector<T>: values until next --
get_arguments(args, "--flag", required, result1, result2)   // exactly two values
```

- Values parsed until the next `--` prefix is seen.
- Required arguments call `exit()` with an error message if missing.
- Uses `stringstream` for type conversion; string specializations bypass it.

---

## `common/log.hxx` — Thread-Safe Logger

All members are static. Each thread must call `Log::set_id(name)` before logging and `Log::release_id(name)` when done — this opens/closes per-thread log files.

### Log levels
```
NONE=0  FATAL=1  ERROR=2  WARNING=3  INFO=4  DEBUG=5  TRACE=6  ALL=7
```

### Key static fields
| Field | Default | Purpose |
|-------|---------|---------|
| `std_message_level` | INFO | Controls console output verbosity |
| `file_message_level` | INFO | Controls file output verbosity |
| `output_directory` | `"./logs"` | Where per-thread `.log` files are written |
| `process_rank` | -1 | MPI rank (set via `Log::set_rank()`) |
| `restricted_rank` | -1 | If ≥ 0, only that rank prints |

### Initialization
```cpp
Log::initialize(arguments);   // reads --std_message_level, --file_message_level, --output_directory
Log::set_rank(mpi_rank);      // MPI processes
Log::set_id("thread_name");   // must call per thread before logging
```

### Logging calls
```cpp
Log::info("processed %d genomes", count);
Log::debug("fitness = %f", fitness);
Log::fatal("out of memory");   // also logs to stderr
```

### Thread safety
- `shared_mutex log_ids_mutex`: write-locked during `set_id`/`release_id`, read-locked otherwise
- Per-file `mutex` on each `LogFile` prevents interleaved writes within the same thread's file

---

## `common/random.hxx` — Random Number Generation

Uses `minstd_rand0` (C++11 minimal standard PRNG) throughout the codebase — not thread-safe, each thread/EXAMM instance holds its own generator.

```cpp
float random_0_1(minstd_rand0& gen)                     // Uniform [0, 1)
fisher_yates_shuffle(minstd_rand0& gen, vector<int>& v) // in-place shuffle
```

### `NormalDistribution`
Box-Muller method; generates pairs and caches the second value:
```cpp
NormalDistribution nd;
float x = nd.random(gen, mu, sigma);  // samples N(mu, sigma)
```
Supports `<<`/`>>` serialization and `==`/`!=` comparison (used when serializing genome RNG state).

---

## `common/files.hxx` — File Utilities

```cpp
string get_file_as_string(string path)   // read entire file; throws runtime_error on failure
int    mkpath(const char* path, mode_t)  // mkdir -p equivalent; returns 0 on success
```

`get_file_as_string` strips `\r` characters for Windows compatibility.

---

## `common/exp.hxx` — Deterministic Math

Avoids floating-point inconsistencies across compilers/platforms:

```cpp
float exact_exp(float z)    // e^z via 50-term Taylor series
float exact_sqrt(float s)   // sqrt via Newton's method (7 Heron iterations)
```

Used inside node activation functions for reproducible gradient tests.

---

## `common/process_arguments.hxx` — Domain Argument Processing

High-level factory functions that bridge CLI arguments → EXAMM objects:

```cpp
EXAMM* generate_examm_from_arguments(arguments, time_series_sets, weight_rules, seed_genome)
SpeciationStrategy* generate_speciation_strategy_from_arguments(...)
IslandSpeciationStrategy* generate_island_speciation_strategy_from_arguments(...)
NeatSpeciationStrategy* generate_neat_speciation_strategy_from_arguments(...)
void get_train_validation_data(arguments, time_series_sets, training_inputs, training_outputs, ...)
void slice_input_data(data, sequence_length, sliced_data)
```

Key arguments consumed:
- `--island_size`, `--number_islands`, `--max_genomes`, `--max_wallclock_seconds`
- `--extinction_event_generation_number`, `--islands_to_exterminate`, `--repopulation_method`
- `--num_mutations`, `--mutation_rate`, `--intra_island_crossover_rate`
- `--growth_phase_genomes`, `--reduction_phase_genomes`
- `--train_sequence_length`, `--validation_sequence_length`
- `--speciation_method` (`"island"` or `"neat"`)
