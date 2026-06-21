# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

EXACT is a C++20 neuroevolution research project implementing two algorithms:
- **EXAMM** (Evolutionary eXploration of Augmenting Memory Models): evolves RNNs for time series forecasting using LSTM, GRU, MGU, UGRNN, Delta-RNN, and simple memory cells
- **EXA-GP** (Evolutionary Exploration of Augmenting Genetic Programs): evolves interpretable genetic programs using GP operations (sum, multiply, sin, cos, tanh, sigmoid, inverse)

## Build

```bash
mkdir build
cd build
cmake ..
make
```

Dependencies (macOS): `cmake`, `open-mpi` (5.0.1), `mysql`, `libtiff`, `libpng`, `clang-format`

The default build flags (`CMakeLists.txt:29`) include `-fsanitize=address` and `-ggdb3`. Switch to the cluster flags (line 27) for HPC runs.

## Running

```bash
# Multithreaded (from repo root)
sh scripts/base_run/coal_mt.sh

# MPI distributed
sh scripts/base_run/coal_mpi.sh

# Direct invocation (from build/)
./multithreaded/examm_mt --number_threads 8 \
  --training_filenames ../datasets/2018_coal/burner_[0-9].csv \
  --validation_filenames ../datasets/2018_coal/burner_1[0-1].csv \
  --time_offset 1 --input_parameter_names <cols> --output_parameter_names <cols> \
  --number_islands 10 --island_size 10 --max_genomes 2000 \
  --bp_iterations 5 --output_directory ../test_output/run1
```

## Gradient Tests

Individual node gradient tests are in `build/rnn_tests/`. Run them directly:

```bash
cd build
./rnn_tests/test_lstm_gradients
./rnn_tests/test_gru_gradients
# etc. — one binary per node type
```

## Code Style

CI enforces clang-format 18 on all source directories (`common`, `examm`, `mpi`, `multithreaded`, `rnn`, `rnn_examples`, `rnn_tests`, `time_series`, `weights`). Run before committing:

```bash
clang-format -i <file>
# or check all:
find . -path ./build -prune -o \( -name "*.cxx" -o -name "*.hxx" -o -name "*.h" \) -print | xargs clang-format --dry-run --Werror
```

## PR Checklist

From `.github/pull_request_template.md`: test on macOS **and** Ubuntu, generate no new warnings, include a run script for any new feature.

## Session Logging Protocol

This thesis project keeps a development ledger in `docs/session_logs/`. When a Codex coding session reaches a completed goal, a major implementation/design turning point, or the end of a long work session, add a milestone entry with:

```bash
python3 scripts/session_log.py add --trigger milestone \
  --title "<short checkpoint title>" \
  --goal "<goal or checkpoint reached>" \
  --change "<notable implementation or documentation change>" \
  --decision "<decision and rationale>" \
  --test "<command or verification result>" \
  --next "<recommended next action>"
```

If the user asks to "log this session so far", create a manual entry with `--trigger manual`. If milestone logging was missed during the day, run `python3 scripts/session_log.py daily --date today` before stopping. The log should capture human-readable reasoning and verification notes; exact code reconstruction should rely on Git history rather than full patch snapshots.

## Architecture

Libraries are built first and linked by executables:

```
exact_common      → logging, RNG, argument parsing, file I/O
exact_time_series → CSV loading, normalization (min_max / avg_std_dev)
examm_nn (rnn/)   → all node types + genome representation
exact_weights     → optimizers: adam, adam-bias, rmsprop, momentum, nesterov, adagrad, vanilla
examm_strategy    → island/NEAT speciation, population management, mutation/crossover ops
```

Executables:
- `multithreaded/examm_mt` — pthreads-based, one main thread manages evolution + N worker threads train genomes
- `mpi/examm_mpi` / `examm_mpi_multi` — MPI-based distributed; rank 0 manages population, other ranks train
- `rnn_examples/evaluate_rnn` — load a `.bin` genome and run inference
- `rnn_tests/test_*_gradients` — finite-difference gradient checks per node type

### Genome Representation (`rnn/`)

A genome holds `RNN_Node` objects (one per `possible_node_types` entry) connected by `RNN_Edge` (feedforward) and `RNN_Recurrent_Edge` (recurrent, with configurable time-skip 1–10). Each node type implements `forward_pass` / `backward_pass`. GP nodes (`*_gp` variants) fix edge weights to 1 and use different initialization.

### Evolution Loop

1. Main thread/process generates a child genome via mutation or crossover from the island population
2. Worker receives genome, runs backpropagation for `bp_iterations` epochs, computes validation MSE
3. Main inserts genome into the appropriate island; extinction events prune low-fitness islands at `--extinction_event_generation_number` intervals
4. Phased evolution (growth/reduction phases) can be enabled with `--growth_phase_genomes` / `--reduction_phase_genomes`

### Output Files

Each run produces in `--output_directory`:
- `fitness_log.csv` — per-genome metrics (time, fitness, island stats, network size)
- `global_best_genome_<id>.{bin,txt,gv}` — serialized best genome; render `.gv` with `dot -T pdf`
- `rnn_genome_<id>.json` — optional (flag `--generate_visualization_json`), used by the Genetic Distance Projection visualization framework
- `completed` — empty sentinel file on clean exit
