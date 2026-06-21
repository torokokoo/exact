# Training Vanilla EXACT / EXAMM

This runbook shows how to train the repository's default EXAMM time-series setup: no SNN-specific flags, no RL evaluator, and the standard RNN/memory-cell node set used by `scripts/base_run/coal_mt.sh`.

## 1. Build

From the repository root:

```bash
mkdir -p build
cd build
cmake ..
make
```

The default CMake build enables AddressSanitizer through `EXACT_ENABLE_ASAN=ON`. For long cluster or benchmarking runs, use a non-ASan build directory:

```bash
cmake -S . -B build_release -DEXACT_ENABLE_ASAN=OFF
cmake --build build_release
```

## 2. Run the Included Vanilla Example

The simplest way to run the project as it ships is:

```bash
sh scripts/base_run/coal_mt.sh
```

That script changes into `build/` and runs:

```bash
./multithreaded/examm_mt --number_threads 8 \
  --training_filenames ../datasets/2018_coal/burner_[0-9].csv \
  --validation_filenames ../datasets/2018_coal/burner_1[0-1].csv \
  --time_offset 1 \
  --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int \
  --output_parameter_names Main_Flm_Int \
  --number_islands 10 \
  --island_size 10 \
  --max_genomes 2000 \
  --bp_iterations 5 \
  --num_mutations 2 \
  --output_directory ../test_output/coal_mt \
  --possible_node_types simple UGRNN MGU GRU delta LSTM \
  --std_message_level INFO \
  --file_message_level INFO
```

This is the vanilla EXAMM path. It evolves recurrent neural-network topologies, trains each generated genome with stochastic BPTT, evaluates validation MSE, and keeps the best genomes.

## 3. Short Smoke Run

For a fast test after building, reduce the search size:

```bash
cd build

INPUT_PARAMETERS="Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int"
OUTPUT_PARAMETERS="Main_Flm_Int"

./multithreaded/examm_mt --number_threads 2 \
  --training_filenames ../datasets/2018_coal/burner_[0-2].csv \
  --validation_filenames ../datasets/2018_coal/burner_10.csv \
  --time_offset 1 \
  --input_parameter_names $INPUT_PARAMETERS \
  --output_parameter_names $OUTPUT_PARAMETERS \
  --number_islands 3 \
  --island_size 3 \
  --max_genomes 30 \
  --bp_iterations 2 \
  --num_mutations 1 \
  --output_directory ../test_output/coal_mt_smoke \
  --possible_node_types simple UGRNN MGU GRU delta LSTM \
  --std_message_level INFO \
  --file_message_level INFO
```

Use this only to verify the pipeline. It is too small to be a meaningful experiment.

## 4. Important Flags

Data:

- `--training_filenames`: CSV files used for weight training.
- `--validation_filenames`: CSV files used only for validation fitness.
- `--input_parameter_names`: input CSV columns.
- `--output_parameter_names`: target CSV columns.
- `--time_offset`: forecast horizon in rows. `1` means row `t` predicts row `t + 1`.
- `--normalize min_max` or `--normalize avg_std_dev`: optional normalization. The coal example is already normalized, so the script does not pass this flag.

Evolution:

- `--number_threads`: worker threads. Each worker trains generated genomes independently.
- `--number_islands`: number of island populations.
- `--island_size`: maximum genomes per island.
- `--max_genomes`: stop after this many evaluated/generated genomes.
- `--max_wallclock_seconds`: optional wall-clock stop condition.
- `--num_mutations`: mutation operations applied when generating a mutated child.
- `--possible_node_types`: node types allowed during topology evolution. Vanilla EXAMM normally uses `simple UGRNN MGU GRU delta LSTM`.

Training:

- `--bp_iterations`: BPTT epochs per generated genome.
- `--weight_update`: optimizer. Default is `adam`; alternatives include `vanilla`, `momentum`, `nesterov`, `adagrad`, `rmsprop`, `adam`, and `adam-bias`.
- `--learning_rate`: optimizer learning rate. Default is `0.001`.
- `--weight_initialize`, `--weight_inheritance`, `--mutated_component_weight`: defaults are Xavier initialization and Lamarckian inheritance for parent/new-component weights.

## 5. Outputs

The output directory contains:

- `fitness_log.csv`: inserted-genome fitness and population statistics.
- `genome_stats_log.csv`: initial/final fitness, BP epochs, and BP time per trained genome.
- `rnn_genome_<id>.bin/.txt/.gv`: each new global best genome saved during the run when `--save_genome_option all_best_genomes` is active.
- `global_best_genome_<id>.bin/.txt/.gv`: final global best saved when the run terminates.
- `op_log.csv`: only when `--generate_op_log` is enabled.
- `size_log/`: only when `--genome_size_log 1` is enabled.

Render a `.gv` genome with Graphviz:

```bash
dot -Tpdf test_output/coal_mt/global_best_genome_<id>.gv -o test_output/coal_mt/global_best_genome_<id>.pdf
```

## 6. Evaluate a Trained Genome

After training, use `rnn_examples/evaluate_rnn` to run inference with a saved `.bin` genome:

```bash
cd build

./rnn_examples/evaluate_rnn \
  --genome_file ../test_output/coal_mt/global_best_genome_<id>.bin \
  --testing_filenames ../datasets/2018_coal/burner_11.csv \
  --time_offset 1 \
  --output_directory ../test_output/coal_mt_eval
```

The genome stores the training normalization parameters, so evaluation applies the same normalization policy used during training.
