# Using the SNN Implementations

EXACT currently has two practical SNN usage paths:

1. `examm_mt` with `--possible_node_types lif`: LIF neurons inside the normal time-series EXAMM loop, trained with BPTT and a surrogate gradient.
2. `snn_rl_mt`: LIF SNN policies evaluated by RL reward, with optional derivative-free local weight search (`none`, `perturb`, or `spsa`).

There is also a Python `snn_examm/` package that mirrors the RL/evolution design for experimentation and tests. Prefer the C++ executables for runs that should match the main EXACT codebase.

## 1. LIF Time-Series EXAMM with Surrogate Gradients

Use this path when the task is still a supervised time-series forecasting problem, but hidden nodes should be spiking LIF neurons.

Build normally:

```bash
mkdir -p build
cd build
cmake ..
make
```

Run the included coal SNN example:

```bash
sh scripts/base_run/coal_snn_mt.sh
```

Direct command from `build/`:

```bash
INPUT_PARAMETERS="Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int"
OUTPUT_PARAMETERS="Main_Flm_Int"

./multithreaded/examm_mt --number_threads 8 \
  --training_filenames ../datasets/2018_coal/burner_[0-9].csv \
  --validation_filenames ../datasets/2018_coal/burner_1[0-1].csv \
  --time_offset 1 \
  --input_parameter_names $INPUT_PARAMETERS \
  --output_parameter_names $OUTPUT_PARAMETERS \
  --number_islands 10 \
  --island_size 10 \
  --max_genomes 2000 \
  --bp_iterations 5 \
  --num_mutations 2 \
  --output_directory ../test_output/coal_snn_mt \
  --possible_node_types lif \
  --std_message_level INFO \
  --file_message_level INFO
```

What changes compared with vanilla EXAMM:

- `--possible_node_types lif` restricts hidden-node mutations to `LIF_Node`.
- Each LIF node has three evolvable parameters: `v_thresh`, `beta`, and `bias`.
- Forward pass uses hard spiking: output is `1.0` when membrane potential reaches threshold, otherwise `0.0`.
- Backward pass uses a fast-sigmoid surrogate derivative around the threshold, so the existing BPTT and optimizer code can train LIF weights.

You can also mix SNN and standard RNN cells:

```bash
--possible_node_types lif simple UGRNN MGU GRU delta LSTM
```

That lets EXAMM decide whether a child topology should add LIF nodes, conventional memory cells, or both.

### Surrogate-Gradient Verification

The LIF-specific tests live in `rnn_tests/`:

```bash
cd build
./rnn_tests/test_lif_node
./rnn_tests/test_lif_gradients
```

`test_lif_gradients` compares the implemented surrogate-gradient path against the gradient-test harness. Near-threshold behavior is inherently approximate because the forward spike is discontinuous while the backward derivative is smooth.

## 2. SNN RL Evaluation without BPTT

Use `snn_rl_mt` when the genome is a spiking policy evaluated inside an RL environment rather than a supervised time-series predictor.

This executable still uses EXACT's island evolution and Lamarckian weight inheritance, but evaluation is external reward:

```text
genome -> run RL episodes -> average reward -> fitness = -average_reward
```

The negative sign is used because core EXAMM treats lower fitness as better.

For Week 7, the supported reproducible SNN-RL prototype is CartPole. Pendulum and local-search modes are available for experimentation, but the official Week 7 evidence should come from `scripts/base_run/snn_rl_cartpole_mt.sh` and its `rl_fitness_log.csv`.

For RL experiments on macOS, use a non-ASan build. The rl-tools path can hang during startup with the repository's default AddressSanitizer flags:

```bash
cmake -S . -B build_rl -DEXACT_ENABLE_ASAN=OFF
cmake --build build_rl --target snn_rl_mt test_rl_tools_environments rl_tools_pendulum_sac_baseline
```

Run the official CartPole SNN-RL script:

```bash
sh scripts/base_run/snn_rl_cartpole_mt.sh
```

Direct command from `build_rl/`:

```bash
./multithreaded/snn_rl_mt --number_threads 1 \
  --rl_environment cartpole \
  --rl_episodes 3 \
  --rl_t_sim 5 \
  --rl_seed 1337 \
  --rl_seed_hidden_nodes 4 \
  --rl_local_search none \
  --number_islands 5 \
  --island_size 5 \
  --max_genomes 100 \
  --num_mutations 2 \
  --output_directory ../test_output/cartpole_week7_lif \
  --possible_node_types lif \
  --std_message_level INFO \
  --file_message_level INFO
```

Supported environments:

| Environment flag | Observations | Actions | Default decoder |
| --- | --- | --- | --- |
| `cartpole` / `CartPole-v1` | `x x_dot theta theta_dot` | `action_0 action_1` | `discrete_argmax` |
| `pendulum` / `Pendulum-v1` | `cos_theta sin_theta theta_dot` | `torque` | `continuous_tanh` |

Useful RL flags:

- `--rl_episodes`: number of episodes averaged per genome.
- `--rl_t_sim`: SNN timesteps per environment action. Higher values give spikes more time to accumulate before decoding an action.
- `--rl_max_steps`: optional episode cap. Defaults to the environment's cap.
- `--rl_seed`: base seed; episode seeds are derived from it.
- `--rl_seed_hidden_nodes`: number of hidden LIF nodes in the seed genome.
- `--rl_action_decoder`: `discrete_argmax` or `continuous_tanh`; default depends on environment.
- `--rl_observation_clip`: clips observations before current injection.
- `--rl_write_trace`: writes `best_episode_trace.csv` by default.
- `--rl_write_neuron_trace`: writes `best_neuron_trace.csv` by default for LIF current/membrane/spike analysis.

## 3. Experimental Derivative-Free Local Search: Perturb and SPSA

The RL path does not use BPTT or the LIF surrogate derivative. Instead, it can optionally improve a generated genome's weights before insertion with local reward search.

Available modes:

- `--rl_local_search none`: default. Evaluate the generated genome once and insert it.
- `--rl_local_search perturb`: sample random Gaussian weight perturbations and keep improvements.
- `--rl_local_search spsa`: use paired `theta + c * delta` and `theta - c * delta` probes to estimate a reward-improving direction.

Run the included SPSA example:

```bash
sh scripts/base_run/snn_rl_mt_spsa.sh
```

Direct command from `build_rl/`:

```bash
./multithreaded/snn_rl_mt --number_threads 4 \
  --rl_environment cartpole \
  --rl_episodes 3 \
  --rl_t_sim 5 \
  --rl_seed_hidden_nodes 4 \
  --rl_local_search spsa \
  --rl_local_search_iterations 3 \
  --rl_local_search_step 0.02 \
  --rl_local_search_perturbation 0.05 \
  --number_islands 5 \
  --island_size 5 \
  --max_genomes 100 \
  --num_mutations 2 \
  --output_directory ../test_output/snn_rl_cartpole_spsa \
  --possible_node_types lif \
  --std_message_level INFO \
  --file_message_level INFO
```

Local-search tuning:

- `--rl_local_search_iterations`: update attempts per generated genome.
- `--rl_local_search_step`: SPSA update step size.
- `--rl_local_search_perturbation`: random-search standard deviation, or SPSA plus/minus probe size.
- `--rl_local_search_seed`: seed for perturbation directions. Defaults to `--rl_seed`.

Start with small values. Local search multiplies environment evaluations per genome, so it can quickly dominate runtime.

## 4. RL Outputs

`snn_rl_mt` writes:

- `rl_fitness_log.csv`: official RL log for reward, translated fitness, environment, decoder, genome size, LIF count, recurrent count, evaluation time, and local-search statistics.
- `fitness_log.csv`: compatibility log emitted by EXAMM internals. It is useful for confirming insertion counts, but it should not be used as the RL analysis source.
- `best_episode_trace.csv`: one rollout of the best genome with observations, raw outputs, decoded actions, reward, and termination flags.
- `best_neuron_trace.csv`: one rollout of the best genome with per-LIF-neuron `input_current`, `membrane_potential`, `spike_output`, and `output_value` for each environment step and SNN substep.
- `completed`: empty sentinel file after clean completion.
- Standard EXACT genome artifacts, depending on `--save_genome_option`.

For RL logs, higher `avg_reward` is better, while lower `fitness` is better because `fitness = -avg_reward`.

## 5. Python SNN-EXAMM Reference Path

The `snn_examm/` package is a Python implementation used for experiments and tests. It uses Gym/Gymnasium environments, direct current injection for observations, spike-count/action decoding for discrete actions, and reward maximization.

Example:

```bash
python3 -m snn_examm \
  --env_name CartPole-v1 \
  --n_eval_episodes 5 \
  --t_sim 15 \
  --number_islands 10 \
  --island_size 10 \
  --max_genomes 2000 \
  --number_threads 4 \
  --output_directory snn_output
```

This path writes `evolution_stats.json`. It is useful for quick algorithm work, but the C++ `snn_rl_mt` path is the one integrated with the main EXACT executable structure and output conventions.

## 6. Which Path Should I Use?

Use vanilla EXAMM when the goal is standard time-series forecasting with the original memory-cell set.

Use `examm_mt --possible_node_types lif` when the goal is time-series forecasting with spiking hidden nodes and surrogate-gradient training.

Use `snn_rl_mt --rl_local_search none` when the goal is pure evolutionary RL evaluation of LIF policies.

Use `snn_rl_mt --rl_local_search spsa` when the goal is RL evaluation plus derivative-free weight improvement before a genome enters the population.
