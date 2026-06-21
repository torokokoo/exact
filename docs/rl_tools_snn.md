# rl-tools SNN-EXAMM Integration

`multithreaded/snn_rl_mt` evolves LIF-based SNN policies with EXACT's island evolution while using rl-tools-style native RL environments for evaluation.

For Week 7, the supported reproducible prototype is CartPole through `scripts/base_run/snn_rl_cartpole_mt.sh`. Pendulum and SPSA are kept as experimental extensions and are not part of the Week 7 acceptance criteria.

## Supported Environments

| `--rl_environment` | Observations | Actions | Decoder |
|--------------------|--------------|---------|---------|
| `cartpole` / `CartPole-v1` | `x x_dot theta theta_dot` | `action_0 action_1` | `discrete_argmax` |
| `pendulum` / `Pendulum-v1` | `cos_theta sin_theta theta_dot` | `torque` | `continuous_tanh` |

CartPole keeps the current EXACT physics for continuity. Pendulum uses rl-tools' native `Pendulum` implementation.

## Week 7 CartPole Command

The rl-tools dependency is tracked as the `external/rl_tools` Git submodule. Fresh checkouts should initialize it before configuring CMake:

```bash
git submodule update --init --recursive external/rl_tools
```

Configure the RL build with AddressSanitizer disabled for macOS smoke runs:

```bash
cmake -S . -B build_rl -DEXACT_ENABLE_ASAN=OFF
cmake --build build_rl --target snn_rl_mt test_rl_tools_environments
```

Run the official reproducible CartPole script:

```bash
sh scripts/base_run/snn_rl_cartpole_mt.sh
```

The script uses one thread, seed `1337`, no local search, LIF nodes only, and writes to `test_output/cartpole_week7_lif`.

Equivalent direct command from `build_rl/`:

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
  --possible_node_types lif
```

## Experimental Pendulum Command

```bash
./build/multithreaded/snn_rl_mt --number_threads 4 \
  --rl_environment pendulum \
  --rl_episodes 3 \
  --rl_t_sim 5 \
  --rl_seed_hidden_nodes 4 \
  --number_islands 5 \
  --island_size 5 \
  --max_genomes 100 \
  --output_directory test_output/snn_rl_pendulum \
  --possible_node_types lif
```

On macOS, rl-tools binaries may hang during startup when built with this repository's default AddressSanitizer flags. For RL experiments or smoke tests, configure a non-ASan build:

```bash
cmake -S . -B build_rl -DEXACT_ENABLE_ASAN=OFF
cmake --build build_rl --target snn_rl_mt test_rl_tools_environments rl_tools_pendulum_sac_baseline
```

## RL Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--rl_environment` | `cartpole` | Environment to evaluate. |
| `--rl_episodes` | `3` | Episodes per genome evaluation. |
| `--rl_t_sim` | `5` | SNN timesteps per environment action. |
| `--rl_max_steps` | Environment default | Episode cap. |
| `--rl_seed` | `1337` | Base seed; episode seed is `seed + episode`. |
| `--rl_seed_hidden_nodes` | `4` | LIF nodes in the initial seed genome. |
| `--rl_action_decoder` | Environment default | `discrete_argmax` or `continuous_tanh`. |
| `--rl_observation_clip` | `10.0` | Clips observations before repeated current injection. |
| `--rl_local_search` | `none` | Optional per-genome weight local search: `none`, `perturb`, or `spsa`. |
| `--rl_local_search_iterations` | `0` | Local-search update attempts per generated genome. |
| `--rl_local_search_step` | `0.05` | SPSA update step size. |
| `--rl_local_search_perturbation` | `0.10` | Perturbation size for random search and SPSA plus/minus probes. |
| `--rl_local_search_seed` | `--rl_seed` | Seed for local-search perturbations. |

EXACT stores lower numeric fitness as better, so RL reward is recorded as `fitness = -average_reward`. The original reward is logged separately.

## Experimental Weight Local Search / SPSA

By default, `snn_rl_mt` keeps the original behavior: EXAMM generates a genome, evaluates its RL reward, and inserts it into the island population. Setting `--rl_local_search` adds a per-genome weight improvement phase before insertion while leaving topology mutation/crossover unchanged.

```bash
# Baseline: current behavior
./build/multithreaded/snn_rl_mt ... --rl_local_search none

# EXAMM + SPSA local weight search
./build/multithreaded/snn_rl_mt ... \
  --rl_local_search spsa \
  --rl_local_search_iterations 3 \
  --rl_local_search_step 0.02 \
  --rl_local_search_perturbation 0.05
```

`perturb` samples random Gaussian weight changes and keeps improvements. `spsa` evaluates paired `theta + c * delta` and `theta - c * delta` probes, estimates a reward-improving direction, and accepts the updated weights only when reward improves.

## Outputs

Each run writes:

- `rl_fitness_log.csv`: the official RL log for reward, translated fitness, environment, decoder, genome size, LIF counts, recurrent counts, evaluation time, and local-search statistics.
- `fitness_log.csv`: EXAMM compatibility log. It is still emitted by the core evolution loop, but do not use it as the RL analysis source.
- `best_episode_trace.csv`: one rollout of the best genome with observations, raw policy outputs, decoded actions, reward, and termination.
- Existing EXACT best genome artifacts, depending on `--save_genome_option`.

Generated run directories belong under `test_output/` or another ignored output path. Do not commit `rl_fitness_log.csv`, `fitness_log.csv`, traces, genome snapshots, or thread logs from experiments.

## Baseline

`./build/exact_rl_tools/rl_tools_pendulum_sac_baseline` runs a small rl-tools SAC Pendulum baseline. It is intended as a comparison point for the thesis, not as the SNN training method.
