#!/bin/sh
# Official Week 7 CartPole SNN-RL run.

set -eu

repo_root="$(CDPATH= cd "$(dirname "$0")/../.." && pwd)"
build_dir="$repo_root/build_rl"
output_dir="$repo_root/test_output/cartpole_week7_lif"

if [ ! -x "$build_dir/multithreaded/snn_rl_mt" ]; then
    echo "Missing $build_dir/multithreaded/snn_rl_mt" >&2
    echo "Build first with: cmake -S . -B build_rl -DEXACT_ENABLE_ASAN=OFF && cmake --build build_rl --target snn_rl_mt" >&2
    exit 1
fi

rm -rf "$output_dir"
mkdir -p "$output_dir"

cd "$build_dir"

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
    --output_directory "$output_dir" \
    --possible_node_types lif \
    --std_message_level INFO \
    --file_message_level INFO
