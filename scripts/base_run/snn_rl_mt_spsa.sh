#!/bin/sh
# Experimental SNN-RL run with SPSA weight local search; not part of the Week 7 CartPole acceptance path.

cd build

exp_name="../test_output/snn_rl_cartpole_spsa"
mkdir -p "$exp_name"

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
--output_directory "$exp_name" \
--possible_node_types lif \
--std_message_level INFO \
--file_message_level INFO
