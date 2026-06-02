#include <algorithm>
using std::max;

#include <chrono>
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::system_clock;

#include <fstream>
using std::ofstream;

#include <mutex>
using std::lock_guard;
using std::mutex;

#include <string>
using std::string;
using std::to_string;

#include <thread>
using std::thread;

#include <vector>
using std::vector;

#include "cartpole_env.hxx"
#include "common/arguments.hxx"
#include "common/log.hxx"
#include "common/process_arguments.hxx"
#include "examm/examm.hxx"
#include "rnn/generate_nn.hxx"
#include "rnn/lif_node.hxx"
#include "rnn/rnn.hxx"
#include "rnn/rnn_edge.hxx"
#include "rnn/rnn_genome.hxx"
#include "rnn/rnn_node.hxx"
#include "rnn/rnn_node_interface.hxx"
#include "weights/weight_rules.hxx"

mutex examm_mutex;
mutex log_mutex;

vector<string> arguments;

EXAMM* examm;
ofstream rl_log_file;

string output_directory;
string rl_environment = "cartpole";
int32_t number_threads = 1;
int32_t rl_episodes = 3;
int32_t rl_t_sim = 5;
int32_t rl_max_steps = 500;
int32_t rl_seed = 1337;
int32_t evaluated_genomes = 0;
bool rl_write_trace = true;
double global_best_reward = -1.0e300;

struct RLEvaluation {
    double average_reward;
    int32_t total_steps;
    long evaluation_milliseconds;
};

RNN_Genome* create_cartpole_seed_genome(
    const vector<string>& input_parameter_names, const vector<string>& output_parameter_names, WeightRules* weight_rules
) {
    vector<RNN_Node_Interface*> nodes;
    vector<RNN_Edge*> edges;
    vector<RNN_Recurrent_Edge*> recurrent_edges;

    int32_t node_innovation = 0;
    int32_t edge_innovation = 0;

    vector<RNN_Node_Interface*> input_nodes;
    for (int32_t i = 0; i < (int32_t) input_parameter_names.size(); i++) {
        RNN_Node* input_node = new RNN_Node(++node_innovation, INPUT_LAYER, 0.0, SIMPLE_NODE, input_parameter_names[i]);
        nodes.push_back(input_node);
        input_nodes.push_back(input_node);
    }

    vector<RNN_Node_Interface*> lif_nodes;
    for (int32_t i = 0; i < 4; i++) {
        LIF_Node* lif_node = new LIF_Node(++node_innovation, HIDDEN_LAYER, 0.5);
        nodes.push_back(lif_node);
        lif_nodes.push_back(lif_node);
    }

    vector<RNN_Node_Interface*> output_nodes;
    for (int32_t i = 0; i < (int32_t) output_parameter_names.size(); i++) {
        RNN_Node* output_node =
            new RNN_Node(++node_innovation, OUTPUT_LAYER, 1.0, SIMPLE_NODE, output_parameter_names[i]);
        nodes.push_back(output_node);
        output_nodes.push_back(output_node);
    }

    for (auto input_node : input_nodes) {
        for (auto lif_node : lif_nodes) {
            edges.push_back(new RNN_Edge(++edge_innovation, input_node, lif_node));
        }
    }

    for (auto lif_node : lif_nodes) {
        for (auto output_node : output_nodes) {
            edges.push_back(new RNN_Edge(++edge_innovation, lif_node, output_node));
        }
    }

    RNN_Genome* genome = new RNN_Genome(nodes, edges, recurrent_edges, weight_rules);
    genome->set_parameter_names(input_parameter_names, output_parameter_names);
    return genome;
}

int32_t decode_action(RNN* rnn, const vector<double>& observation, int32_t number_outputs) {
    vector<vector<double> > inputs(observation.size(), vector<double>(rl_t_sim, 0.0));
    for (int32_t i = 0; i < (int32_t) observation.size(); i++) {
        for (int32_t t = 0; t < rl_t_sim; t++) {
            inputs[i][t] = observation[i];
        }
    }

    vector<vector<double> > expected_outputs(number_outputs, vector<double>(rl_t_sim, 0.0));
    vector<double> predictions = rnn->get_predictions(inputs, expected_outputs, false, 0.0);
    vector<double> action_scores(number_outputs, 0.0);

    for (int32_t t = 0; t < rl_t_sim; t++) {
        for (int32_t action = 0; action < number_outputs; action++) {
            action_scores[action] += predictions[(t * number_outputs) + action];
        }
    }

    int32_t best_action = 0;
    for (int32_t action = 1; action < number_outputs; action++) {
        if (action_scores[action] > action_scores[best_action]) {
            best_action = action;
        }
    }
    return best_action;
}

RLEvaluation evaluate_cartpole(RNN_Genome* genome) {
    auto start = system_clock::now();
    RNN* rnn = genome->get_rnn();
    int32_t number_outputs = genome->get_number_outputs();

    double total_reward = 0.0;
    int32_t total_steps = 0;

    for (int32_t episode = 0; episode < rl_episodes; episode++) {
        CartPoleEnv env;
        vector<double> observation = env.reset(rl_seed + episode);
        double episode_reward = 0.0;

        for (int32_t step = 0; step < rl_max_steps; step++) {
            int32_t action = decode_action(rnn, observation, number_outputs);
            CartPoleStep result = env.step(action);
            episode_reward += result.reward;
            total_steps++;
            observation = result.observation;

            if (result.terminated) {
                break;
            }
        }

        total_reward += episode_reward;
    }

    delete rnn;

    auto end = system_clock::now();
    long eval_ms = duration_cast<milliseconds>(end - start).count();
    return {total_reward / rl_episodes, total_steps, eval_ms};
}

void write_rl_log_header() {
    rl_log_file.open(output_directory + "/rl_fitness_log.csv");
    rl_log_file << "evaluated_genomes,generation_id,thread_id,env,seed,episodes,avg_reward,best_avg_reward,fitness,"
                   "steps_total,enabled_nodes,enabled_lif_nodes,enabled_edges,enabled_recurrent_edges,evaluation_ms,"
                   "inserted\n";
    rl_log_file.flush();
}

void write_rl_log_row(
    int32_t row, RNN_Genome* genome, int32_t thread_id, const RLEvaluation& evaluation, double fitness, bool inserted
) {
    rl_log_file << row << "," << genome->get_generation_id() << "," << thread_id << "," << rl_environment << ","
                << rl_seed << "," << rl_episodes << "," << evaluation.average_reward << "," << global_best_reward << ","
                << fitness << "," << evaluation.total_steps << "," << genome->get_enabled_node_count() << ","
                << genome->get_enabled_node_count(LIF_NODE) << "," << genome->get_enabled_edge_count() << ","
                << genome->get_enabled_recurrent_edge_count() << "," << evaluation.evaluation_milliseconds << ","
                << (inserted ? 1 : 0) << "\n";
    rl_log_file.flush();
}

void snn_rl_thread(int32_t id) {
    while (true) {
        examm_mutex.lock();
        Log::set_id("main");
        RNN_Genome* genome = examm->generate_genome();
        examm_mutex.unlock();

        if (genome == NULL) {
            break;
        }

        string log_id = "rl_genome_" + to_string(genome->get_generation_id()) + "_thread_" + to_string(id);
        Log::set_id(log_id);
        RLEvaluation evaluation = evaluate_cartpole(genome);
        double fitness = -evaluation.average_reward;
        genome->record_external_evaluation(fitness, evaluation.average_reward, evaluation.evaluation_milliseconds);
        Log::info(
            "generation %d avg_reward: %lf fitness: %lf steps: %d\n", genome->get_generation_id(),
            evaluation.average_reward, fitness, evaluation.total_steps
        );
        Log::release_id(log_id);

        examm_mutex.lock();
        Log::set_id("main");
        bool inserted = examm->insert_genome(genome);
        examm_mutex.unlock();

        lock_guard<mutex> guard(log_mutex);
        evaluated_genomes++;
        global_best_reward = max(global_best_reward, evaluation.average_reward);
        write_rl_log_row(evaluated_genomes, genome, id, evaluation, fitness, inserted);

        delete genome;
    }
}

void write_best_episode_trace(RNN_Genome* genome) {
    if (genome == NULL) {
        return;
    }

    ofstream trace(output_directory + "/best_episode_trace.csv");
    trace << "step,x,x_dot,theta,theta_dot,action,reward,terminated\n";

    RNN* rnn = genome->get_rnn();
    CartPoleEnv env;
    vector<double> observation = env.reset(rl_seed);
    int32_t number_outputs = genome->get_number_outputs();

    for (int32_t step = 0; step < rl_max_steps; step++) {
        int32_t action = decode_action(rnn, observation, number_outputs);
        CartPoleStep result = env.step(action);
        trace << step << "," << observation[0] << "," << observation[1] << "," << observation[2] << ","
              << observation[3] << "," << action << "," << result.reward << "," << (result.terminated ? 1 : 0) << "\n";
        observation = result.observation;
        if (result.terminated) {
            break;
        }
    }

    delete rnn;
}

void append_argument_if_missing(const string& argument, const string& value) {
    if (!argument_exists(arguments, argument)) {
        arguments.push_back(argument);
        arguments.push_back(value);
    }
}

int main(int argc, char** argv) {
    arguments = vector<string>(argv, argv + argc);

    append_argument_if_missing("--speciation_method", "island");
    append_argument_if_missing("--bp_iterations", "0");
    append_argument_if_missing("--possible_node_types", "lif");
    append_argument_if_missing("--max_recurrent_depth", "1");

    Log::initialize(arguments);
    Log::set_id("main");

    get_argument(arguments, "--number_threads", true, number_threads);
    get_argument(arguments, "--output_directory", true, output_directory);
    get_argument(arguments, "--rl_environment", false, rl_environment);
    get_argument(arguments, "--rl_episodes", false, rl_episodes);
    get_argument(arguments, "--rl_t_sim", false, rl_t_sim);
    get_argument(arguments, "--rl_max_steps", false, rl_max_steps);
    get_argument(arguments, "--rl_seed", false, rl_seed);
    get_argument(arguments, "--rl_write_trace", false, rl_write_trace);
    vector<string> possible_node_type_strings;
    get_argument_vector(arguments, "--possible_node_types", false, possible_node_type_strings);

    if (rl_environment != "cartpole" && rl_environment != "CartPole-v1") {
        Log::fatal("Only --rl_environment cartpole is supported in the week 7 executable.\n");
        exit(1);
    }

    vector<string> input_parameter_names{"x", "x_dot", "theta", "theta_dot"};
    vector<string> output_parameter_names{"action_0", "action_1"};

    WeightRules* weight_rules = new WeightRules();
    weight_rules->initialize_from_args(arguments);

    RNN_Genome* seed_genome = create_cartpole_seed_genome(input_parameter_names, output_parameter_names, weight_rules);
    seed_genome->initialize_randomly();
    Log::info("Generated RL seed genome for CartPole with four LIF hidden nodes\n");

    SpeciationStrategy* speciation_strategy = generate_speciation_strategy_from_arguments(arguments, seed_genome);

    GenomeProperty* genome_property = new GenomeProperty();
    genome_property->generate_genome_property_from_arguments(arguments);
    genome_property->set_parameter_names(input_parameter_names, output_parameter_names);

    int32_t island_size;
    int32_t number_islands;
    int32_t max_genomes = 0;
    int32_t max_wallclock_seconds = 0;
    string save_genome_option = "all_best_genomes";
    int32_t growth_phase_genomes = 0;
    int32_t reduction_phase_genomes = 0;
    int32_t genome_size_log = 0;
    int32_t is_harada_selection = 0;
    int32_t is_sweet = 0;
    double harada_selection_ratio = 0.0;
    bool generate_op_log = false;
    bool generate_visualization_json = false;

    get_argument(arguments, "--island_size", true, island_size);
    get_argument(arguments, "--number_islands", true, number_islands);
    get_argument(arguments, "--max_genomes", false, max_genomes);
    get_argument(arguments, "--max_wallclock_seconds", false, max_wallclock_seconds);
    get_argument(arguments, "--save_genome_option", false, save_genome_option);
    get_argument(arguments, "--growth_phase_genomes", false, growth_phase_genomes);
    get_argument(arguments, "--reduction_phase_genomes", false, reduction_phase_genomes);
    get_argument(arguments, "--genome_size_log", false, genome_size_log);
    get_argument(arguments, "--is_harada_selection", false, is_harada_selection);
    get_argument(arguments, "--harada_selection_ratio", false, harada_selection_ratio);
    get_argument(arguments, "--is_sweet", false, is_sweet);
    get_argument(arguments, "--generate_op_log", false, generate_op_log);
    get_argument(arguments, "--generate_visualization_json", false, generate_visualization_json);

    examm = new EXAMM(
        island_size, number_islands, max_genomes, max_wallclock_seconds, speciation_strategy, weight_rules,
        genome_property, output_directory, save_genome_option, generate_op_log, generate_visualization_json,
        growth_phase_genomes, reduction_phase_genomes, genome_size_log, is_harada_selection, harada_selection_ratio,
        is_sweet, possible_node_type_strings
    );

    write_rl_log_header();

    vector<thread> threads;
    for (int32_t i = 0; i < number_threads; i++) {
        threads.push_back(thread(snn_rl_thread, i));
    }

    for (int32_t i = 0; i < number_threads; i++) {
        threads[i].join();
    }

    if (rl_write_trace) {
        write_best_episode_trace(examm->get_best_genome());
    }

    ofstream completed(output_directory + "/completed");
    completed.close();

    if (rl_log_file.is_open()) {
        rl_log_file.close();
    }

    Log::info("completed CartPole RL run!\n");
    Log::release_id("main");

    delete examm;

    return 0;
}
