#include <algorithm>
#include <chrono>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common/arguments.hxx"
#include "common/log.hxx"
#include "common/process_arguments.hxx"
#include "exact_rl_tools/rl_environment.hxx"
#include "exact_rl_tools/rl_evaluator.hxx"
#include "exact_rl_tools/rl_local_search.hxx"
#include "examm/examm.hxx"
#include "rnn/rnn_genome.hxx"
#include "weights/weight_rules.hxx"

using std::lock_guard;
using std::mutex;
using std::ofstream;
using std::string;
using std::thread;
using std::to_string;
using std::vector;

mutex examm_mutex;
mutex log_mutex;

vector<string> arguments;

EXAMM* examm;
ofstream rl_log_file;

string output_directory;
RLEvaluationOptions rl_options;
RLLocalSearchOptions local_search_options;
RLEnvironmentSpec rl_spec;
int32_t number_threads = 1;
int32_t evaluated_genomes = 0;
int32_t rl_seed_hidden_nodes = 4;
bool rl_write_trace = true;
double global_best_reward = -1.0e300;

void write_rl_log_header() {
    rl_log_file.open(output_directory + "/rl_fitness_log.csv");
    rl_log_file
        << "evaluated_genomes,generation_id,thread_id,env,decoder,seed,episodes,avg_reward,"
           "episode_best_reward,episode_worst_reward,evaluation_best_reward,evaluation_mean_reward,"
           "evaluation_worst_reward,best_avg_reward,fitness,steps_total,enabled_nodes,enabled_lif_nodes,enabled_edges,"
           "enabled_recurrent_edges,"
           "evaluation_ms,local_search_method,local_search_iterations,initial_avg_reward,"
           "local_search_evaluations,local_search_ms,local_search_improved,inserted\n";
    rl_log_file.flush();
}

void write_rl_log_row(
    int32_t row, RNN_Genome* genome, int32_t thread_id, const RLLocalSearchResult& local_search_result, double fitness,
    bool inserted
) {
    const RLEvaluation& evaluation = local_search_result.final_evaluation;
    rl_log_file << row << "," << genome->get_generation_id() << "," << thread_id << "," << rl_spec.name << ","
                << action_decoder_name(rl_options.action_decoder) << "," << rl_options.seed << ","
                << rl_options.episodes << "," << evaluation.average_reward << "," << evaluation.episode_best_reward
                << "," << evaluation.episode_worst_reward << "," << local_search_result.best_evaluation_reward << ","
                << local_search_result.mean_evaluation_reward << "," << local_search_result.worst_evaluation_reward
                << "," << global_best_reward << "," << fitness << "," << evaluation.total_steps << ","
                << genome->get_enabled_node_count() << "," << genome->get_enabled_node_count(LIF_NODE) << ","
                << genome->get_enabled_edge_count() << "," << genome->get_enabled_recurrent_edge_count() << ","
                << evaluation.evaluation_milliseconds << "," << rl_local_search_method_name(local_search_options.method)
                << "," << local_search_options.iterations << ","
                << local_search_result.initial_evaluation.average_reward << "," << local_search_result.evaluations
                << "," << local_search_result.local_search_milliseconds << "," << (local_search_result.improved ? 1 : 0)
                << "," << (inserted ? 1 : 0) << "\n";
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
        RLLocalSearchResult local_search_result = run_rl_local_search(genome, rl_options, local_search_options);
        RLEvaluation evaluation = local_search_result.final_evaluation;
        double fitness = -evaluation.average_reward;
        genome->record_external_evaluation(
            fitness, evaluation.average_reward, local_search_result.local_search_milliseconds
        );
        Log::info(
            "generation %d env: %s avg_reward: %lf initial_reward: %lf fitness: %lf steps: %d local_search: %s "
            "evals: %d improved: %d\n",
            genome->get_generation_id(), rl_spec.name.c_str(), evaluation.average_reward,
            local_search_result.initial_evaluation.average_reward, fitness, evaluation.total_steps,
            rl_local_search_method_name(local_search_options.method).c_str(), local_search_result.evaluations,
            local_search_result.improved ? 1 : 0
        );
        Log::release_id(log_id);

        examm_mutex.lock();
        Log::set_id("main");
        bool inserted = examm->insert_genome(genome);
        examm_mutex.unlock();

        lock_guard<mutex> guard(log_mutex);
        evaluated_genomes++;
        global_best_reward = std::max(global_best_reward, evaluation.average_reward);
        write_rl_log_row(evaluated_genomes, genome, id, local_search_result, fitness, inserted);

        delete genome;
    }
}

void write_best_episode_trace(RNN_Genome* genome) {
    if (genome == NULL) {
        return;
    }

    ofstream trace(output_directory + "/best_episode_trace.csv");
    trace << "step";
    for (const auto& name : rl_spec.observation_names) {
        trace << "," << name;
    }
    for (const auto& name : rl_spec.action_names) {
        trace << ",raw_" << name;
    }
    if (rl_options.action_decoder == RLActionDecoder::DISCRETE_ARGMAX) {
        trace << ",decoded_action";
    } else {
        for (const auto& name : rl_spec.action_names) {
            trace << ",decoded_" << name;
        }
    }
    trace << ",reward,terminated\n";

    vector<RLTraceRow> rows = trace_rl_episode(genome, rl_options);
    for (const auto& row : rows) {
        trace << row.step;
        for (double value : row.observation) {
            trace << "," << value;
        }
        for (double value : row.raw_action_output) {
            trace << "," << value;
        }
        for (double value : row.action) {
            trace << "," << value;
        }
        trace << "," << row.reward << "," << (row.terminated ? 1 : 0) << "\n";
    }
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
    get_argument(arguments, "--rl_environment", false, rl_options.environment_name);
    get_argument(arguments, "--rl_episodes", false, rl_options.episodes);
    get_argument(arguments, "--rl_t_sim", false, rl_options.t_sim);
    get_argument(arguments, "--rl_max_steps", false, rl_options.max_steps);
    get_argument(arguments, "--rl_seed", false, rl_options.seed);
    get_argument(arguments, "--rl_write_trace", false, rl_write_trace);
    get_argument(arguments, "--rl_seed_hidden_nodes", false, rl_seed_hidden_nodes);
    get_argument(arguments, "--rl_observation_clip", false, rl_options.observation_clip);
    string local_search_method;
    get_argument(arguments, "--rl_local_search", false, local_search_method);
    local_search_options.method = parse_rl_local_search_method(local_search_method);
    get_argument(arguments, "--rl_local_search_iterations", false, local_search_options.iterations);
    get_argument(arguments, "--rl_local_search_step", false, local_search_options.step);
    get_argument(arguments, "--rl_local_search_perturbation", false, local_search_options.perturbation);
    local_search_options.seed = rl_options.seed;
    get_argument(arguments, "--rl_local_search_seed", false, local_search_options.seed);
    string action_decoder;
    get_argument(arguments, "--rl_action_decoder", false, action_decoder);
    vector<string> possible_node_type_strings;
    get_argument_vector(arguments, "--possible_node_types", false, possible_node_type_strings);

    rl_spec = get_rl_environment_spec(rl_options.environment_name);
    rl_options.environment_name = rl_spec.name;
    rl_options.action_decoder = parse_action_decoder(action_decoder, rl_spec.default_decoder);

    if (rl_options.episodes <= 0 || rl_options.t_sim <= 0) {
        Log::fatal("--rl_episodes and --rl_t_sim must be > 0.\n");
        exit(1);
    }
    if (rl_seed_hidden_nodes <= 0) {
        Log::fatal("--rl_seed_hidden_nodes must be > 0.\n");
        exit(1);
    }
    if (local_search_options.iterations < 0) {
        Log::fatal("--rl_local_search_iterations must be >= 0.\n");
        exit(1);
    }
    if (local_search_options.step < 0.0) {
        Log::fatal("--rl_local_search_step must be >= 0.\n");
        exit(1);
    }
    if (local_search_options.perturbation < 0.0) {
        Log::fatal("--rl_local_search_perturbation must be >= 0.\n");
        exit(1);
    }

    WeightRules* weight_rules = new WeightRules();
    weight_rules->initialize_from_args(arguments);

    RNN_Genome* seed_genome = create_rl_seed_genome(rl_spec, rl_seed_hidden_nodes, weight_rules);
    seed_genome->initialize_randomly();
    Log::info(
        "Generated RL seed genome for %s with %d inputs, %d outputs, and %d LIF hidden nodes\n", rl_spec.name.c_str(),
        (int32_t) rl_spec.observation_names.size(), (int32_t) rl_spec.action_names.size(), rl_seed_hidden_nodes
    );

    SpeciationStrategy* speciation_strategy = generate_speciation_strategy_from_arguments(arguments, seed_genome);

    GenomeProperty* genome_property = new GenomeProperty();
    genome_property->generate_genome_property_from_arguments(arguments);
    genome_property->set_parameter_names(rl_spec.observation_names, rl_spec.action_names);

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

    Log::info("completed %s RL run!\n", rl_spec.name.c_str());
    Log::release_id("main");

    delete examm;

    return 0;
}
