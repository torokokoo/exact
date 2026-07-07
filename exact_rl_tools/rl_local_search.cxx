#include "rl_local_search.hxx"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <random>

#include "common/log.hxx"

using std::string;
using std::vector;

RLLocalSearchMethod parse_rl_local_search_method(const string& method) {
    if (method == "" || method == "none") {
        return RLLocalSearchMethod::NONE;
    }
    if (method == "perturb") {
        return RLLocalSearchMethod::PERTURB;
    }
    if (method == "spsa") {
        return RLLocalSearchMethod::SPSA;
    }

    Log::fatal("Unsupported --rl_local_search '%s'. Supported: none, perturb, spsa.\n", method.c_str());
    exit(1);
}

string rl_local_search_method_name(RLLocalSearchMethod method) {
    if (method == RLLocalSearchMethod::PERTURB) {
        return "perturb";
    }
    if (method == RLLocalSearchMethod::SPSA) {
        return "spsa";
    }
    return "none";
}

RLEvaluation evaluate_rl_genome_with_parameters(
    RNN_Genome* genome, const vector<double>& parameters, const RLEvaluationOptions& evaluation_options
) {
    vector<double> original_parameters;
    genome->get_weights(original_parameters);
    genome->set_weights(parameters);
    RLEvaluation evaluation = evaluate_rl_genome(genome, evaluation_options);
    genome->set_weights(original_parameters);
    return evaluation;
}

static void add_scaled_direction(
    vector<double>& candidate, const vector<double>& base, const vector<double>& direction
) {
    candidate.resize(base.size());
    for (int32_t i = 0; i < (int32_t) base.size(); i++) {
        candidate[i] = base[i] + direction[i];
    }
}

static void record_local_search_evaluation(RLLocalSearchResult& result, const RLEvaluation& evaluation) {
    result.evaluations++;
    result.evaluation_reward_sum += evaluation.average_reward;

    if (result.evaluations == 1) {
        result.best_evaluation_reward = evaluation.average_reward;
        result.worst_evaluation_reward = evaluation.average_reward;
    } else {
        result.best_evaluation_reward = std::max(result.best_evaluation_reward, evaluation.average_reward);
        result.worst_evaluation_reward = std::min(result.worst_evaluation_reward, evaluation.average_reward);
    }

    result.mean_evaluation_reward = result.evaluation_reward_sum / result.evaluations;
}

static RLLocalSearchResult initialize_local_search_result(const RLEvaluation& initial_evaluation) {
    RLLocalSearchResult result;
    result.initial_evaluation = initial_evaluation;
    result.final_evaluation = initial_evaluation;
    record_local_search_evaluation(result, initial_evaluation);
    return result;
}

static RLLocalSearchResult run_perturb_search(
    RNN_Genome* genome, const RLEvaluationOptions& evaluation_options, const RLLocalSearchOptions& local_options,
    const vector<double>& initial_parameters, const RLEvaluation& initial_evaluation
) {
    RLLocalSearchResult result = initialize_local_search_result(initial_evaluation);

    vector<double> best_parameters = initial_parameters;
    vector<double> candidate_parameters;

    std::minstd_rand0 generator(local_options.seed + genome->get_generation_id());
    std::normal_distribution<double> normal(0.0, local_options.perturbation);

    for (int32_t iteration = 0; iteration < local_options.iterations; iteration++) {
        candidate_parameters.resize(best_parameters.size());
        for (int32_t i = 0; i < (int32_t) best_parameters.size(); i++) {
            candidate_parameters[i] = best_parameters[i] + normal(generator);
        }

        RLEvaluation candidate_evaluation =
            evaluate_rl_genome_with_parameters(genome, candidate_parameters, evaluation_options);
        record_local_search_evaluation(result, candidate_evaluation);

        if (candidate_evaluation.average_reward > result.final_evaluation.average_reward) {
            best_parameters = candidate_parameters;
            result.final_evaluation = candidate_evaluation;
            result.improved = true;
        }
    }

    genome->set_weights(best_parameters);
    return result;
}

static RLLocalSearchResult run_spsa_search(
    RNN_Genome* genome, const RLEvaluationOptions& evaluation_options, const RLLocalSearchOptions& local_options,
    const vector<double>& initial_parameters, const RLEvaluation& initial_evaluation
) {
    RLLocalSearchResult result = initialize_local_search_result(initial_evaluation);

    vector<double> current_parameters = initial_parameters;
    vector<double> delta(current_parameters.size(), 0.0);
    vector<double> plus_direction(current_parameters.size(), 0.0);
    vector<double> minus_direction(current_parameters.size(), 0.0);
    vector<double> plus_parameters;
    vector<double> minus_parameters;
    vector<double> candidate_parameters(current_parameters.size(), 0.0);

    std::minstd_rand0 generator(local_options.seed + genome->get_generation_id());
    std::uniform_int_distribution<int32_t> sign_distribution(0, 1);

    for (int32_t iteration = 0; iteration < local_options.iterations; iteration++) {
        for (int32_t i = 0; i < (int32_t) current_parameters.size(); i++) {
            delta[i] = sign_distribution(generator) == 0 ? -1.0 : 1.0;
            plus_direction[i] = local_options.perturbation * delta[i];
            minus_direction[i] = -local_options.perturbation * delta[i];
        }

        add_scaled_direction(plus_parameters, current_parameters, plus_direction);
        add_scaled_direction(minus_parameters, current_parameters, minus_direction);

        RLEvaluation plus_evaluation = evaluate_rl_genome_with_parameters(genome, plus_parameters, evaluation_options);
        RLEvaluation minus_evaluation =
            evaluate_rl_genome_with_parameters(genome, minus_parameters, evaluation_options);
        record_local_search_evaluation(result, plus_evaluation);
        record_local_search_evaluation(result, minus_evaluation);

        double reward_delta = plus_evaluation.average_reward - minus_evaluation.average_reward;
        double scale = local_options.step * reward_delta / (2.0 * local_options.perturbation);
        for (int32_t i = 0; i < (int32_t) current_parameters.size(); i++) {
            candidate_parameters[i] = current_parameters[i] + scale * delta[i];
        }

        RLEvaluation candidate_evaluation =
            evaluate_rl_genome_with_parameters(genome, candidate_parameters, evaluation_options);
        record_local_search_evaluation(result, candidate_evaluation);

        if (candidate_evaluation.average_reward > result.final_evaluation.average_reward) {
            current_parameters = candidate_parameters;
            result.final_evaluation = candidate_evaluation;
            result.improved = true;
        }
    }

    genome->set_weights(current_parameters);
    return result;
}

RLLocalSearchResult run_rl_local_search(
    RNN_Genome* genome, const RLEvaluationOptions& evaluation_options, const RLLocalSearchOptions& local_options
) {
    auto start = std::chrono::system_clock::now();

    vector<double> initial_parameters;
    genome->get_weights(initial_parameters);
    RLEvaluation initial_evaluation = evaluate_rl_genome(genome, evaluation_options);

    RLLocalSearchResult result;
    if (local_options.method == RLLocalSearchMethod::NONE || local_options.iterations <= 0) {
        result = initialize_local_search_result(initial_evaluation);
        result.improved = false;
    } else if (local_options.method == RLLocalSearchMethod::PERTURB) {
        result = run_perturb_search(genome, evaluation_options, local_options, initial_parameters, initial_evaluation);
    } else {
        if (local_options.perturbation <= 0.0) {
            Log::fatal("--rl_local_search_perturbation must be > 0 for SPSA.\n");
            exit(1);
        }
        result = run_spsa_search(genome, evaluation_options, local_options, initial_parameters, initial_evaluation);
    }

    auto end = std::chrono::system_clock::now();
    result.local_search_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return result;
}
