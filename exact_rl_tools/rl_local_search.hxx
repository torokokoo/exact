#ifndef EXACT_RL_TOOLS_LOCAL_SEARCH_HXX
#define EXACT_RL_TOOLS_LOCAL_SEARCH_HXX

#include <cstdint>
#include <string>
#include <vector>

#include "exact_rl_tools/rl_evaluator.hxx"
#include "rnn/rnn_genome.hxx"

enum class RLLocalSearchMethod {
    NONE,
    PERTURB,
    SPSA,
};

struct RLLocalSearchOptions {
    RLLocalSearchMethod method = RLLocalSearchMethod::NONE;
    int32_t iterations = 0;
    double step = 0.05;
    double perturbation = 0.10;
    int32_t seed = 1337;
};

struct RLLocalSearchResult {
    RLEvaluation initial_evaluation;
    RLEvaluation final_evaluation;
    int32_t evaluations = 0;
    double evaluation_reward_sum = 0.0;
    double best_evaluation_reward = 0.0;
    double mean_evaluation_reward = 0.0;
    double worst_evaluation_reward = 0.0;
    long local_search_milliseconds = 0;
    bool improved = false;
};

RLLocalSearchMethod parse_rl_local_search_method(const std::string& method);
std::string rl_local_search_method_name(RLLocalSearchMethod method);

RLEvaluation evaluate_rl_genome_with_parameters(
    RNN_Genome* genome, const std::vector<double>& parameters, const RLEvaluationOptions& evaluation_options
);

RLLocalSearchResult run_rl_local_search(
    RNN_Genome* genome, const RLEvaluationOptions& evaluation_options, const RLLocalSearchOptions& local_search_options
);

#endif
