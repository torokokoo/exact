#ifndef EXACT_RL_TOOLS_EVALUATOR_HXX
#define EXACT_RL_TOOLS_EVALUATOR_HXX

#include <cstdint>
#include <string>
#include <vector>

#include "exact_rl_tools/rl_environment.hxx"
#include "rnn/rnn_genome.hxx"
#include "weights/weight_rules.hxx"

struct RLEvaluationOptions {
    std::string environment_name = "cartpole";
    int32_t episodes = 3;
    int32_t t_sim = 5;
    int32_t max_steps = 0;
    int32_t seed = 1337;
    RLActionDecoder action_decoder = RLActionDecoder::DISCRETE_ARGMAX;
    double observation_clip = 10.0;
};

struct RLEvaluation {
    double average_reward;
    int32_t total_steps;
    long evaluation_milliseconds;
};

struct RLTraceRow {
    int32_t step;
    std::vector<double> observation;
    std::vector<double> raw_action_output;
    std::vector<double> action;
    double reward;
    bool terminated;
};

RNN_Genome* create_rl_seed_genome(const RLEnvironmentSpec& spec, int32_t hidden_nodes, WeightRules* weight_rules);
RLEvaluation evaluate_rl_genome(RNN_Genome* genome, const RLEvaluationOptions& options);
std::vector<RLTraceRow> trace_rl_episode(RNN_Genome* genome, const RLEvaluationOptions& options);
std::vector<double> decode_rl_action(
    RNN* rnn, const std::vector<double>& observation, int32_t number_outputs, const RLEvaluationOptions& options,
    std::vector<double>* raw_outputs = nullptr
);

#endif
