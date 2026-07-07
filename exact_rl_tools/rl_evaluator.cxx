#include "rl_evaluator.hxx"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include "rnn/generate_nn.hxx"
#include "rnn/lif_node.hxx"
#include "rnn/rnn.hxx"
#include "rnn/rnn_edge.hxx"
#include "rnn/rnn_node.hxx"

using std::vector;

RNN_Genome* create_rl_seed_genome(const RLEnvironmentSpec& spec, int32_t hidden_nodes, WeightRules* weight_rules) {
    vector<RNN_Node_Interface*> nodes;
    vector<RNN_Edge*> edges;
    vector<RNN_Recurrent_Edge*> recurrent_edges;

    int32_t node_innovation = 0;
    int32_t edge_innovation = 0;

    vector<RNN_Node_Interface*> input_nodes;
    for (const auto& name : spec.observation_names) {
        RNN_Node* input_node = new RNN_Node(++node_innovation, INPUT_LAYER, 0.0, SIMPLE_NODE, name);
        nodes.push_back(input_node);
        input_nodes.push_back(input_node);
    }

    vector<RNN_Node_Interface*> hidden_lif_nodes;
    for (int32_t i = 0; i < hidden_nodes; i++) {
        LIF_Node* lif_node = new LIF_Node(++node_innovation, HIDDEN_LAYER, 0.5);
        nodes.push_back(lif_node);
        hidden_lif_nodes.push_back(lif_node);
    }

    vector<RNN_Node_Interface*> output_nodes;
    for (const auto& name : spec.action_names) {
        RNN_Node* output_node = new RNN_Node(++node_innovation, OUTPUT_LAYER, 1.0, SIMPLE_NODE, name);
        nodes.push_back(output_node);
        output_nodes.push_back(output_node);
    }

    for (auto input_node : input_nodes) {
        for (auto lif_node : hidden_lif_nodes) {
            edges.push_back(new RNN_Edge(++edge_innovation, input_node, lif_node));
        }
    }

    for (auto lif_node : hidden_lif_nodes) {
        for (auto output_node : output_nodes) {
            edges.push_back(new RNN_Edge(++edge_innovation, lif_node, output_node));
        }
    }

    RNN_Genome* genome = new RNN_Genome(nodes, edges, recurrent_edges, weight_rules);
    genome->set_parameter_names(spec.observation_names, spec.action_names);
    return genome;
}

std::vector<double> decode_rl_action(
    RNN* rnn, const std::vector<double>& observation, int32_t number_outputs, const RLEvaluationOptions& options,
    std::vector<double>* raw_outputs
) {
    vector<vector<double> > inputs(observation.size(), vector<double>(options.t_sim, 0.0));
    for (int32_t i = 0; i < (int32_t) observation.size(); i++) {
        double value = observation[i];
        value = std::max(-options.observation_clip, std::min(options.observation_clip, value));
        for (int32_t t = 0; t < options.t_sim; t++) {
            inputs[i][t] = value;
        }
    }

    vector<vector<double> > expected_outputs(number_outputs, vector<double>(options.t_sim, 0.0));
    vector<double> predictions = rnn->get_predictions(inputs, expected_outputs, false, 0.0);
    vector<double> action_scores(number_outputs, 0.0);

    for (int32_t t = 0; t < options.t_sim; t++) {
        for (int32_t action = 0; action < number_outputs; action++) {
            action_scores[action] += predictions[(t * number_outputs) + action];
        }
    }

    if (raw_outputs != nullptr) {
        *raw_outputs = action_scores;
    }

    if (options.action_decoder == RLActionDecoder::DISCRETE_ARGMAX) {
        int32_t best_action = 0;
        for (int32_t action = 1; action < number_outputs; action++) {
            if (action_scores[action] > action_scores[best_action]) {
                best_action = action;
            }
        }
        return {static_cast<double>(best_action)};
    }

    vector<double> continuous_action(number_outputs, 0.0);
    for (int32_t i = 0; i < number_outputs; i++) {
        continuous_action[i] = std::tanh(action_scores[i] / std::max(1, options.t_sim));
    }
    return continuous_action;
}

RLEvaluation evaluate_rl_genome(RNN_Genome* genome, const RLEvaluationOptions& options) {
    auto start = std::chrono::system_clock::now();
    RNN* rnn = genome->get_rnn();
    int32_t number_outputs = genome->get_number_outputs();
    int32_t max_steps = options.max_steps;
    if (max_steps <= 0) {
        max_steps = get_rl_environment_spec(options.environment_name).default_max_steps;
    }

    double total_reward = 0.0;
    double episode_best_reward = -std::numeric_limits<double>::max();
    double episode_worst_reward = std::numeric_limits<double>::max();
    int32_t total_steps = 0;

    for (int32_t episode = 0; episode < options.episodes; episode++) {
        std::unique_ptr<RLEnvironment> env = make_rl_environment(options.environment_name);
        vector<double> observation = env->reset(options.seed + episode);
        double episode_reward = 0.0;

        for (int32_t step = 0; step < max_steps; step++) {
            vector<double> action = decode_rl_action(rnn, observation, number_outputs, options);
            RLStepResult result = env->step(action);
            episode_reward += result.reward;
            total_steps++;
            observation = result.observation;

            if (result.terminated) {
                break;
            }
        }

        total_reward += episode_reward;
        episode_best_reward = std::max(episode_best_reward, episode_reward);
        episode_worst_reward = std::min(episode_worst_reward, episode_reward);
    }

    delete rnn;
    auto end = std::chrono::system_clock::now();
    long eval_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return {total_reward / options.episodes, episode_best_reward, episode_worst_reward, total_steps, eval_ms};
}

std::vector<RLTraceRow> trace_rl_episode(RNN_Genome* genome, const RLEvaluationOptions& options) {
    RNN* rnn = genome->get_rnn();
    int32_t number_outputs = genome->get_number_outputs();
    int32_t max_steps = options.max_steps;
    if (max_steps <= 0) {
        max_steps = get_rl_environment_spec(options.environment_name).default_max_steps;
    }

    std::unique_ptr<RLEnvironment> env = make_rl_environment(options.environment_name);
    vector<double> observation = env->reset(options.seed);
    vector<RLTraceRow> trace;

    for (int32_t step = 0; step < max_steps; step++) {
        vector<double> raw_outputs;
        vector<double> action = decode_rl_action(rnn, observation, number_outputs, options, &raw_outputs);
        RLStepResult result = env->step(action);
        trace.push_back({step, observation, raw_outputs, action, result.reward, result.terminated});
        observation = result.observation;
        if (result.terminated) {
            break;
        }
    }

    delete rnn;
    return trace;
}
