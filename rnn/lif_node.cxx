#include <cmath>
#include <fstream>
using std::ostream;

#include <iomanip>
using std::setw;

#include <string>
using std::string;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <vector>
using std::vector;

#include "common/log.hxx"
#include "common/random.hxx"
#include "lif_node.hxx"
#include "mse.hxx"
#include "rnn_node_interface.hxx"

#define NUMBER_LIF_WEIGHTS 3

LIF_Node::LIF_Node(int32_t _innovation_number, int32_t _type, double _depth)
    : RNN_Node_Interface(_innovation_number, _type, _depth) {
    node_type = LIF_NODE;
}

LIF_Node::~LIF_Node() {
}

void LIF_Node::initialize_lamarckian(
    minstd_rand0& generator, NormalDistribution& normal_distribution, double mu, double sigma
) {
    v_thresh = bound(normal_distribution.random(generator, mu, sigma));
    beta = bound(normal_distribution.random(generator, mu, sigma));
    bias = bound(normal_distribution.random(generator, mu, sigma));
}

void LIF_Node::initialize_xavier(minstd_rand0& generator, uniform_real_distribution<double>& rng_1_1, double range) {
    v_thresh = range * (rng_1_1(generator));
    beta = range * (rng_1_1(generator));
    bias = range * (rng_1_1(generator));
}

void LIF_Node::initialize_kaiming(minstd_rand0& generator, NormalDistribution& normal_distribution, double range) {
    v_thresh = range * normal_distribution.random(generator, 0, 1);
    beta = range * normal_distribution.random(generator, 0, 1);
    bias = range * normal_distribution.random(generator, 0, 1);
}

void LIF_Node::initialize_uniform_random(minstd_rand0& generator, uniform_real_distribution<double>& rng) {
    v_thresh = rng(generator);
    beta = rng(generator);
    bias = rng(generator);
}

double LIF_Node::get_gradient(string gradient_name) {
    double gradient_sum = 0.0;

    for (int32_t i = 0; i < series_length; i++) {
        if (gradient_name == "v_thresh") {
            gradient_sum += d_v_thresh[i];
        } else if (gradient_name == "beta") {
            gradient_sum += d_beta[i];
        } else if (gradient_name == "bias") {
            gradient_sum += d_bias[i];
        } else {
            Log::fatal("ERROR: tried to get unknown gradient: '%s'\n", gradient_name.c_str());
            exit(1);
        }
    }

    return gradient_sum;
}

void LIF_Node::print_gradient(string gradient_name) {
    Log::info("\tgradient['%s']: %lf\n", gradient_name.c_str(), get_gradient(gradient_name));
}

void LIF_Node::input_fired(int32_t time, double incoming_output) {
    inputs_fired[time]++;

    input_values[time] += incoming_output;

    if (inputs_fired[time] < total_inputs) {
        return;
    } else if (inputs_fired[time] > total_inputs) {
        Log::fatal(
            "ERROR: inputs_fired on LIF_Node %d at time %d is %d and total_inputs is %d\n", innovation_number, time,
            inputs_fired[time], total_inputs
        );
        exit(1);
    }

    double x = input_values[time];

    double v_prev = 0.0;
    if (time > 0 && spike_output[time - 1] <= 0.5) {
        v_prev = membrane_potential[time - 1];
    }

    double v = beta * v_prev + x + bias;
    membrane_potential[time] = v;
    spike_output[time] = (v >= v_thresh) ? 1.0 : 0.0;
    output_values[time] = spike_output[time];
}

void LIF_Node::try_update_deltas(int32_t time) {
    if (outputs_fired[time] < total_outputs) {
        return;
    } else if (outputs_fired[time] > total_outputs) {
        Log::fatal(
            "ERROR: outputs_fired on LIF_Node %d at time %d is %d and total_outputs is %d\n", innovation_number, time,
            outputs_fired[time], total_outputs
        );
        exit(1);
    }

    double v = membrane_potential[time];

    double v_prev = 0.0;
    if (time > 0 && spike_output[time - 1] <= 0.5) {
        v_prev = membrane_potential[time - 1];
    }

    // Surrogate gradient: fast sigmoid approximation
    double diff = v - v_thresh;
    double sg = 1.0 / ((1.0 + 25.0 * fabs(diff)) * (1.0 + 25.0 * fabs(diff)));

    double d_h = error_values[time];
    if (time < (series_length - 1)) {
        d_h += d_membrane[time + 1];
    }

    double d_v = d_h * sg;

    d_bias[time] = d_v;
    d_beta[time] = d_v * v_prev;
    d_v_thresh[time] = -d_h * sg;

    double reset_mask = 1.0;
    if (time > 0 && spike_output[time - 1] > 0.5) {
        reset_mask = 0.0;
    }

    d_membrane[time] = d_v * beta * reset_mask;
    d_input[time] = d_v;
}

void LIF_Node::error_fired(int32_t time, double error) {
    outputs_fired[time]++;

    error_values[time] *= error;

    try_update_deltas(time);
}

void LIF_Node::output_fired(int32_t time, double delta) {
    outputs_fired[time]++;

    error_values[time] += delta;

    try_update_deltas(time);
}

int32_t LIF_Node::get_number_weights() const {
    return NUMBER_LIF_WEIGHTS;
}

void LIF_Node::get_weights(vector<double>& parameters) const {
    parameters.resize(get_number_weights());
    int32_t offset = 0;
    get_weights(offset, parameters);
}

void LIF_Node::set_weights(const vector<double>& parameters) {
    int32_t offset = 0;
    set_weights(offset, parameters);
}

void LIF_Node::set_weights(int32_t& offset, const vector<double>& parameters) {
    v_thresh = bound(parameters[offset++]);
    beta = bound(parameters[offset++]);
    bias = bound(parameters[offset++]);
}

void LIF_Node::get_weights(int32_t& offset, vector<double>& parameters) const {
    parameters[offset++] = v_thresh;
    parameters[offset++] = beta;
    parameters[offset++] = bias;
}

void LIF_Node::get_gradients(vector<double>& gradients) {
    gradients.assign(NUMBER_LIF_WEIGHTS, 0.0);

    for (int32_t i = 0; i < series_length; i++) {
        gradients[0] += d_v_thresh[i];
        gradients[1] += d_beta[i];
        gradients[2] += d_bias[i];
    }
}

void LIF_Node::reset(int32_t _series_length) {
    series_length = _series_length;

    d_v_thresh.assign(series_length, 0.0);
    d_beta.assign(series_length, 0.0);
    d_bias.assign(series_length, 0.0);

    d_membrane.assign(series_length, 0.0);

    membrane_potential.assign(series_length, 0.0);
    spike_output.assign(series_length, 0.0);

    // reset values from rnn_node_interface
    d_input.assign(series_length, 0.0);
    error_values.assign(series_length, 0.0);

    input_values.assign(series_length, 0.0);
    output_values.assign(series_length, 0.0);

    inputs_fired.assign(series_length, 0);
    outputs_fired.assign(series_length, 0);
}

RNN_Node_Interface* LIF_Node::copy() const {
    LIF_Node* n = new LIF_Node(innovation_number, layer_type, depth);

    // copy LIF_Node values
    n->v_thresh = v_thresh;
    n->beta = beta;
    n->bias = bias;

    n->d_v_thresh = d_v_thresh;
    n->d_beta = d_beta;
    n->d_bias = d_bias;

    n->d_membrane = d_membrane;

    n->membrane_potential = membrane_potential;
    n->spike_output = spike_output;

    // copy RNN_Node_Interface values
    n->series_length = series_length;
    n->input_values = input_values;
    n->output_values = output_values;
    n->error_values = error_values;
    n->d_input = d_input;

    n->inputs_fired = inputs_fired;
    n->total_inputs = total_inputs;
    n->outputs_fired = outputs_fired;
    n->total_outputs = total_outputs;
    n->enabled = enabled;
    n->forward_reachable = forward_reachable;
    n->backward_reachable = backward_reachable;

    return n;
}

void LIF_Node::write_to_stream(ostream& out) {
    RNN_Node_Interface::write_to_stream(out);
}
