#include <cmath>
using std::fabs;

#include <cstdio>
using std::remove;

#include <cstdlib>
using std::exit;

#include <iostream>
using std::cerr;
using std::cout;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "rnn/generate_nn.hxx"
#include "rnn/lif_node.hxx"
#include "rnn/rnn_genome.hxx"
#include "rnn/rnn_node_interface.hxx"
#include "weights/weight_rules.hxx"

void fail(const string& message) {
    cerr << "FAILED: " << message << "\n";
    exit(1);
}

void require(bool condition, const string& message) {
    if (!condition) {
        fail(message);
    }
}

void require_close(double actual, double expected, const string& message, double tolerance = 1e-12) {
    if (fabs(actual - expected) > tolerance) {
        cerr << "FAILED: " << message << " actual=" << actual << " expected=" << expected
             << " tolerance=" << tolerance << "\n";
        exit(1);
    }
}

double lif_surrogate_gradient(double membrane_potential, double threshold) {
    double diff = membrane_potential - threshold;
    double scale = 1.0 + 25.0 * fabs(diff);
    return 1.0 / (scale * scale);
}

void test_registration_and_factory() {
    string lif = "lif";
    require(node_type_from_string(lif) == LIF_NODE, "node_type_from_string resolves lif");
    require(NODE_TYPES[LIF_NODE] == "lif", "NODE_TYPES labels LIF_NODE as lif");
    require(NUMBER_NODE_TYPES > LIF_NODE, "NUMBER_NODE_TYPES includes LIF_NODE");

    int32_t innovation = 10;
    RNN_Node_Interface* node = create_hidden_node(LIF_NODE, innovation, 0.5);
    require(node != nullptr, "create_hidden_node returns a LIF node");
    require(innovation == 11, "create_hidden_node increments innovation counter");
    require(node->get_node_type() == LIF_NODE, "created node has LIF type");
    require(node->get_layer_type() == HIDDEN_LAYER, "created LIF node is hidden");
    require_close(node->get_depth(), 0.5, "created LIF node depth");
    delete node;
}

void test_weight_management() {
    LIF_Node node(1, HIDDEN_LAYER, 0.5);
    require(node.get_number_weights() == 3, "LIF node exposes three evolvable parameters");

    vector<double> weights{15.0, -15.0, 0.25};
    node.set_weights(weights);

    vector<double> observed;
    node.get_weights(observed);
    require(observed.size() == 3, "LIF get_weights returns three parameters");
    require_close(observed[0], 10.0, "v_thresh is bounded on set_weights");
    require_close(observed[1], -10.0, "beta is bounded on set_weights");
    require_close(observed[2], 0.25, "bias is preserved on set_weights");

    int32_t offset = 0;
    vector<double> flat{1.25, 0.5, -0.1, 99.0};
    node.set_weights(offset, flat);
    require(offset == 3, "offset set_weights advances by three");
    observed.assign(4, 0.0);
    offset = 1;
    node.get_weights(offset, observed);
    require(offset == 4, "offset get_weights advances by three");
    require_close(observed[1], 1.25, "offset get_weights writes v_thresh");
    require_close(observed[2], 0.5, "offset get_weights writes beta");
    require_close(observed[3], -0.1, "offset get_weights writes bias");
}

void test_forward_dynamics() {
    LIF_Node node(1, HIDDEN_LAYER, 0.5);
    node.set_weights(vector<double>{1.0, 0.5, 0.0});
    node.reset(4);
    node.total_inputs = 1;
    node.total_outputs = 1;

    node.input_fired(0, 0.4);
    node.input_fired(1, 0.7);
    node.input_fired(2, 0.7);
    node.input_fired(3, 0.4);

    require_close(node.output_values[0], 0.0, "subthreshold first step does not spike");
    require_close(node.output_values[1], 0.0, "membrane accumulation can remain subthreshold");
    require_close(node.output_values[2], 1.0, "accumulated membrane crossing threshold spikes");
    require_close(node.output_values[3], 0.0, "previous spike hard-resets next timestep membrane");

    node.reset(2);
    require(node.output_values.size() == 2, "reset resizes output values");
    require_close(node.output_values[0], 0.0, "reset clears output values");
    require_close(node.d_input[0], 0.0, "reset clears input gradients");
    require(node.inputs_fired[0] == 0, "reset clears forward counters");
    require(node.outputs_fired[0] == 0, "reset clears backward counters");
}

void test_surrogate_backward_dynamics() {
    LIF_Node node(1, HIDDEN_LAYER, 0.5);
    node.set_weights(vector<double>{1.0, 0.5, 0.0});
    node.reset(3);
    node.total_inputs = 1;
    node.total_outputs = 1;

    node.input_fired(0, 0.4);
    node.input_fired(1, 0.7);
    node.input_fired(2, 0.7);

    node.output_fired(2, 2.0);
    node.output_fired(1, 1.0);
    node.output_fired(0, -0.5);

    const double v0 = 0.4;
    const double v1 = 0.9;
    const double v2 = 1.15;
    const double sg2 = lif_surrogate_gradient(v2, 1.0);
    const double dv2 = 2.0 * sg2;
    const double dmem2 = dv2 * 0.5;

    const double sg1 = lif_surrogate_gradient(v1, 1.0);
    const double dh1 = 1.0 + dmem2;
    const double dv1 = dh1 * sg1;
    const double dmem1 = dv1 * 0.5;

    const double sg0 = lif_surrogate_gradient(v0, 1.0);
    const double dh0 = -0.5 + dmem1;
    const double dv0 = dh0 * sg0;

    require_close(node.d_input[2], dv2, "time 2 d_input uses surrogate gradient");
    require_close(node.d_input[1], dv1, "time 1 d_input includes future membrane gradient");
    require_close(node.d_input[0], dv0, "time 0 d_input includes future membrane gradient");

    vector<double> gradients;
    node.get_gradients(gradients);
    require(gradients.size() == 3, "LIF get_gradients returns three parameters");
    require_close(gradients[0], -(2.0 * sg2) - (dh1 * sg1) - (dh0 * sg0), "summed v_thresh gradient");
    require_close(gradients[1], (dv2 * v1) + (dv1 * v0), "summed beta gradient");
    require_close(gradients[2], dv2 + dv1 + dv0, "summed bias gradient");
}

void test_genome_serialization_roundtrip() {
    WeightRules weight_rules;
    vector<string> input_names{"x"};
    vector<string> output_names{"y"};
    RNN_Genome* genome = create_lif(input_names, 1, 1, output_names, 1, &weight_rules);
    genome->initialize_randomly();

    require(genome->get_node_count(LIF_NODE) == 1, "create_lif creates one LIF hidden node");
    require(genome->get_enabled_node_count(LIF_NODE) == 1, "created LIF node is enabled");

    vector<double> before;
    genome->get_weights(before);

    const string path = "/private/tmp/exact_lif_roundtrip.bin";
    genome->write_to_file(path);

    RNN_Genome loaded(path);
    vector<double> after = loaded.get_best_parameters();

    require(loaded.get_node_count(LIF_NODE) == 1, "roundtrip preserves LIF node type");
    require(loaded.get_enabled_node_count(LIF_NODE) == 1, "roundtrip preserves enabled LIF node");
    require(before.size() == after.size(), "roundtrip preserves best-parameter vector size");
    for (int32_t i = 0; i < (int32_t) before.size(); i++) {
        require_close(after[i], before[i], "roundtrip preserves serialized best parameters");
    }

    loaded.set_weights(after);
    vector<double> restored;
    loaded.get_weights(restored);
    require(before.size() == restored.size(), "roundtrip best parameters can be restored to live weights");
    for (int32_t i = 0; i < (int32_t) before.size(); i++) {
        require_close(restored[i], before[i], "roundtrip restored live weights");
    }

    remove(path.c_str());
    delete genome;
}

int main() {
    test_registration_and_factory();
    test_weight_management();
    test_forward_dynamics();
    test_surrogate_backward_dynamics();
    test_genome_serialization_roundtrip();
    cout << "ALL LIF NODE TESTS PASSED\n";
    return 0;
}
