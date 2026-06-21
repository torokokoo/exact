#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "exact_rl_tools/rl_environment.hxx"
#include "exact_rl_tools/rl_evaluator.hxx"
#include "weights/weight_rules.hxx"

using std::cerr;
using std::cout;
using std::string;
using std::vector;

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
    if (std::fabs(actual - expected) > tolerance) {
        cerr << "FAILED: " << message << " actual=" << actual << " expected=" << expected << "\n";
        exit(1);
    }
}

void test_specs() {
    RLEnvironmentSpec cartpole = get_rl_environment_spec("CartPole-v1");
    require(cartpole.name == "cartpole", "CartPole-v1 normalizes to cartpole");
    require(cartpole.observation_names.size() == 4, "cartpole has four observations");
    require(cartpole.action_names.size() == 2, "cartpole has two actions");
    require(cartpole.default_decoder == RLActionDecoder::DISCRETE_ARGMAX, "cartpole uses discrete decoder");

    RLEnvironmentSpec pendulum = get_rl_environment_spec("Pendulum-v1");
    require(pendulum.name == "pendulum", "Pendulum-v1 normalizes to pendulum");
    require(pendulum.observation_names.size() == 3, "pendulum has three observations");
    require(pendulum.action_names.size() == 1, "pendulum has one action");
    require(pendulum.default_decoder == RLActionDecoder::CONTINUOUS_TANH, "pendulum uses continuous decoder");
}

void test_cartpole_step() {
    std::unique_ptr<RLEnvironment> env = make_rl_environment("cartpole");
    vector<double> observation = env->reset(7);
    require(observation.size() == 4, "cartpole reset returns four observations");

    RLStepResult left = env->step({0.0});
    require(left.observation.size() == 4, "cartpole step returns four observations");
    require_close(left.reward, 1.0, "cartpole reward is one per nonterminal-style step");
}

void test_pendulum_step() {
    std::unique_ptr<RLEnvironment> env = make_rl_environment("pendulum");
    vector<double> observation = env->reset(7);
    require(observation.size() == 3, "pendulum reset returns Fourier observations");

    RLStepResult result = env->step({0.25});
    require(result.observation.size() == 3, "pendulum step returns Fourier observations");
    require(result.reward <= 0.0, "pendulum reward is negative cost");
}

void test_seed_genome_dimensions() {
    WeightRules weight_rules;
    RLEnvironmentSpec pendulum = get_rl_environment_spec("pendulum");
    RNN_Genome* genome = create_rl_seed_genome(pendulum, 3, &weight_rules);
    genome->initialize_randomly();
    require(genome->get_number_inputs() == 3, "pendulum seed genome has three inputs");
    require(genome->get_number_outputs() == 1, "pendulum seed genome has one output");
    require(genome->get_node_count(LIF_NODE) == 3, "seed genome has requested LIF nodes");
    delete genome;
}

int main() {
    cout << "test_specs\n";
    cout.flush();
    test_specs();
    cout << "test_cartpole_step\n";
    cout.flush();
    test_cartpole_step();
    cout << "test_pendulum_step\n";
    cout.flush();
    test_pendulum_step();
    cout << "test_seed_genome_dimensions\n";
    cout.flush();
    test_seed_genome_dimensions();
    cout << "ALL RL TOOLS ENVIRONMENT TESTS PASSED\n";
    return 0;
}
