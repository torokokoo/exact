// clang-format off
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/rl/environments/pendulum/operations_cpu.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/random_uniform/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#include <rl_tools/rl/algorithms/sac/loop/core/config.h>
#include <rl_tools/rl/algorithms/sac/loop/core/operations_generic.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>
#include <rl_tools/rl/loop/steps/timing/config.h>
#include <rl_tools/rl/loop/steps/timing/operations_cpu.h>
// clang-format on

#include <iostream>

namespace rlt = rl_tools;

using TypePolicy = rlt::numeric_types::Policy<float>;
using Device = rlt::devices::DEVICE_FACTORY<>;
using RNG = typename Device::SPEC::RANDOM::ENGINE<>;
using T = float;
using TI = typename Device::index_t;
using PendulumSpec = rlt::rl::environments::pendulum::Specification<T, TI>;
using Environment = rlt::rl::environments::Pendulum<PendulumSpec>;

struct LoopCoreParameters : rlt::rl::algorithms::sac::loop::core::DefaultParameters<TypePolicy, TI, Environment> {
    struct SAC_PARAMETERS : rlt::rl::algorithms::sac::DefaultParameters<TypePolicy, TI, Environment::ACTION_DIM> {
        static constexpr TI ACTOR_BATCH_SIZE = 32;
        static constexpr TI CRITIC_BATCH_SIZE = 32;
    };
    static constexpr TI STEP_LIMIT = 100;
    static constexpr TI REPLAY_BUFFER_CAP = STEP_LIMIT;
    static constexpr TI ACTOR_NUM_LAYERS = 3;
    static constexpr TI ACTOR_HIDDEN_DIM = 32;
    static constexpr TI CRITIC_NUM_LAYERS = 3;
    static constexpr TI CRITIC_HIDDEN_DIM = 32;
    static constexpr TI N_ENVIRONMENTS = 1;
};

using LoopCoreConfig = rlt::rl::algorithms::sac::loop::core::Config<
    TypePolicy, TI, RNG, Environment, LoopCoreParameters, rlt::rl::algorithms::sac::loop::core::ConfigApproximatorsMLP>;

struct EvalParameters : rlt::rl::loop::steps::evaluation::Parameters<TypePolicy, TI, LoopCoreConfig> {
    static constexpr TI EVALUATION_INTERVAL = 50;
    static constexpr TI NUM_EVALUATION_EPISODES = 2;
    static constexpr TI N_EVALUATIONS = LoopCoreParameters::STEP_LIMIT / EVALUATION_INTERVAL;
};

using EvalConfig = rlt::rl::loop::steps::evaluation::Config<LoopCoreConfig, EvalParameters>;
using TimingConfig =
    rlt::rl::loop::steps::timing::Config<EvalConfig, rlt::rl::loop::steps::timing::Parameters<TI, 250>>;
using LoopState = TimingConfig::State<TimingConfig>;

int main(int argc, char** argv) {
    TI seed = 0;
    if (argc > 1) {
        seed = std::stoi(argv[1]);
    }
    std::cout << "starting rl-tools Pendulum SAC baseline seed=" << seed << "\n";
    std::cout.flush();

    Device device;
    LoopState state;
    rlt::malloc(device, state);
    rlt::init(device, state, seed);

    while (!rlt::step(device, state)) {
    }

    std::cout << "rl-tools Pendulum SAC baseline completed seed=" << seed << " steps=" << state.step << "\n";
    rlt::free(device, state);
    return 0;
}
