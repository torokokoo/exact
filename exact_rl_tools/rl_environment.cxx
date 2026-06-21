#include "rl_environment.hxx"

#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/pendulum/operations_cpu.h>

#include <algorithm>
#include <cmath>
#include <random>

#include "common/log.hxx"

namespace rlt = rl_tools;

RLEnvironment::~RLEnvironment() {
}

namespace {

constexpr double CARTPOLE_GRAVITY = 9.8;
constexpr double CARTPOLE_MASS_CART = 1.0;
constexpr double CARTPOLE_MASS_POLE = 0.1;
constexpr double CARTPOLE_TOTAL_MASS = CARTPOLE_MASS_CART + CARTPOLE_MASS_POLE;
constexpr double CARTPOLE_LENGTH = 0.5;
constexpr double CARTPOLE_POLE_MASS_LENGTH = CARTPOLE_MASS_POLE * CARTPOLE_LENGTH;
constexpr double CARTPOLE_FORCE_MAG = 10.0;
constexpr double CARTPOLE_TAU = 0.02;
constexpr double CARTPOLE_THETA_THRESHOLD = 12.0 * 2.0 * 3.14159265358979323846 / 360.0;
constexpr double CARTPOLE_X_THRESHOLD = 2.4;

const RLEnvironmentSpec CARTPOLE_SPEC{
    "cartpole", {"x", "x_dot", "theta", "theta_dot"},
     {"action_0", "action_1"},
     RLActionDecoder::DISCRETE_ARGMAX, 500,
};

const RLEnvironmentSpec PENDULUM_SPEC{
    "pendulum", {"cos_theta", "sin_theta", "theta_dot"},
     {"torque"},
     RLActionDecoder::CONTINUOUS_TANH, 200,
};

class CartPoleRLEnvironment : public RLEnvironment {
   private:
    std::vector<double> state;
    std::minstd_rand0 generator;
    std::uniform_real_distribution<double> reset_distribution;

   public:
    CartPoleRLEnvironment() : state(4, 0.0), generator(0), reset_distribution(-0.05, 0.05) {
    }

    std::vector<double> reset(uint32_t seed) override {
        generator = std::minstd_rand0(seed);
        for (int32_t i = 0; i < 4; i++) {
            state[i] = reset_distribution(generator);
        }
        return state;
    }

    RLStepResult step(const std::vector<double>& action) override {
        int32_t discrete_action = 0;
        if (action.size() > 0 && action[0] >= 0.5) {
            discrete_action = 1;
        }

        double x = state[0];
        double x_dot = state[1];
        double theta = state[2];
        double theta_dot = state[3];

        double force = discrete_action == 1 ? CARTPOLE_FORCE_MAG : -CARTPOLE_FORCE_MAG;
        double costheta = std::cos(theta);
        double sintheta = std::sin(theta);

        double temp = (force + CARTPOLE_POLE_MASS_LENGTH * theta_dot * theta_dot * sintheta) / CARTPOLE_TOTAL_MASS;
        double thetaacc =
            (CARTPOLE_GRAVITY * sintheta - costheta * temp)
            / (CARTPOLE_LENGTH * (4.0 / 3.0 - CARTPOLE_MASS_POLE * costheta * costheta / CARTPOLE_TOTAL_MASS));
        double xacc = temp - CARTPOLE_POLE_MASS_LENGTH * thetaacc * costheta / CARTPOLE_TOTAL_MASS;

        x += CARTPOLE_TAU * x_dot;
        x_dot += CARTPOLE_TAU * xacc;
        theta += CARTPOLE_TAU * theta_dot;
        theta_dot += CARTPOLE_TAU * thetaacc;

        state[0] = x;
        state[1] = x_dot;
        state[2] = theta;
        state[3] = theta_dot;

        bool terminated = x < -CARTPOLE_X_THRESHOLD || x > CARTPOLE_X_THRESHOLD || theta < -CARTPOLE_THETA_THRESHOLD
                          || theta > CARTPOLE_THETA_THRESHOLD;

        return {state, 1.0, terminated};
    }

    const RLEnvironmentSpec& spec() const override {
        return CARTPOLE_SPEC;
    }
};

class PendulumRLEnvironment : public RLEnvironment {
   private:
    using Device = rlt::devices::DefaultCPU;
    using TI = typename Device::index_t;
    using Value = double;
    using PendulumSpec = rlt::rl::environments::pendulum::Specification<
        Value, TI, rlt::rl::environments::pendulum::DefaultParameters<Value>>;
    using Environment = rlt::rl::environments::Pendulum<PendulumSpec>;
    using ActionMatrix = rlt::Matrix<rlt::matrix::Specification<Value, TI, 1, Environment::ACTION_DIM, false>>;
    using ObservationMatrix =
        rlt::Matrix<rlt::matrix::Specification<Value, TI, 1, Environment::Observation::DIM, false>>;

    Device device;
    typename Device::SPEC::RANDOM::ENGINE<> rng;
    Environment environment;
    Environment::Parameters parameters;
    Environment::State state;

    std::vector<double> observe_state() {
        ObservationMatrix observation;
        rlt::observe(device, environment, parameters, state, typename Environment::Observation{}, observation, rng);
        std::vector<double> values(Environment::Observation::DIM, 0.0);
        for (TI i = 0; i < Environment::Observation::DIM; i++) {
            values[i] = rlt::get(observation, 0, i);
        }
        return values;
    }

   public:
    PendulumRLEnvironment() {
        rlt::init(device);
        rlt::initial_parameters(device, environment, parameters);
    }

    std::vector<double> reset(uint32_t seed) override {
        rlt::init(device, rng, seed);
        rlt::sample_initial_state(device, environment, parameters, state, rng);
        return observe_state();
    }

    RLStepResult step(const std::vector<double>& action) override {
        ActionMatrix action_matrix;
        double raw_action = action.empty() ? 0.0 : action[0];
        raw_action = std::max(-1.0, std::min(1.0, raw_action));
        rlt::set(action_matrix, 0, 0, raw_action);

        Environment::State next_state;
        rlt::step(device, environment, parameters, state, action_matrix, next_state, rng);
        double reward = rlt::reward(device, environment, parameters, state, action_matrix, next_state, rng);
        bool terminated = rlt::terminated(device, environment, parameters, next_state, rng);
        state = next_state;
        return {observe_state(), reward, terminated};
    }

    const RLEnvironmentSpec& spec() const override {
        return PENDULUM_SPEC;
    }
};

}  // namespace

std::string normalize_rl_environment_name(const std::string& name) {
    std::string normalized = name;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
        return std::tolower(c);
    });

    if (normalized == "" || normalized == "cartpole" || normalized == "cartpole-v1") {
        return "cartpole";
    }
    if (normalized == "pendulum" || normalized == "pendulum-v1") {
        return "pendulum";
    }

    Log::fatal("Unsupported --rl_environment '%s'. Supported: cartpole, pendulum.\n", name.c_str());
    exit(1);
}

RLEnvironmentSpec get_rl_environment_spec(const std::string& name) {
    std::string normalized = normalize_rl_environment_name(name);
    if (normalized == "cartpole") {
        return CARTPOLE_SPEC;
    }
    return PENDULUM_SPEC;
}

std::unique_ptr<RLEnvironment> make_rl_environment(const std::string& name) {
    std::string normalized = normalize_rl_environment_name(name);
    if (normalized == "cartpole") {
        return std::make_unique<CartPoleRLEnvironment>();
    }
    return std::make_unique<PendulumRLEnvironment>();
}

RLActionDecoder parse_action_decoder(const std::string& decoder, RLActionDecoder fallback) {
    if (decoder == "") {
        return fallback;
    }
    std::string normalized = decoder;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    if (normalized == "discrete_argmax") {
        return RLActionDecoder::DISCRETE_ARGMAX;
    }
    if (normalized == "continuous_tanh") {
        return RLActionDecoder::CONTINUOUS_TANH;
    }

    Log::fatal("Unsupported --rl_action_decoder '%s'. Supported: discrete_argmax, continuous_tanh.\n", decoder.c_str());
    exit(1);
}

std::string action_decoder_name(RLActionDecoder decoder) {
    if (decoder == RLActionDecoder::DISCRETE_ARGMAX) {
        return "discrete_argmax";
    }
    return "continuous_tanh";
}
