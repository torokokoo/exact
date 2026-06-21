#ifndef EXACT_RL_TOOLS_ENVIRONMENT_HXX
#define EXACT_RL_TOOLS_ENVIRONMENT_HXX

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

enum class RLActionDecoder {
    DISCRETE_ARGMAX,
    CONTINUOUS_TANH,
};

struct RLStepResult {
    std::vector<double> observation;
    double reward;
    bool terminated;
};

struct RLEnvironmentSpec {
    std::string name;
    std::vector<std::string> observation_names;
    std::vector<std::string> action_names;
    RLActionDecoder default_decoder;
    int32_t default_max_steps;
};

class RLEnvironment {
   public:
    virtual ~RLEnvironment();
    virtual std::vector<double> reset(uint32_t seed) = 0;
    virtual RLStepResult step(const std::vector<double>& action) = 0;
    virtual const RLEnvironmentSpec& spec() const = 0;
};

std::string normalize_rl_environment_name(const std::string& name);
RLEnvironmentSpec get_rl_environment_spec(const std::string& name);
std::unique_ptr<RLEnvironment> make_rl_environment(const std::string& name);
RLActionDecoder parse_action_decoder(const std::string& decoder, RLActionDecoder fallback);
std::string action_decoder_name(RLActionDecoder decoder);

#endif
