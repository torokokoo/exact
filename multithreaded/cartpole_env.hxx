#ifndef EXACT_CARTPOLE_ENV_HXX
#define EXACT_CARTPOLE_ENV_HXX

#include <random>
#include <vector>
using std::minstd_rand0;
using std::uniform_real_distribution;
using std::vector;

struct CartPoleStep {
    vector<double> observation;
    double reward;
    bool terminated;
};

class CartPoleEnv {
   private:
    static constexpr double gravity = 9.8;
    static constexpr double masscart = 1.0;
    static constexpr double masspole = 0.1;
    static constexpr double total_mass = masscart + masspole;
    static constexpr double length = 0.5;
    static constexpr double polemass_length = masspole * length;
    static constexpr double force_mag = 10.0;
    static constexpr double tau = 0.02;
    static constexpr double theta_threshold_radians = 12.0 * 2.0 * 3.14159265358979323846 / 360.0;
    static constexpr double x_threshold = 2.4;

    vector<double> state;
    minstd_rand0 generator;
    uniform_real_distribution<double> reset_distribution;

   public:
    CartPoleEnv();

    vector<double> reset(uint32_t seed);
    CartPoleStep step(int32_t action);
    const vector<double>& get_state() const;
};

#endif
