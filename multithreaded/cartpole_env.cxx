#include "cartpole_env.hxx"

#include <cmath>
using std::cos;
using std::sin;

#include <cstdint>

CartPoleEnv::CartPoleEnv() : state(4, 0.0), generator(0), reset_distribution(-0.05, 0.05) {
}

vector<double> CartPoleEnv::reset(uint32_t seed) {
    generator = minstd_rand0(seed);
    for (int32_t i = 0; i < 4; i++) {
        state[i] = reset_distribution(generator);
    }
    return state;
}

CartPoleStep CartPoleEnv::step(int32_t action) {
    double x = state[0];
    double x_dot = state[1];
    double theta = state[2];
    double theta_dot = state[3];

    double force = action == 1 ? force_mag : -force_mag;
    double costheta = cos(theta);
    double sintheta = sin(theta);

    double temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass;
    double thetaacc =
        (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass));
    double xacc = temp - polemass_length * thetaacc * costheta / total_mass;

    x += tau * x_dot;
    x_dot += tau * xacc;
    theta += tau * theta_dot;
    theta_dot += tau * thetaacc;

    state[0] = x;
    state[1] = x_dot;
    state[2] = theta;
    state[3] = theta_dot;

    bool terminated =
        x < -x_threshold || x > x_threshold || theta < -theta_threshold_radians || theta > theta_threshold_radians;

    return {state, 1.0, terminated};
}

const vector<double>& CartPoleEnv::get_state() const {
    return state;
}
