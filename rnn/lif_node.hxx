#ifndef EXAMM_LIF_NODE_HXX
#define EXAMM_LIF_NODE_HXX

#include <string>
using std::string;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <vector>
using std::vector;

#include "common/random.hxx"
#include "rnn_node_interface.hxx"

class LIF_Node : public RNN_Node_Interface {
   private:
    double v_thresh;
    double beta;
    double bias;

    vector<double> d_v_thresh;
    vector<double> d_beta;
    vector<double> d_bias;

    vector<double> d_membrane;

    vector<double> membrane_potential;
    vector<double> spike_output;

   public:
    LIF_Node(int32_t _innovation_number, int32_t _type, double _depth);
    ~LIF_Node();

    void initialize_lamarckian(
        minstd_rand0& generator, NormalDistribution& normal_distribution, double mu, double sigma
    );
    void initialize_xavier(minstd_rand0& generator, uniform_real_distribution<double>& rng1_1, double range);
    void initialize_kaiming(minstd_rand0& generator, NormalDistribution& normal_distribution, double range);
    void initialize_uniform_random(minstd_rand0& generator, uniform_real_distribution<double>& rng);

    double get_gradient(string gradient_name);
    void print_gradient(string gradient_name);

    void input_fired(int32_t time, double incoming_output);

    void try_update_deltas(int32_t time);
    void error_fired(int32_t time, double error);
    void output_fired(int32_t time, double delta);

    int32_t get_number_weights() const;

    void get_weights(vector<double>& parameters) const;
    void set_weights(const vector<double>& parameters);

    void get_weights(int32_t& offset, vector<double>& parameters) const;
    void set_weights(int32_t& offset, const vector<double>& parameters);

    void get_gradients(vector<double>& gradients);

    void reset(int32_t _series_length);

    void write_to_stream(ostream& out);

    RNN_Node_Interface* copy() const;

    friend class RNN_Edge;
};
#endif
