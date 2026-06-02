# Training and Backpropagation — Inner Workings

---

## ELI5 — Simple Explanation

Imagine the network is a machine with a bunch of knobs (weights). You feed it some data, it produces a guess, and you measure how wrong that guess was. Then you work backwards through the machine to figure out: "if I had turned *this* knob a little bit, would the guess have been better or worse?" That measurement — the gradient — tells you which direction to turn each knob. You nudge all the knobs a tiny bit in the direction that makes the error smaller. You repeat this many times (epochs) across all your training data until the knobs are set well.

In a recurrent network, the same machine is run at every time step of a sequence (t=0, t=1, t=2…). Some knobs feed their output *back into the machine at a later time step* (that is the recurrence). When computing "how wrong was I?", you have to unroll the machine across all time steps and trace the blame backwards through time as well as backwards through the layers. This is called **Backpropagation Through Time (BPTT)**.

The final score ("fitness") that evolution uses is the **validation MSE** — how wrong the best-ever set of knob settings was on data the network had never seen during training.

---

## Data Layout

Training data arrives as a 3D vector:

```
inputs [series][timestep][input_dim]
outputs[series][timestep][output_dim]
```

- `series`: one entry per loaded CSV file (or per slice if `--train_sequence_length` is set)
- `timestep`: rows of the CSV, reordered by `time_offset` so that input row `t` predicts output row `t + time_offset`
- `input_dim` / `output_dim`: number of input/output columns

Validation data has the same shape but comes from the validation files and is never seen during weight updates.

---

## High-Level Training Flow

Both `backpropagate()` and `backpropagate_stochastic()` live in `rnn/rnn_genome.cxx`. The stochastic version is used in normal EXAMM runs.

```
backpropagate_stochastic(inputs, outputs, validation_inputs, validation_outputs, weight_update):

    parameters = initial_parameters          ← start from genome's birth weights
    velocity   = zeros(n_parameters)         ← optimizer state
    prev_velocity = zeros(n_parameters)

    rnn = get_rnn()                          ← create stateless executor from topology
    rnn.set_weights(parameters)

    ── Baseline (before any training) ──
    for each series i:
        rnn.get_analytic_gradient(parameters, inputs[i], outputs[i], ...)
    validation_mse = get_mse(parameters, validation_inputs, validation_outputs)
    initial_fitness_before_bp = validation_mse
    best_validation_mse = validation_mse
    best_parameters = parameters

    ── Training loop ──
    for epoch in 0..bp_iterations:
        shuffle_order = fisher_yates_shuffle([0..n_series])

        for k in shuffle_order:                   ← stochastic: one series at a time
            analytic_gradient = rnn.get_analytic_gradient(parameters, inputs[k], outputs[k])
            norm = weight_update.get_norm(analytic_gradient)

            if norm is NaN or Inf:
                best_validation_mse = NaN         ← mark genome as dead
                return

            weight_update.norm_gradients(analytic_gradient, norm)
            weight_update.update_weights(parameters, velocity, prev_velocity,
                                         analytic_gradient, epoch)

        validation_mse = get_mse(parameters, validation_inputs, validation_outputs)
        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_validation_mae = get_mae(...)
            best_parameters = parameters           ← snapshot the best weights seen

    set_weights(best_parameters)                   ← restore best-ever weights
    get_mu_sigma(best_parameters, mu, sigma)       ← compute stats for Lamarckian children
    bp_time_milliseconds = elapsed time
```

The key property: `best_parameters` is the weight snapshot from the **epoch with the lowest validation MSE**, not necessarily the last epoch. This acts as implicit early stopping.

---

## `get_analytic_gradient()` — One Step in Detail

Lives in `rnn/rnn.cxx`. Called once per training series per weight-update step.

```
get_analytic_gradient(parameters, inputs, outputs, → mse, → gradient):

    1. set_weights(parameters)
    2. forward_pass(inputs)
    3. mse = calculate_error_mse(outputs)
          ↳ fills output_nodes[i].error_values[t] = predicted[i][t] - expected[i][t]
    4. error_scale = mse * (1.0 / T) * 2.0
          ↳ derivative of MSE w.r.t. the raw output error
    5. backward_pass(error_scale)
    6. collect gradients:
          for each reachable node:  gradient += node.get_gradients()
          for each reachable edge:  gradient += edge.get_gradient()
          for each reachable recurrent_edge: gradient += recurrent_edge.get_gradient()
```

The non-stochastic `backpropagate()` uses a multi-threaded variant (`RNN_Genome::get_analytic_gradient`) that runs `forward_pass` for all series in parallel (one thread per series), then runs `backward_pass` sequentially and sums the gradients across all series.

---

## Forward Pass (`rnn/rnn.cxx::forward_pass`)

```
forward_pass(series_data):

    1. series_length = series_data[0].size()
    2. reset all nodes, edges, recurrent_edges   ← clears all time-indexed buffers

    3. first_propagate_forward() on all recurrent_edges
          ↳ injects 0 into output nodes for time steps [0..recurrent_depth-1]
             so their inputs_fired counters start correctly

    4. for t = 0..series_length-1:
          a. for each input node:
                input_node.input_fired(t, series_data[input_idx][t])
                ↳ since input nodes have no incoming edges, inputs_fired[t]
                  immediately reaches total_inputs; node fires instantly:
                  output_values[t] = series_data[input_idx][t]  (identity)

          b. for each edge (depth-sorted, so shallower first):
                edge.propagate_forward(t)
                ↳ output_node.input_fired(t, weight * input_node.output_values[t])
                ↳ when output_node.inputs_fired[t] == total_inputs → node computes
                  its activation and stores output_values[t]

          c. for each recurrent_edge:
                recurrent_edge.propagate_forward(t)
                ↳ injects input_node.output_values[t] * weight into
                  output_node at time t + recurrent_depth
```

**Counter-based scheduling**: no explicit topological sort at runtime. Each node fires exactly once per time step, automatically, when all its incoming edge signals have arrived.

**Activation computations** (done inside each node when `inputs_fired[t] == total_inputs`):

| Node | `output_values[t]` |
|------|--------------------|
| RNN_Node | `tanh(input_values[t] + bias)` |
| LSTM | `output_gate[t] * cell[t]` (gates via sigmoid, cell via forget+input) |
| GRU | `z[t]*h_prev + (1-z[t])*h_tanh[t]` |
| MGU | `(1-f[t])*h_prev + f[t]*h_tanh[t]` |
| UGRNN | `g[t]*h_prev + (1-g[t])*c[t]` |
| Delta | `tanh(z_cap[t]*(1-r[t]) + r[t]*h_prev)` |
| SIN/COS/TANH/SIGMOID/INVERSE | elementwise math on `input_values[t]` |
| SUM | `input_values[t]` (identity) |
| MULTIPLY | `product(all individual inputs at t)` |

For recurrent nodes, `h_prev = output_values[t-1]` (or 0 at t=0). The recurrent edge carries this value: `output_values[t]` is injected at the node itself via self-reference, not through a recurrent_edge.

---

## Loss Computation (`calculate_error_mse`)

```
for each output node i:
    for each timestep t:
        error = predicted[i][t] - expected[i][t]
        output_node[i].error_values[t] = error    ← stored for backward pass
        mse_i += error * error

    mse_i /= T                                     ← average over time steps
    mse_sum += mse_i                               ← sum over output dimensions

return mse_sum
```

`error_values` are stored directly on each output node — the backward pass reads them from there.

---

## Backward Pass — BPTT (`rnn/rnn.cxx::backward_pass`)

```
backward_pass(error_scale):

    1. first_propagate_backward() on all recurrent_edges
          ↳ injects 0 into input nodes for the last recurrent_depth time steps
            so their outputs_fired counters start correctly

    2. for t = series_length-1 .. 0:    ← reverse time order

          a. for each output node:
                output_node.error_fired(t, error_scale)
                ↳ d_input[t] = error_values[t] * error_scale * ld_output[t]
                  (ld_output = local derivative of the activation function)

          b. for each edge (reverse depth order, deepest first):
                edge.propagate_backward(t)
                ↳ delta = output_node.d_input[t]
                  d_weight += delta * input_node.output_values[t]
                  input_node.output_fired(t, delta * weight)
                  (when outputs_fired[t] == total_outputs → node fires backward)

          c. for each recurrent_edge (reverse order):
                recurrent_edge.propagate_backward(t)
                ↳ delta = output_node.d_input[t]
                  d_weight += delta * input_node.output_values[t - recurrent_depth]
                  input_node.output_fired(t - recurrent_depth, delta * weight)
```

The same counter mechanism as forward pass: a node fires its own backward computation only when it has received gradient signals from all its outgoing edges. This ensures correct aggregation without explicit scheduling.

### BPTT across recurrent connections

A recurrent edge with depth `d` routes the gradient from time `t` back to time `t - d`. This means errors propagate backward in both the network topology and in time simultaneously. LSTM's `d_prev_cell[t+1]` is an additional channel that carries cell-state gradients to earlier time steps, enabling long-range dependency learning without vanishing gradients.

### Node backward computation (when `outputs_fired[t] == total_outputs`)

Each node applies the chain rule using its stored local derivative:

```
d_input[t] *= ld_output[t]        ← chain rule through activation function
d_bias     += d_input[t]          ← gradient for bias (summed across all t)
```

For gated nodes (LSTM/GRU/MGU/UGRNN/Delta), multiple intermediate values stored during the forward pass (gate values, local derivatives) are read back to compute gradients for each gate weight. These gradients accumulate across all time steps.

---

## Gradient Collection

After `backward_pass`, each node and edge holds accumulated gradient values (summed across all time steps). `get_analytic_gradient` reads them out into a flat vector in this order:

```
1. Node weights (in nodes[] order):
      for each reachable node: node.get_gradients()
         → returns [d_bias] for RNN_Node
         → returns [d_output_gate_update_weight, ..., d_cell_bias] for LSTM (11 values)
         → etc.

2. Edge weights (in edges[] order):
      for each reachable edge: edge.get_gradient()
         → returns d_weight

3. Recurrent edge weights (in recurrent_edges[] order):
      for each reachable recurrent_edge: recurrent_edge.get_gradient()
         → returns d_weight
```

This flat vector is the same size and order as `parameters` (the flat weight vector), enabling direct elementwise optimizer updates.

---

## Gradient Norm Thresholding

Applied to the gradient vector **before** the optimizer update, in `weight_update.norm_gradients()`:

```
L2 = sqrt(sum(g[i]^2))

if L2 > high_threshold (default 1.0):
    g[i] *= high_threshold / L2      ← scale down: prevents exploding gradients

if 0 < L2 < low_threshold (default 0.05):
    g[i] *= low_threshold / L2       ← scale up: prevents vanishing gradients
```

If `L2` is NaN or Inf, the genome is marked as dead (`best_validation_mse = NaN`) and training aborts immediately — a NaN gradient means the weights have diverged and further training is pointless.

---

## Weight Update Step

`weight_update.update_weights(parameters, velocity, prev_velocity, gradient, epoch)` modifies `parameters` in-place. The exact formula depends on the chosen optimizer (see `docs/weights.md`). The default in EXAMM runs is **adam** or **rmsprop**.

After the update, all weights are clipped to `[-10, 10]`.

The optimizer's state vectors (`velocity`, `prev_velocity`) are **local to one genome's training session** — initialized to zero at the start of `backpropagate_stochastic` and discarded at the end.

---

## Validation Check and Best Snapshot

After all series in an epoch have been processed:

```
validation_mse = get_mse(parameters, validation_inputs, validation_outputs)

if validation_mse < best_validation_mse:
    best_validation_mse = validation_mse
    best_validation_mae = get_mae(...)
    best_parameters = copy of parameters       ← snapshot
```

`get_mse` runs a pure forward pass (no backward, no gradient) on the validation data using the current `parameters` without modifying any state.

At the end of training, `set_weights(best_parameters)` restores the best-ever snapshot. This is what gets inserted into the island population and what becomes `initial_parameters` for any child genome that inherits from this one.

---

## Full Data-Flow Diagram

```
initial_parameters
       │
       ▼
┌──────────────────────────────────────────────────────┐
│  for epoch in 0..bp_iterations                       │
│    shuffle training series                           │
│    for each series k (shuffled):                     │
│                                                      │
│      ┌─── forward_pass(inputs[k]) ─────────────────┐ │
│      │  t=0: inject inputs → edges fire → nodes    │ │
│      │        activate → recurrent edges carry      │ │
│      │        values to t+depth                     │ │
│      │  t=1..T: same, using previous outputs        │ │
│      └─────────────────────────────────────────────┘ │
│                                                      │
│      mse = calculate_error_mse(outputs[k])           │
│        → stores error_values[t] on output nodes      │
│                                                      │
│      ┌─── backward_pass(mse * 2/T) ───────────────┐  │
│      │  t=T-1..0: output nodes receive error       │  │
│      │    → edges propagate delta backward          │  │
│      │    → recurrent edges carry delta to t-depth  │  │
│      │    → each node accumulates d_weight          │  │
│      └─────────────────────────────────────────────┘  │
│                                                      │
│      gradient = collect all d_weight values          │
│      norm_gradients(gradient)                        │
│      update_weights(parameters, gradient)  ← in-place│
│                                                      │
│    end series loop                                   │
│    check validation_mse → snapshot if best          │
│  end epoch loop                                      │
└──────────────────────────────────────────────────────┘
       │
       ▼
  best_parameters  →  genome.best_validation_mse  →  island fitness
```
