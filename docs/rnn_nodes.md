# RNN Nodes and Edges — Inner Workings

## Inheritance Hierarchy

```
RNN_Node_Interface  (abstract base)
  ├── RNN_Node          (simple tanh/identity node)
  ├── LSTM_Node
  ├── GRU_Node
  ├── MGU_Node
  ├── UGRNN_Node
  ├── Delta_Node
  ├── SIN_Node / COS_Node / TANH_Node / SIGMOID_Node / INVERSE_Node
  ├── SUM_Node / MULTIPLY_Node
  └── *_GP variants of the above (same math, different gradient handling)
```

---

## `RNN_Node_Interface` — Base Class

### Core time-indexed buffers (allocated in `reset(series_length)`)
```cpp
double* input_values[t]      // accumulated weighted inputs at each time step
double* output_values[t]     // activation output at each time step
double* error_values[t]      // error signal injected by output edges during backprop
double* d_input[t]           // gradient w.r.t. input_values, propagated backward
double* ordered_d_input[t]   // per-incoming-edge gradient (MULTIPLY_NODE only)
int32_t inputs_fired[t]      // counts inputs received; fires when == total_inputs
int32_t outputs_fired[t]     // counts errors received; fires when == total_outputs
```

### Synchronization protocol
- **Forward**: An edge calls `node->input_fired(t, value)`. The node accumulates the value and fires its own forward computation only when `inputs_fired[t] == total_inputs`.
- **Backward**: An edge calls `node->output_fired(t, delta)`. The node accumulates the delta and fires its own backprop only when `outputs_fired[t] == total_outputs`.

This counter-based scheme replaces explicit scheduling — the graph executes itself in dependency order without a topological sort at runtime.

### Activation helpers (defined on the interface, used by subclasses)
```cpp
sigmoid(x)              = 1 / (1 + e^-x)
sigmoid_derivative(y)   = y * (1 - y)          // y is already sigmoid(x)
tanh_derivative(y)      = 1 - y * y             // y is already tanh(x)
bound(x)                = clamp(x, -10.0, 10.0)
```

### Reachability flags
`forward_reachable`, `backward_reachable`, `enabled` — set by `RNN_Genome::assign_reachability()`. Nodes/edges that are not reachable in both directions are skipped during forward/backward passes and excluded from the structural hash.

---

## Node Types

### `RNN_Node` — Simple Node
**Weights**: `bias` (1 total)

**Forward**:
```
output[t] = tanh(input_values[t] + bias)
ld_output[t] = tanh_derivative(output[t])
```
GP variant uses identity: `output[t] = input_values[t] + bias`.

**Backward**:
```
d_input[t] *= ld_output[t]
d_bias += d_input[t]         (accumulated across all t)
```

---

### `LSTM_Node` — Long Short-Term Memory
**Weights** (11 total, stored in this order):

| Index | Weight | Role |
|-------|--------|------|
| 0 | `output_gate_update_weight` | output gate ← previous cell state |
| 1 | `output_gate_weight` | output gate ← input |
| 2 | `output_gate_bias` | output gate bias |
| 3 | `input_gate_update_weight` | input gate ← previous cell state |
| 4 | `input_gate_weight` | input gate ← input |
| 5 | `input_gate_bias` | input gate bias |
| 6 | `forget_gate_update_weight` | forget gate ← previous cell state |
| 7 | `forget_gate_weight` | forget gate ← input |
| 8 | `forget_gate_bias` | forget gate bias (+1 offset applied at runtime) |
| 9 | `cell_weight` | cell candidate ← input |
| 10 | `cell_bias` | cell candidate bias |

**Forward** (using `x = input_values[t]`, `c_prev = cell_values[t-1]`):
```
forget_gate_bias += 1.0       ← bias trick for long-range dependencies

o[t] = sigmoid(output_gate_weight*x + output_gate_update_weight*c_prev + output_gate_bias)
i[t] = sigmoid(input_gate_weight*x  + input_gate_update_weight*c_prev  + input_gate_bias)
f[t] = sigmoid(forget_gate_weight*x + forget_gate_update_weight*c_prev + forget_gate_bias)
g[t] = tanh(cell_weight*x + cell_bias)

cell[t] = f[t]*c_prev + i[t]*g[t]
output[t] = o[t] * cell[t]          ← cell activation is identity (not tanh)

forget_gate_bias -= 1.0
```

**Backward**: Standard BPTT with `d_prev_cell[t+1]` carrying cell-state gradients across time steps.

---

### `GRU_Node` — Gated Recurrent Unit
**Weights** (9 total): `zw, zu, z_bias, rw, ru, r_bias, hw, hu, h_bias`

(z = update gate, r = reset gate, h = hidden candidate)

**Forward** (`x = input_values[t]`, `h = output_values[t-1]`):
```
z[t] = sigmoid(z_bias + h*zu + x*zw)
r[t] = sigmoid(r_bias + x*rw + h*ru)
h̃[t] = tanh(h_bias + x*hw + h*hu*r[t])
output[t] = z[t]*h + (1 - z[t])*h̃[t]
```

**Backward**: Gradients flow through both update gate and reset gate branches; `d_h_prev[t+1]` carries inter-step gradient.

---

### `MGU_Node` — Minimal Gated Unit
**Weights** (6 total): `fw, fu, f_bias, hw, hu, h_bias`

(f = forget/update gate, h = hidden candidate)

**Forward**:
```
f[t]  = sigmoid(f_bias + h*fu + x*fw)
h̃[t]  = tanh(h_bias + x*hw + h*hu*f[t])
output[t] = (1 - f[t])*h + f[t]*h̃[t]
```

Structurally identical to GRU but with one fewer gate (reset = update).

---

### `UGRNN_Node` — Update Gate RNN
**Weights** (6 total): `cw, ch, c_bias, gw, gh, g_bias`

**Forward**:
```
c[t] = tanh(c_bias + x*cw + h*ch)
g[t] = sigmoid(g_bias + x*gw + h*gh)
output[t] = g[t]*h + (1 - g[t])*c[t]
```

---

### `Delta_Node`
**Weights** (6 total): `alpha, beta1, beta2, v, r_bias, z_hat_bias`

Offsets applied at runtime for stable initialization: `alpha += 2`, `beta1 += 1`, `beta2 += 1` before computation, restored after.

**Forward** (`d2 = input_values[t]`, `d1 = v * output_values[t-1]`):
```
z̃[t]  = tanh(d1*d2*alpha + d1*beta1 + d2*beta2 + z_hat_bias)
r[t]   = sigmoid(d2 + r_bias)
z[t]   = z̃[t]*(1 - r[t]) + r[t]*output_values[t-1]
output[t] = tanh(z[t])
```

---

### GP Nodes (EXA-GP)

Standard nodes: `SIN_Node`, `COS_Node`, `TANH_Node`, `SIGMOID_Node`, `INVERSE_Node`, `SUM_Node`, `MULTIPLY_Node`

GP variants (`*_GP` suffix): same forward computation, but **edge weight gradients are set to zero** during backward pass. Edges in EXA-GP programs are fixed-weight structural connections; only the node biases and GP-specific parameters are learned.

| Node | Forward |
|------|---------|
| SIN | `output = sin(input)` |
| COS | `output = cos(input)` |
| TANH | `output = tanh(input)` |
| SIGMOID | `output = sigmoid(input)` |
| INVERSE | `output = 1.0 / input` |
| SUM | `output = input` (identity) |
| MULTIPLY | `output = product(all inputs)` — uses `ordered_d_input` for partial derivatives |

**MULTIPLY_NODE gradient handling**: Each incoming edge stores its `input_number[t]` (the slot it wrote to). During backprop, it reads from `ordered_d_input[t][input_number-1]` rather than the shared `d_input[t]`, giving the correct partial derivative `∂output/∂input_i = ∏_{j≠i} input_j`.

---

## Edges

### `RNN_Edge` — Feed-Forward Connection
**Fields**: `weight`, `d_weight`, `outputs[t]`, `deltas[t]`, `dropped_out[t]`, `input_number[t]`

**Forward**:
```
out = input_node->output_values[t] * weight
if dropout training: out = 0 with prob dropout_probability
outputs[t] = out
output_node->input_fired(t, out)
```

**Backward**:
```
delta = output_node->d_input[t]   (or ordered_d_input for MULTIPLY_NODE)
d_weight += delta * input_node->output_values[t]
input_node->output_fired(t, delta * weight)
```

**Dropout at inference**: `out *= (1 - dropout_probability)` (expectation correction).

### `RNN_Recurrent_Edge` — Time-Delayed Connection
Identical fields to `RNN_Edge` plus `recurrent_depth` (integer, 1–10).

**Key difference — temporal offset**:
- Forward: value at time `t` is injected into the output node at time `t + recurrent_depth`
- Backward: error at time `t` flows back to input node at time `t - recurrent_depth`
- First `recurrent_depth` time steps are initialized with 0 input (`first_propagate_forward()`)
- Last `recurrent_depth` time steps are initialized with 0 error (`first_propagate_backward()`)

---

## Weight Storage Order

`get_weights(offset, parameters)` serializes node weights starting at `offset` in the flat vector. Order is fixed per node type and must match `set_weights`:

| Node | Count | Order |
|------|-------|-------|
| RNN_Node | 1 | bias |
| LSTM_Node | 11 | o_gate(upd,w,b), i_gate(upd,w,b), f_gate(upd,w,b), cell(w,b) |
| GRU_Node | 9 | z(w,u,b), r(w,u,b), h(w,u,b) |
| MGU_Node | 6 | f(w,u,b), h(w,u,b) |
| UGRNN_Node | 6 | c(w,h,b), g(w,h,b) |
| Delta_Node | 6 | alpha, beta1, beta2, v, r_bias, z_hat_bias |

In `RNN_Genome`, the full flat vector is: all `RNN_Edge` weights, then all `RNN_Recurrent_Edge` weights, then all node weights in node-vector order.

---

## Initialization Per Node

All initializations write to the same weight fields; all final values are bounded to `[-10, 10]`:

| Strategy | Formula |
|----------|---------|
| Lamarckian | `Normal(mu, sigma)` |
| Xavier | `range * Uniform[-1, 1]`, where `range = sqrt(6) / sqrt(fan_in + fan_out)` |
| Kaiming | `range * Normal(0, 1)`, where `range = sqrt(2) / sqrt(fan_in)` |
| Random | `Uniform[0, 1]` |
| GP | `1.0` for all weights |
