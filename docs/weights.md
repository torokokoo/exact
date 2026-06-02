# Weight Optimizers — Inner Workings

## Overview

Two separate systems:
- **`WeightRules`** — controls *how weights are initialized* at genome birth and during mutation/crossover
- **`WeightUpdate`** — controls *how weights are updated* during backpropagation

---

## `WeightRules` (`weights/weight_rules.hxx`)

### Enum
```cpp
enum WeightType { RANDOM=0, XAVIER=1, KAIMING=2, LAMARCKIAN=3, GP=4, NONE=-1 };
```

### Three independent strategies
| Strategy | CLI flag | Default |
|----------|----------|---------|
| `weight_initialize` | `--weight_initialize` | `xavier` |
| `weight_inheritance` | `--weight_inheritance` | `lamarckian` |
| `mutated_components_weight` | `--mutated_component_weight` | `lamarckian` |

See `docs/lamarckian_weights.md` for full detail on how these interact.

### Validation rule (enforced at startup)
- `weight_initialize` cannot be `LAMARCKIAN`
- `weight_inheritance` must be `LAMARCKIAN` or equal `weight_initialize`
- `mutated_components_weight` must be `LAMARCKIAN` or equal `weight_initialize`

---

## `WeightUpdate` (`weights/weight_update.hxx`)

Selected via `--weight_update <method>`. All methods:
1. Apply gradient norm thresholding before the update
2. Clip all weights to `[-10, 10]` after the update

### Gradient norm thresholding (applied to every gradient vector before update)
```
L2 = sqrt(sum(g[i]^2))

if L2 > high_threshold (default 1.0):
    g[i] *= high_threshold / L2         ← scale down (prevent exploding gradients)

if L2 < low_threshold (default 0.05) and L2 > 0:
    g[i] *= low_threshold / L2          ← scale up (prevent vanishing gradients)
```

CLI: `--high_threshold`, `--low_threshold`

---

## Update Rules

### `vanilla`
```
w[i] -= lr * g[i]
```
State: none.

---

### `momentum`
```
v[i] = mu * v[i] - lr * g[i]
w[i] += v[i]
```
State: `velocity` vector (same size as weight vector).
CLI: `--mu` (default 0.9), `--learning_rate` (default 0.001)

---

### `nesterov`
```
pv[i] = v[i]
v[i]  = mu * v[i] - lr * g[i]
w[i] += -mu * pv[i] + (1 + mu) * v[i]
```
State: `velocity`, `prev_velocity`.
CLI: `--mu` (default 0.9), `--learning_rate`

---

### `adagrad`
```
cache[i] += g[i]^2
w[i]     -= lr * g[i] / (sqrt(cache[i]) + eps)
```
State: `velocity` (accumulated squared gradients, never decays).
CLI: `--eps` (default 1e-8), `--learning_rate`

---

### `rmsprop`
```
cache[i] = decay * cache[i] + (1 - decay) * g[i]^2
w[i]    -= lr * g[i] / (sqrt(cache[i]) + eps)
```
State: `velocity` (exponential moving average of squared gradients).
CLI: `--decay_rate` (default 0.9), `--eps` (default 1e-8), `--learning_rate`

---

### `adam`
```
m[i] = beta1 * m[i] + (1 - beta1) * g[i]          ← first moment
v[i] = beta2 * v[i] + (1 - beta2) * g[i]^2        ← second moment
w[i] -= lr * m[i] / (sqrt(v[i]) + eps)
```
No bias correction.
State: `prev_velocity` (m), `velocity` (v).
CLI: `--beta1` (default 0.9), `--beta2` (default 0.99), `--eps` (default 1e-8), `--learning_rate` (default 0.001)

---

### `adam-bias`
```
m[i] = beta1 * m[i] + (1 - beta1) * g[i]
mt   = m[i] / (1 - beta1^epoch)                    ← bias-corrected
v[i] = beta2 * v[i] + (1 - beta2) * g[i]^2
vt   = v[i] / (1 - beta2^epoch)                    ← bias-corrected
w[i] -= lr * mt / (sqrt(vt) + eps)
```
State: same as `adam` plus epoch counter.
CLI: same as `adam`

---

## State Initialization

All state vectors (`velocity`, `prev_velocity`) are initialized to zero. They are reset to zero at the start of each new genome's backpropagation call — state is **not** carried across genomes. The `WeightUpdate` object is shared across all threads and genome evaluations; the state vectors are reallocated per genome in each call to `update()`.

---

## How Optimizers Are Created

```cpp
WeightUpdate* wu = WeightUpdate::generate_from_arguments(arguments);
```

This factory reads `--weight_update` and the relevant hyperparameter flags, then instantiates the appropriate subclass.

The same `WeightUpdate` object is passed into `genome->backpropagate_stochastic(...)` and used for every weight update step inside the training loop.
