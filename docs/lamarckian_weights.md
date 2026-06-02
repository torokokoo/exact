# Lamarckian Weight Initialization — Inner Workings

## Three Independent Weight Strategies

Every run has three independently configurable strategies (`weights/weight_rules.hxx`):

| Strategy flag | Controls |
|---------------|---------|
| `--weight_initialize` | How weights are set when building the **initial population** (no parent) |
| `--weight_inheritance` | How weights are set when creating a child via **crossover** (has parents) |
| `--mutated_component_weight` | How weights are set on **new structural parts** added by mutation |

Available values for each: `random`, `xavier`, `kaiming`, `lamarckian`, `gp` (EXA-GP-MIN only).

**Defaults** (from `weights/weight_rules.cxx`):
- `weight_initialize = XAVIER`
- `weight_inheritance = LAMARCKIAN`
- `mutated_component_weight = LAMARCKIAN`

**Constraint** (enforced by `EXAMM::check_weight_initialize_validity()`):
- `weight_initialize` cannot be `LAMARCKIAN`
- `weight_inheritance` must equal `weight_initialize` OR be `LAMARCKIAN`
- `mutated_component_weight` must equal `weight_initialize` OR be `LAMARCKIAN`

---

## What "Lamarckian" Means Here

In biological Lamarckism, traits acquired during an organism's lifetime are passed to offspring. Here:

> A parent genome's **trained weights** influence the probability distribution used to initialize weights in its child.

Specifically, the mean (μ) and standard deviation (σ) of the parent's trained weight vector are computed, and all new structural components in the child are initialized by sampling from **N(μ, σ)**.

This is distinct from copying weights directly — it uses the parent's learned weight statistics as a prior for new components.

---

## Where μ and σ Come From

**`RNN_Genome::get_mu_sigma(vector<double>& params, double& mu, double& sigma)`** (`rnn/rnn_genome.cxx`):

```
mu    = mean(params)
sigma = sqrt( mean( (x - mu)^2 for x in params ) )
mu    = clamp(mu,    -11, 11)
sigma = clamp(sigma, 0,   30)
```

`params` is either `best_parameters` (weights after backprop training) or `initial_parameters` (weights before training, as fallback).

The flattened weight vector `params` includes, in order:
1. All `RNN_Edge::weight` values
2. All `RNN_Recurrent_Edge::weight` values
3. All node-internal weights (e.g., GRU: zw, zu, z_bias, rw, ru, r_bias, hw, hu, h_bias)

---

## Weight Storage in Genomes

**Edges** each store a single `double weight`.

**Nodes** store type-specific internal weight vectors:

| Node type | Internal weights |
|-----------|-----------------|
| Simple | bias (1) |
| LSTM | i_w, i_u, i_bias, f_w, f_u, f_bias, o_w, o_u, o_bias, c_w, c_u, c_bias (12) |
| GRU | zw, zu, z_bias, rw, ru, r_bias, hw, hu, h_bias (9) |
| MGU | fw, fu, f_bias, hw, hu, h_bias (6) |
| UGRNN | fw, fu, f_bias, cw, cu, c_bias (6) |
| Delta | … similar pattern |

`get_weights(vector<double>& out)` / `set_weights(vector<double>& in)` serialize all weights into a flat vector in a deterministic order, enabling snapshot/restore of the entire genome state.

---

## Initialization Methods Compared

### Random
```
edge_weight = Uniform[-1, 1]
node_internal = Uniform[-1, 1]
```

### Xavier
```
fan_in  = (input edges) + (recurrent input edges) to node
fan_out = output edges from node
range   = sqrt(6) / sqrt(fan_in + fan_out)

edge_weight     = range * Uniform[-1, 1]
node_internals  = range * Uniform[-1, 1]
```
Keeps activation variance stable across layers.

### Kaiming
```
fan_in = (input edges) + (recurrent input edges) to node
range  = sqrt(2) / sqrt(fan_in)

edge_weight    = range * Normal(0, 1)
node_internals = range * Normal(0, 1)
```
Optimized for ReLU-like activations.

### Lamarckian
```
edge_weight    = bound( Normal(mu, sigma) )
node_internals = bound( Normal(mu, sigma) )  // each sampled independently

bound(x) = clamp(x, -10, 10)
```
Uses parent's trained weight statistics as a prior.

### GP (EXA-GP-MIN only)
```
// Edges from input node i to output node o:
  weight = 1.0  if input and output share the same parameter name
  weight = 0.0  otherwise

// Hidden node internals:
  all weights = 1.0
```
Initializes to identity/pass-through to bootstrap interpretable programs.

---

## Flow 1: Initial Population (no parents)

```
generate_for_initializing_island():
    genome = seed_genome.copy()
    mutate(1, genome)               ← structural mutation to diversify
    genome.initialize_weights()     ← calls weight_initialize strategy
        if XAVIER   → xavier for every node/edge
        if KAIMING  → kaiming for every node/edge
        if RANDOM   → uniform random for every node/edge
        if GP       → GP identity init
    genome.initial_parameters = genome.get_weights()
    return genome
```

---

## Flow 2: Mutation of an Existing Genome

`mutate()` calls structural operations like `add_node`, `add_edge`, `split_edge`, etc. Each operation creates new components and must initialize their weights.

### New node creation (`rnn_genome.cxx::create_node()`)
```
if mutated_component_weight == LAMARCKIAN:
    mu, sigma = get_mu_sigma(genome.best_parameters or initial_parameters)
    node.initialize_lamarckian(generator, normal_dist, mu, sigma)
        → each internal weight = bound( Normal(mu, sigma) )
else:
    genome.initialize_node_randomly(node)
        → uses weight_initialize strategy (xavier / kaiming / random)
```

### New edge creation (`rnn_genome.cxx`)
```
if mutated_component_weight == LAMARCKIAN:
    edge.weight = bound( Normal(mu, sigma) )
else:
    edge.weight = weight_initialize_strategy()
```

**Key point**: pre-existing weights on the copied parent genome are never touched during mutation. Only the weights on *new* structural components are set.

---

## Flow 3: Crossover of Two Parents

`examm.cxx::crossover(p1, p2)`:

### Step 1 — Prepare parent weight statistics
```
p1_weights = p1.best_parameters  (or initial_parameters if not trained yet)
p1_mu, p1_sigma = get_mu_sigma(p1_weights)
p2_weights = p2.best_parameters
p2_mu, p2_sigma = get_mu_sigma(p2_weights)
```

### Step 2 — Decide global weight strategy
```
if weight_inheritance == weight_initialize:
    // All weights will be re-initialized from scratch after structure merge
    re_initialize = true
else:  // weight_inheritance == LAMARCKIAN
    // Weights are inherited; no global re-initialization
    re_initialize = false
```

### Step 3 — Structural merge (always happens regardless of weight strategy)

Edges are aligned by **innovation number**:

```
for each aligned pair (same innovation):
    MATCHING edge — blend weights:
        t = Uniform[-0.5, 1.5]
        child_edge.weight = t * (p2_edge.weight - p1_edge.weight) + p1_edge.weight
        (same blending for connected node internals)

for each unmatched p1 edge (disjoint/excess from fitter parent):
    include with probability more_fit_crossover_rate
    child_edge.weight = p1_edge.weight   ← inherit directly

for each unmatched p2 edge (disjoint/excess from less fit parent):
    include with probability less_fit_crossover_rate
    child_edge.weight = p2_edge.weight   ← inherit directly
```

`t` outside [0, 1] allows extrapolation beyond both parent weights.

### Step 4 — New components (edges/nodes with no parent equivalent)
```
if weight_inheritance == LAMARCKIAN:
    new_edge.weight = bound( Normal(p1_mu, p1_sigma) )   ← from fitter parent stats
    new_node.initialize_lamarckian(generator, normal_dist, p1_mu, p1_sigma)
else:
    new_edge.weight = weight_initialize_strategy()
    new_node.initialize_randomly()
```

### Step 5 — Optional global re-initialization
```
if re_initialize:
    child.initialize_randomly()   ← overwrites ALL weights (matching + unmatched)
```
When `weight_inheritance == LAMARCKIAN`, this step is skipped and all inherited weights are preserved.

---

## Flow 4: Genome Copy (clone)

`RNN_Genome::copy()`:
```
deep copy all nodes (node->copy())
deep copy all edges with updated node pointers
copy initial_parameters  (weights at birth)
copy best_parameters     (weights after training)
```
A clone retains the parent's full trained weight state. The `clone` mutation operation in EXAMM uses this directly without further modification, allowing Lamarckian weight flow across generations without structural change.

---

## Summary: What Gets Inherited vs. Re-Initialized

| Situation | Matched edges | Unmatched edges | New structural components |
|-----------|--------------|-----------------|--------------------------|
| Initial population | — | — | `weight_initialize` strategy |
| Mutation (Lamarckian) | unchanged (from parent copy) | unchanged | N(μ_parent, σ_parent) |
| Crossover (Lamarckian) | blended from both parents | direct from source parent | N(μ_p1, σ_p1) |
| Crossover (non-Lamarckian) | all overwritten | all overwritten | `weight_initialize` strategy |

The Lamarckian design ensures:
1. Weights that proved useful (surviving selection and backprop) propagate forward
2. New structural additions are initialized in a weight-space consistent with their neighbors
3. Random or Xavier re-initialization only happens when explicitly chosen for the initial population
