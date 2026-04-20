# Model 3: Custom LSTM (Manual Implementation)

## Why LSTM?

Vanilla RNNs suffer from **vanishing gradients**: during backpropagation through time, gradients are repeatedly multiplied by `W_hh` and `tanh'(z)` at each step. Since `tanh'(z) <= 1` and `W_hh` often has eigenvalues < 1, the gradient signal from early time steps shrinks exponentially.

**LSTM solves this** by introducing a **cell state** with an **additive** update rule, creating a gradient highway similar to ResNet skip connections.

## Architecture Overview

```
Input tokens -> Embedding -> [Manual LSTM Loop] -> FC Layer -> Output
                              Updates h_t AND c_t
```

LSTM maintains **two** state vectors at each time step:
- **Cell state `c_t`** (long-term memory): carries information across many time steps
- **Hidden state `h_t`** (short-term memory): the "output" at each step

## The LSTM Equations

At each time step t, given input `x_t`, previous hidden `h_{t-1}`, and previous cell `c_{t-1}`:

### Three Gates (all use sigmoid -> values in [0, 1])

```
f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)   # Forget gate
i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)   # Input gate
o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)   # Output gate
```

### Candidate Value (uses tanh -> values in [-1, 1])

```
g_t = tanh(W_g * [h_{t-1}, x_t] + b_g)      # New information to potentially add
```

### State Updates

```
c_t = f_t * c_{t-1} + i_t * g_t              # Cell state (ADDITIVE update!)
h_t = o_t * tanh(c_t)                        # Hidden state
```

## Gate Intuition

| Gate | Sigmoid Output | Purpose | Analogy |
|------|---------------|---------|---------|
| **Forget (f_t)** | 0 = forget, 1 = keep | Erase irrelevant old memories | Erasing a whiteboard |
| **Input (i_t)** | 0 = ignore, 1 = write | Select which new info to store | Choosing what to write |
| **Output (o_t)** | 0 = hide, 1 = expose | Control what to reveal from memory | Choosing what to say |
| **Candidate (g_t)** | [-1, 1] range | The actual new information | The content to write |

## Why This Solves Vanishing Gradients

The key is the **cell state update**:

```
c_t = f_t * c_{t-1} + i_t * g_t
```

During backpropagation, the gradient of `c_t` w.r.t. `c_{t-1}` is simply `f_t` (element-wise). If `f_t ≈ 1` (the network decides to remember), the gradient flows backward **almost unchanged** through the `+` operation. This is exactly like a **ResNet skip connection**!

Compare with vanilla RNN where the gradient must pass through `tanh` AND a matrix multiplication at every step.

## Implementation Details

Each gate requires two `nn.Linear` layers (input-to-hidden and hidden-to-hidden), giving us **8 linear layers** total:

| Gate | i2h Layer | h2h Layer |
|------|-----------|-----------|
| Forget | `forget_i2h` | `forget_h2h` |
| Input | `input_i2h` | `input_h2h` |
| Candidate | `candidate_i2h` | `candidate_h2h` |
| Output | `output_i2h` | `output_h2h` |

This is 4x more parameters than vanilla RNN, but the improved gradient flow makes it far more effective on longer sequences.

## Parameter Count Comparison

For hidden_size=128, embed_size=128:

| Model | Trainable Parameters (RNN cell only) |
|-------|--------------------------------------|
| Vanilla RNN | `128*128 + 128*128 + 128 = 32,896` |
| LSTM | `4 * (128*128 + 128*128 + 128) = 131,584` |

LSTM has ~4x more parameters because it has 4 gate/candidate computations instead of 1.
