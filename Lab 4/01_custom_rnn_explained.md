# Model 1: Custom Vanilla RNN (Manual Implementation)

## Architecture Overview

```
Input tokens -> Embedding -> [Manual RNN Loop] -> FC Layer -> Output (2 classes)
     (B, L)     (B, L, E)    h_t at each step    (B, 2)
```

- **B** = batch size, **L** = sequence length (max_len), **E** = embed_size (128)

## The RNN Equation

At each time step `t`, the hidden state is updated:

```
h_t = tanh(W_hx * x_t + W_hh * h_{t-1} + b_h)
```

Where:
- `x_t` is the input at time step t (the embedding of the current token)
- `h_{t-1}` is the hidden state from the previous time step
- `W_hx` (embed_size x hidden_size) transforms the input
- `W_hh` (hidden_size x hidden_size) transforms the previous hidden state
- `b_h` is the bias term
- `tanh` squashes the output to [-1, 1]

## Implementation Details

We use **two `nn.Linear` layers** to implement the RNN cell manually:

| Layer | What it does | Shape |
|-------|-------------|-------|
| `i2h` (input-to-hidden) | Computes `W_hx * x_t + b_hx` | (128, 128) |
| `h2h` (hidden-to-hidden) | Computes `W_hh * h_{t-1}` (no bias) | (128, 128) |

The bias is only on `i2h` because having bias on both would be redundant (two biases added together = one bias).

## Forward Pass Step-by-Step

1. **Embed**: Convert token indices to 128-dim vectors
2. **Initialize h_0 = zeros**: Start with a blank hidden state
3. **Loop over time steps**:
   - For each token in the sequence (t = 0, 1, ..., max_len-1):
   - `h_t = tanh(i2h(x_t) + h2h(h_t))`
4. **Classify**: Pass the **final** hidden state `h_T` through the FC layer

This is the **many-to-one** configuration: the entire sequence is compressed into a single vector `h_T`, which we classify as positive or negative.

## Why tanh (not ReLU)?

1. **Bounded output [-1, 1]**: Prevents exploding activations from repeated multiplication with `W_hh` across many time steps
2. **Zero-centered**: ReLU outputs are always >= 0, which forces all weight gradients to have the same sign, causing inefficient "zig-zagging" optimization paths

## Limitations

- **Vanishing gradients**: For long sequences, gradients from early time steps decay exponentially as they backpropagate through many tanh and matrix multiplications
- **Limited memory**: In practice, vanilla RNNs can only "remember" about 10-20 time steps back
- **Slow**: Python for-loop over time steps is much slower than optimized C++/CUDA kernels
