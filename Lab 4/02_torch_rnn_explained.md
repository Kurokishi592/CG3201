# Model 2: PyTorch Built-in RNN (nn.RNN)

## Architecture Overview

```
Input tokens -> Embedding -> nn.RNN -> FC Layer -> Output (2 classes)
     (B, L)     (B, L, E)   (B, L, H)  (B, 2)
```

## How nn.RNN Differs from Our Manual Implementation

`nn.RNN` implements the **exact same equation** as our CustomRNN:

```
h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
```

The difference is **how** it's implemented:

| Aspect | CustomRNN (manual) | TorchRNN (nn.RNN) |
|--------|-------------------|-------------------|
| **Computation** | Python for-loop over time steps | Optimized C++/CUDA kernel |
| **Speed** | Slower (Python overhead per step) | Faster (fused operations, cuDNN) |
| **GPU utilization** | Lower (sequential Python calls) | Higher (batched matrix ops) |
| **Flexibility** | Full control over internals | Less customizable |
| **Hidden state init** | Shape: `(B, H)` | Shape: `(num_layers, B, H)` |

## Key nn.RNN Parameters

```python
nn.RNN(
    input_size=128,     # Dimension of each input token (embed_size)
    hidden_size=128,    # Dimension of the hidden state
    num_layers=1,       # Single hidden layer (can stack more)
    batch_first=True,   # Input shape: (batch, seq_len, features)
    nonlinearity='tanh' # Activation function (default)
)
```

## nn.RNN Returns

```python
output, h_n = self.rnn(embedded, h_0)
```

- `output`: Hidden states at ALL time steps — shape `(batch, seq_len, hidden_size)`
- `h_n`: Hidden state at the LAST time step only — shape `(num_layers, batch, hidden_size)`

For many-to-one classification, we only need `h_n` (the final summary vector).

## Performance Comparison with CustomRNN

Both models should produce **similar accuracy** since they implement the same math. The differences are:

1. **Training speed**: nn.RNN is typically 2-5x faster due to CUDA optimization
2. **Numerical precision**: Minor floating-point differences from different computation order
3. **Memory**: nn.RNN may be more memory-efficient due to fused operations
4. **Stability**: Both suffer equally from vanishing gradients (same architecture)

## Same Limitation

Like the custom version, `nn.RNN` still suffers from **vanishing gradients** on long sequences. This is a fundamental limitation of the vanilla RNN architecture, not the implementation.
