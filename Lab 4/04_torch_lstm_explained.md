# Model 4: PyTorch Built-in LSTM (nn.LSTM)

## Architecture Overview

```
Input tokens -> Embedding -> nn.LSTM -> FC Layer -> Output (2 classes)
     (B, L)     (B, L, E)   (B, L, H)   (B, 2)
```

## How nn.LSTM Differs from Custom LSTM

`nn.LSTM` implements the same LSTM equations but **fuses all 4 gate computations** into a single matrix multiplication:

```
[i_t, f_t, g_t, o_t] = W * [h_{t-1}, x_t] + b
```

Instead of our 8 separate `nn.Linear` calls, `nn.LSTM` does ONE large matrix multiply and then splits the result into 4 parts, applying sigmoid/tanh accordingly.

| Aspect | CustomLSTM (manual) | TorchLSTM (nn.LSTM) |
|--------|--------------------|--------------------|
| **Gate computation** | 8 separate nn.Linear calls | 1 fused matrix multiply |
| **Speed** | Slower (Python loop + many small ops) | Much faster (cuDNN optimized) |
| **Memory** | Higher (intermediate tensors per gate) | Lower (fused computation) |
| **States shape** | `h_t`: (B, H), `c_t`: (B, H) | `h_n`: (layers, B, H), `c_n`: (layers, B, H) |

## Key nn.LSTM Parameters

```python
nn.LSTM(
    input_size=128,     # Dimension of each input token (embed_size)
    hidden_size=128,    # Dimension of h_t and c_t
    num_layers=1,       # Single LSTM layer
    batch_first=True    # Input shape: (batch, seq_len, features)
)
```

## nn.LSTM Returns

```python
output, (h_n, c_n) = self.lstm(embedded, (h_0, c_0))
```

- `output`: All hidden states `h_t` — shape `(batch, seq_len, hidden_size)`
- `h_n`: Final hidden state — shape `(num_layers, batch, hidden_size)`
- `c_n`: Final cell state — shape `(num_layers, batch, hidden_size)`

Note: nn.LSTM takes and returns **both** `h` and `c` as a tuple, unlike nn.RNN which only has `h`.

## Expected Performance

LSTM models (both custom and built-in) should **outperform** vanilla RNN models, especially when `max_len` is larger, because:

1. **Better long-term memory**: The cell state highway preserves information from early tokens
2. **Stable gradients**: Additive cell update avoids vanishing gradient
3. **Selective memory**: Gates learn what to remember and what to forget

## The max_len Experiment

The lab asks you to experiment with different `max_len` values. Expected behavior:

| max_len | Vanilla RNN | LSTM |
|---------|-------------|------|
| Small (50) | OK — short sequences are manageable | OK — but can't use full context |
| Medium (100) | Starts struggling — early tokens fade | Good — cell state preserves context |
| Large (200+) | Poor — vanishing gradients dominate | Best — can leverage full review |

This demonstrates the LSTM's key advantage: it can effectively use **longer contexts** because the cell state provides a gradient highway that vanilla RNNs lack.

## Summary of All 4 Models

| Model | Implementation | Long-term Memory | Speed |
|-------|---------------|-----------------|-------|
| Custom RNN | Manual (nn.Linear) | Poor (vanishing gradients) | Slowest |
| Torch RNN | nn.RNN | Poor (vanishing gradients) | Fast |
| Custom LSTM | Manual (nn.Linear) | Good (cell state highway) | Slow |
| Torch LSTM | nn.LSTM | Good (cell state highway) | Fastest |
