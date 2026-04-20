# Lab 4 / Lecture 7 Quiz Notes

A running collection of quiz questions, answers, and explanations for CG3201 Lab 4 (RNN/LSTM) and Lecture 7.

---

## Question 1: Default Hidden State Initialization in PyTorch

**Q:** In your PyTorch RNN/LSTM implementation using `nn.RNN` and `nn.LSTM`, what happens to the hidden state if you do not explicitly initialize it before passing a new batch of sequences?

- PyTorch initializes the hidden state with random values from a standard Gaussian distribution.
- ✅ **PyTorch defaults the initial hidden state to a tensor of all zeros.**
- PyTorch reuses the final hidden state from the previous batch.
- PyTorch will throw a RuntimeError.

### Explanation

When you call `nn.RNN` or `nn.LSTM` without passing an initial hidden state, PyTorch automatically creates one filled with zeros. From the PyTorch docs:

> If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.

These two calls are equivalent:
```python
# Explicit (what we did in the lab)
h_0 = torch.zeros(1, batch_size, hidden_size).to(device)
output, h_n = self.rnn(embedded, h_0)

# Implicit (PyTorch fills in zeros automatically)
output, h_n = self.rnn(embedded)
```

### Why the other options are wrong

- ❌ **Random Gaussian values**: PyTorch never does this — it would make training non-deterministic and unstable
- ❌ **Reuses the final hidden state from the previous batch**: Would be a bug for independent batches. Only desirable for **stateful RNNs** (e.g., TBPTT), and even then must be done manually
- ❌ **RuntimeError**: The hidden state is an *optional* argument

### Why zero-init is the sensible default

- Assumes "no prior context" — appropriate when each batch is independent (e.g., IMDB sentiment, where each review is a separate sequence)
- Deterministic → reproducible training
- Matches the mathematical derivation of `h_0` in RNN equations

### When you'd want non-zero init

- **Stateful RNN / TBPTT**: Pass the previous batch's `h_n` as the next batch's `h_0` (with `.detach()` to cut the computation graph)
- **Encoder-decoder models**: Initialize the decoder's `h_0` with the encoder's final hidden state (e.g., machine translation)
- **Learned initial state**: Make `h_0` a learnable `nn.Parameter`

---

## Question 2: FC Layer Input Dimension After LSTM

**Q:** Consider a single-layer LSTM with hidden size 128, followed by a fully connected layer. What is the input dimension of the fully connected layer?

- 64
- ✅ **128**
- 512
- 256

### Explanation

The fully connected layer takes the LSTM's **hidden state** `h_t` as input. The dimension of `h_t` is exactly the LSTM's `hidden_size`, which is **128**.

```python
self.lstm = nn.LSTM(input_size=embed_size, hidden_size=128, ...)
self.fc = nn.Linear(128, output_size)   # input dim = hidden_size
                                  ↑
                            matches hidden_size
```

This is exactly what we did in the lab:
```python
output, (h_n, c_n) = self.lstm(embedded, (h_0, c_0))
# h_n shape: (1, batch_size, 128)  ← 128 = hidden_size
output = self.fc(h_n.squeeze(0))    # FC input dim = 128
```

### Why the other options are wrong

- ❌ **64**: Half of hidden_size — only correct for a **bidirectional** LSTM with `hidden_size=64` (then forward + backward = 128)
- ❌ **512**: This is `4 × 128` — the size of the **fused gate computation** inside `nn.LSTM` (4 gates: i, f, g, o). But that's internal; the *output* `h_t` is still just 128.
- ❌ **256**: Correct for a **bidirectional** LSTM (forward 128 + backward 128 = 256), but the question specifies plain single-layer

### Key Concept

The LSTM has many internal weight matrices, but its **output** at each time step is just one vector of size `hidden_size`. The cell state `c_t` is also `hidden_size`-dimensional but stays internal — only `h_t` is exposed to downstream layers.

**Rule of thumb:** FC input dim = LSTM `hidden_size` (or `2 × hidden_size` if bidirectional).

---

## Question 3: Number of W_hh Matrices in an RNN

**Q:** A single-layer RNN with hidden size 128 processes a sequence of length 256. How many distinct recurrent weight matrices `W_hh` are learned?

- 128
- ✅ **1**
- 2
- 256

### Explanation

A vanilla RNN uses **the same recurrent weight matrix `W_hh`** at every single time step. This is the core idea of **weight sharing** (parameter tying) in RNNs.

```
h_1 = tanh(W_hh * h_0   + W_hx * x_1   + b_h)
h_2 = tanh(W_hh * h_1   + W_hx * x_2   + b_h)   ← same W_hh
h_3 = tanh(W_hh * h_2   + W_hx * x_3   + b_h)   ← same W_hh
...
h_256 = tanh(W_hh * h_255 + W_hx * x_256 + b_h) ← same W_hh
```

Even though the RNN is "unrolled" across 256 time steps in diagrams, all those copies share **one** physical `W_hh` matrix in memory. There is exactly **1** `W_hh` per RNN layer, regardless of sequence length.

### Why the other options are wrong

- ❌ **128**: This is the hidden size. `W_hh` itself is a `128 × 128` matrix, but it's still **one** matrix.
- ❌ **2**: This counts *all* weight matrices in an RNN cell — there are two: `W_hh` and `W_hx`. But the question specifically asks about `W_hh`.
- ❌ **256**: Common misconception — students sometimes think each time step has its own weights. If true, the model couldn't generalize to other sequence lengths and parameters would explode.

### Why weight sharing matters

1. **Generalization across sequence lengths**: A model trained on 100-token reviews can process 50- or 200-token reviews — same `W_hh` reused
2. **Constant parameter count**: Independent of sequence length
3. **Translation invariance in time**: RNN learns *patterns*, not *position-specific rules*
4. **Enables BPTT**: Gradients for the *same* `W_hh` are accumulated across all time steps: `∂L/∂W_hh = Σ_t ∂L_t/∂W_hh`

### Parameter Count

For single-layer RNN with `hidden_size=128`, `input_size=128`:
- `W_hx`: 128 × 128 = 16,384
- `W_hh`: 128 × 128 = 16,384  ← **just one** of these
- `b_h`: 128
- **Total**: 32,896 parameters (independent of sequence length 256)

---

## Question 4: One-Hot Encoding vs Embedding Layer

**Q:** Which of the following statements is FALSE when one-hot encoded word vectors are fed directly into an RNN instead of using an embedding layer?

- The model cannot capture inherent semantic similarities between words directly from the input representation.
- The input dimensionality becomes equal to the vocabulary size.
- The GPU memory required for gradient storage increases.
- ✅ **The computational efficiency increases due to the sparsity of the input.** (FALSE)

### Explanation

Feeding one-hot vectors directly into an RNN actually **decreases** computational efficiency. Even though one-hot vectors are technically sparse (only one element is 1), standard PyTorch/CUDA matrix multiplication treats them as **dense** vectors and performs the full `vocab_size × hidden_size` matrix multiply at every time step. The sparsity is wasted unless you use specialized sparse operations.

In fact, an embedding layer is **the optimization** that exploits one-hot sparsity: instead of computing `W * one_hot(token)`, it just **indexes** row `token` of `W`, which is O(1) per token.

### Why the other three statements are TRUE

- ✅ **"Cannot capture semantic similarities"**: One-hot vectors are mathematically *equidistant*. The dot product of any two distinct one-hots is 0, so "cat" and "dog" are just as different as "cat" and "spaceship". Embeddings learn a continuous space where similar words cluster together.

- ✅ **"Input dimensionality equals vocab size"**: A one-hot vector for vocab size V has length V. With our IMDB vocab of ~5000, every token would be a 5000-dim vector. `W_hx` would be `5000 × 128` instead of `128 × 128`.

- ✅ **"GPU memory increases"**: Larger input dim → larger `W_hx` → more gradient tensors. Input tensors themselves are much bigger (e.g., batch 128 × 100 tokens × 5000-dim ≈ 64M floats vs ~1.6M floats for dense embeddings).

### Why embeddings are the right approach

An embedding layer is mathematically equivalent to:
```
embedding(token) = W_e * one_hot(token)
```
But implemented as a **lookup table**: grab row `token` from `W_e`. This:
1. Avoids the wasteful matrix multiply
2. Produces a dense, low-dimensional vector (128 instead of 5000)
3. Learns semantically meaningful representations during training

### Key Concept (Lecture slide 27)

| Property | One-hot | Embedding |
|---|---|---|
| Dimensionality | Sparse, high-dimensional (= vocab size) | Dense, low-dimensional |
| Semantics | Equidistant — no meaning structure | Similar words are close in vector space |
| Memory | Large input tensors, large `W_hx` | Compact |
| Compute | Wasted multiplications | O(1) lookup |

---

## Question 5: Replacing Forget Gate Sigmoid with ReLU

**Q:** In your LSTM implementation using PyTorch, the default activation function of the forget gate is sigmoid. What is the most likely consequence of switching this activation function to ReLU?

- The forget gate will function similarly to sigmoid because ReLU maps negative inputs to zero.
- The vanishing gradient problem will become more severe.
- The model will converge faster.
- ✅ **The cell state may explode towards infinity for long sequences.**

### Explanation

The forget gate scales the previous cell state via element-wise multiplication:
```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
```

For this to work as a "gate" (smooth on/off switch), `f_t` must be in **[0, 1]**:
- `f_t = 0` → completely forget that dimension
- `f_t = 1` → completely keep it
- `f_t = 0.5` → keep half

Sigmoid maps any real number into [0, 1], which is exactly what's needed.

### What happens with ReLU?

ReLU outputs values in **[0, +∞)** — no upper bound:
- Negative pre-activation → `f_t = 0` (forget — same as sigmoid would do here)
- Pre-activation of 5 → `f_t = 5`, multiplying cell state by 5 at this step
- If `f_t > 1` consistently, cell state grows **exponentially**: `c_t ≈ f_t^T × c_0`

After 100 time steps with average `f_t = 2`: `2^100 ≈ 10^30`. Cell state explodes, loss becomes NaN, training collapses.

### Why the other options are wrong

- ❌ **"Functions similarly because ReLU maps negatives to zero"**: Half-true but misleading. Both clip negatives, but ReLU's *positive* side is unbounded — that's the whole problem
- ❌ **"Vanishing gradient becomes more severe"**: Opposite — ReLU is a standard *fix* for vanishing gradients in feedforward nets. The problem here is **exploding** activations
- ❌ **"Model converges faster"**: Model wouldn't converge at all — it would diverge as `f_t > 1` blows up the cell state

### Key Design Principle: Gates use sigmoid, content uses tanh

All three LSTM gates (forget, input, output) use **sigmoid** because they need [0, 1] gating values. The candidate `g_t` uses **tanh** because it represents *content* (which can be positive or negative), not gating.

Replacing any sigmoid with ReLU breaks LSTM's mathematical guarantees:

| Component | Activation | Why |
|---|---|---|
| Forget gate `f_t` | sigmoid | Must be in [0,1] to scale memory |
| Input gate `i_t` | sigmoid | Must be in [0,1] to scale candidate |
| Output gate `o_t` | sigmoid | Must be in [0,1] to mask output |
| Candidate `g_t` | tanh | Content, bounded in [-1,1] |

### Connection to the lab

```python
f_t = self.sigmoid(self.forget_i2h(x_t) + self.forget_h2h(h_t))
i_t = self.sigmoid(self.input_i2h(x_t) + self.input_h2h(h_t))
o_t = self.sigmoid(self.output_i2h(x_t) + self.output_h2h(h_t))
g_t = self.tanh(self.candidate_i2h(x_t) + self.candidate_h2h(h_t))
```
Three sigmoids (gates) + one tanh (candidate). Swap any sigmoid for ReLU → cell state diverges.

---

## Question 6: Shape of LSTM Fused Weight Matrix

**Q:** Given an LSTM with embedding size `x` and `h` hidden units, what is the shape of the weight matrix that transforms `[h_{t-1}, x_t]` before it is split into the four internal gates?

- ✅ **(4 * h, h + x)**
- (4, h * x)
- (h + x, 4 * h)
- (h, x)

### Explanation

In PyTorch's `nn.LSTM`, the four gates are computed in a single fused operation:
```
[i_t; f_t; g_t; o_t] = W * [h_{t-1}; x_t] + b
```

Where:
- `[h_{t-1}; x_t]` is the concatenation of previous hidden state and current input → shape **(h + x,)**
- `[i_t; f_t; g_t; o_t]` is the concatenation of all four gate pre-activations → shape **(4h,)**

For `W * v` to map `(h+x,)` → `(4h,)`, `W` must have shape **(4h, h + x)**, following PyTorch's standard `(out_features, in_features)` convention.

### Why fused?

Four separate gate matmuls would each be `(h, h+x) @ (h+x,) = (h,)`. Stacking them into one `(4h, h+x)` matrix lets you do **one** large matmul instead of four small ones. GPUs love large matmuls — better memory access, fewer kernel launches.

```python
gates = W @ torch.cat([h, x]) + b      # one matmul: (4h,)
i, f, g, o = gates.chunk(4, dim=0)     # split: 4 × (h,)
i = sigmoid(i); f = sigmoid(f); o = sigmoid(o); g = tanh(g)
```

This is why `nn.LSTM` is much faster than our `CustomLSTM` (which uses 8 separate `nn.Linear` calls instead of 1 fused matmul).

### Why the other options are wrong

- ❌ **(4, h * x)**: Dimensionally meaningless
- ❌ **(h + x, 4 * h)**: **Transpose** of the correct answer. PyTorch uses `(out, in)` convention
- ❌ **(h, x)**: Would be `W_hx` in a vanilla RNN with **one** gate and **no** hidden recurrence — missing the 4x factor and the hidden-state contribution

### PyTorch attribute names

```python
lstm = nn.LSTM(input_size=x, hidden_size=h)
lstm.weight_ih_l0.shape  # (4h, x)   ← input-to-hidden, all 4 gates stacked
lstm.weight_hh_l0.shape  # (4h, h)   ← hidden-to-hidden, all 4 gates stacked
```

PyTorch actually splits the conceptual `(4h, h+x)` matrix into two separate matrices (`weight_ih` and `weight_hh`), but mathematically it's equivalent: `W @ [h; x]` = `W_hh @ h + W_ih @ x`.

### Parameter Count

For `x = h = 128`:
- Weight: `4 × 128 × (128 + 128) = 131,072` parameters
- Bias: `4 × 128 = 512` (or `1024` in PyTorch with two bias terms)
- **Total ≈ 131,584** — exactly **4x a vanilla RNN** with the same dimensions, because LSTM has 4 gates instead of 1

---

## Question 7: Replacing tanh with ReLU in Vanilla RNN

**Q:** When implementing the many-to-one vanilla RNN in PyTorch, the default activation function for hidden states is tanh. What is the most likely consequence of switching this activation function to ReLU for long sequences?

- The model will converge faster due to restricted output range.
- ✅ **The model will become highly susceptible to exploding gradients.**
- The gradient vanishes more rapidly as it travels back to early time steps.
- It helps the model avoid the "zigzagging" update problem.

### Explanation

In a vanilla RNN: `h_t = activation(W_hh * h_{t-1} + W_hx * x_t + b_h)`

The activation function is critical for keeping the hidden state **bounded** as it's repeatedly updated.

### Why tanh keeps things stable

`tanh(z) ∈ [-1, 1]` regardless of input. So no matter how big `W_hh * h_{t-1}` becomes, `h_t` is always in [-1, 1]. The activation acts as a **safety valve**.

### Why ReLU causes explosions

`ReLU(z) ∈ [0, +∞)` — unbounded. Since the same `W_hh` is reused at every step:
```
h_1 ≈ W_hh * h_0
h_2 ≈ W_hh^2 * h_0
...
h_T ≈ W_hh^T * h_0
```

If `W_hh`'s largest eigenvalue **λ > 1**, then `||h_T|| ~ λ^T` — explodes exponentially. Same for gradients during BPTT:
```
∂h_T/∂h_0 ≈ W_hh^T   (when ReLU is in its active region, derivative = 1)
```
λ > 1 → gradients explode → NaN losses → training collapses.

### Why the other options are wrong

- ❌ **"Converge faster due to restricted range"**: ReLU's range is **unrestricted** on the positive side. Tanh has the restricted range, not ReLU.
- ❌ **"Gradient vanishes more rapidly"**: Opposite — ReLU's derivative is **1** in its active region, so gradients pass through unchanged. The problem here is **explosion**, not vanishing.
- ❌ **"Avoids zig-zagging"**: **Backwards.** Lecture slide 14 shows zig-zagging is *caused* by ReLU (all-positive outputs force same-sign weight gradients). Tanh's zero-centered output is what avoids zig-zagging.

### Lecture Slide 13 — Why tanh over ReLU

1. **Preventing exploding activations** from repeated multiplication with `W_hh` across the sequence
2. **Preventing zig-zagging** paths toward optimal weights during training

ReLU breaks **both**.

### Summary

| Activation | Forward stability | Gradient flow | Zig-zag |
|---|---|---|---|
| **tanh** | ✅ Bounded [-1, 1] | Can vanish (`tanh' ≤ 1`) | ✅ Zero-centered |
| **ReLU** | ❌ Unbounded → explodes | Doesn't vanish in active region | ❌ All-positive → zig-zags |

For RNNs, **forward stability is critical** because the same weights are reused over many steps. Tanh wins.

---

## Question 8: Effect of Increasing Hidden Size from 128 to 1024

**Q:** What is the impact of increasing the hidden size from 128 to 1024 neurons, assuming all other hyperparameters remain the same?

- The vanishing gradient problem is resolved due to the larger memory capacity.
- The RNN can now process a maximum sequence length of 1024 tokens instead of 128.
- ✅ **The model's representational capacity increases, but the number of parameters grows quadratically.**
- The input dimension must also be increased to 1024 to match the new hidden size.

### Explanation

Hidden size `h` controls the dimensionality of `h_t`. Increasing from 128 → 1024 boosts representational capacity but parameters scale quadratically due to `W_hh`.

### Why parameters grow quadratically

For a vanilla RNN with embed size `x` and hidden size `h`:

| Matrix | Shape | # params |
|---|---|---|
| `W_hx` | `(h, x)` | `h × x` (linear in h) |
| `W_hh` | `(h, h)` | **`h²`** (quadratic in h) |
| `b_h` | `(h,)` | `h` |

| Hidden size | `W_hh` params |
|---|---|
| 128 | 16,384 |
| 1024 | **1,048,576** |

8x increase in hidden size → **64x increase** in `W_hh` parameters. `(8)² = 64`. Quadratic confirmed.

### Why the other options are wrong

- ❌ **"Vanishing gradient resolved"**: Vanishing gradients are caused by repeated multiplication by `W_hh` and `tanh' ≤ 1` over many steps. Depends on **eigenvalues** of `W_hh`, not size. Solved by **LSTM/GRU**, not bigger `h`.
- ❌ **"Can now process max 1024 tokens"**: Hidden size has **nothing to do** with sequence length. RNNs handle any length thanks to weight sharing — same `W_hh` reused at every step.
- ❌ **"Input dim must also be 1024"**: `x` and `h` are **independent**. `W_hx` is `(h, x)` and maps any `x`-dim input to any `h`-dim hidden state.

### Trade-offs of larger hidden size

**Pros:**
- More representational capacity
- Larger "memory" per step

**Cons:**
- Quadratic parameter growth → much more memory, slower training
- Higher overfitting risk (especially with small datasets like IMDB's 30k)
- 64x more FLOPs per matmul (1024² vs 128²)
- Diminishing returns

### Key Concept

**Hidden size controls capacity, not memory length.** Bigger `h` means richer per-step features but doesn't help the network remember things further back. Long-term memory requires architectural changes (LSTM/GRU/attention), not just bigger `h`.

---

## Question 9: Output Dimension of a 2-Layer LSTM at Each Time Step

**Q:** A 2-layer LSTM with hidden size 128 processes a sequence of length 512. What is the output dimension of the second layer at each time step?

- ✅ **128**
- 1
- 65536
- 512

### Explanation

At each time step, every LSTM layer outputs an `h_t` vector of size = `hidden_size`. With `hidden_size = 128`, the second layer's output at each time step is a **128-dim vector**, regardless of how many layers are stacked or how long the sequence is.

### How a stacked LSTM works

```
Time:        t=1     t=2     t=3    ...    t=512

Layer 2:    h²_1 → h²_2 → h²_3 → ... → h²_512   (each 128-dim)
              ↑      ↑      ↑              ↑
Layer 1:    h¹_1 → h¹_2 → h¹_3 → ... → h¹_512   (each 128-dim)
              ↑      ↑      ↑              ↑
Input:       x_1    x_2    x_3            x_512
```

- Layer 1: takes embedded input `x_t`, outputs `h¹_t` (128-dim)
- Layer 2: takes Layer 1's `h¹_t` as input, outputs `h²_t` (128-dim)
- Final output is `h²_t` (top layer)

```python
nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True)
# output shape: (batch, 512, 128)
```

### Why the other options are wrong

- ❌ **1**: That's a classification layer's output, not the LSTM's
- ❌ **65536**: `512 × 128` — total flattened output across all time steps. Question asks per time step
- ❌ **512**: Sequence length, not a vector dimension

### Key Concept: Stacking adds depth, not width

- **hidden_size** (width) stays the same across layers
- **num_layers** (depth) controls how many stacked LSTM cells
- Analogous to stacking conv layers in a CNN

### Output shapes for 2-layer LSTM with input `(B, 512, 128)`

| Tensor | Shape | What it is |
|---|---|---|
| `output` | `(B, 512, 128)` | Top layer's `h_t` at every time step |
| `h_n` | `(2, B, 128)` | Final hidden state per layer |
| `c_n` | `(2, B, 128)` | Final cell state per layer |

Note: `output[:, -1, :] == h_n[-1]` — both 128-dim, confirming the answer.

---

## Question 10: LSTM vs Vanilla RNN Parameter Count

**Q:** Compared to a vanilla RNN with the same hidden size, an LSTM layer has:

- Parameters depending on sequence length.
- ✅ **More parameters due to multiple gates.**
- Fewer parameters.
- The same number of parameters.

### Explanation

An LSTM has **exactly 4x** the parameters of a vanilla RNN with the same hidden size, because it computes **four** linear transformations per time step (one for each gate) instead of just one.

### Parameter count comparison

Let `x` = input/embedding size, `h` = hidden size:

**Vanilla RNN** — one transformation:
```
h_t = tanh(W_hh * h_{t-1} + W_hx * x_t + b_h)
```
Total: **`h(x + h + 1)`**

**LSTM** — four transformations (forget, input, output, candidate gates):
```
i_t = σ(W_ix * x_t + W_ih * h_{t-1} + b_i)
f_t = σ(W_fx * x_t + W_fh * h_{t-1} + b_f)
o_t = σ(W_ox * x_t + W_oh * h_{t-1} + b_o)
g_t = tanh(W_gx * x_t + W_gh * h_{t-1} + b_g)
```
Total: **`4 × h(x + h + 1)`**

### Concrete example (lab: x = h = 128)

| Model | Calculation | Parameters |
|---|---|---|
| Vanilla RNN | `128 × (128 + 128 + 1)` | **32,896** |
| LSTM | `4 × 128 × (128 + 128 + 1)` | **131,584** |

Exactly 4:1 ratio.

### Why the other options are wrong

- ❌ **"Depending on sequence length"**: Both RNNs and LSTMs have **constant** parameter count regardless of sequence length thanks to **weight sharing** — same weights reused at every step
- ❌ **"Fewer parameters"**: Impossible — LSTM strictly adds gates on top of RNN computation
- ❌ **"Same number of parameters"**: Would be true with one gate, but LSTM has four

### What you get for the 4x cost

1. **Vanishing gradient resistance** (additive cell state highway)
2. **Selective memory** (gates learn what to remember/write/output)
3. **Better long-term dependency modeling**

In our lab: LSTM jumped from 80% (vanilla RNN) → 86% test accuracy on the same data.

### Comparison Table

| Architecture | # transformations | Params vs RNN |
|---|---|---|
| Vanilla RNN | 1 | 1x |
| GRU | 3 | 3x |
| LSTM | 4 | 4x |

---

## Question 11: Why Padding is Used in RNN/LSTM Preprocessing

**Q:** Why are shorter sentences padded with zeros and longer sentences truncated so that all sequences have the same length?

- ✅ **Padding enables efficient batch processing because PyTorch represents batches as fixed-size tensors.**
- Padding shifts meaningful tokens closer to the final hidden state to prevent information loss over long sequences.
- The number of tokens (after padding) must match the number of hidden units.
- Padding improves accuracy by providing additional information about sentence length.

### Explanation

**PyTorch tensors must be rectangular** — every "row" needs the same length. Variable-length sentences can't be stacked into a single 2D tensor without padding.

```python
# DOESN'T work:
batch = [[1, 5, 23], [4, 12, 8, 9, 17], [2, 7]]
torch.tensor(batch)   # ❌ ValueError: inhomogeneous shape

# DOES work after padding:
batch = [
    [0, 0, 1, 5, 23],
    [4, 12, 8, 9, 17],
    [0, 0, 0, 2, 7]
]
torch.tensor(batch).shape   # ✅ (3, 5)
```

### Why batching matters

GPUs achieve speedup by performing the **same operation on many data points in parallel**. All inputs must be the same shape. Without padding:
- Sentences must be processed one at a time (no batching)
- Forward/backward passes ~100x slower
- Training takes days instead of minutes

### From our lab

```python
def _tokenize_and_pad(self, text):
    tokens = text.split()
    seq = [self.word_to_idx.get(t, 1) for t in tokens][-self.max_len:]
    return [0] * (self.max_len - len(seq)) + seq   # ← left-pad with 0
```

Left-padding so meaningful content sits closest to `h_T`.

### Why the other options are wrong

- ❌ **"Shifts tokens closer to final hidden state"**: True consequence of *left*-padding, but not the *reason* padding exists. You'd still need padding with right-padding or with LSTMs.
- ❌ **"Tokens must match hidden units"**: False — sequence length and hidden size are independent
- ❌ **"Provides info about sentence length"**: Pad tokens are noise — they can actually *hurt* performance, which is why production systems use masking/packing

### Left vs Right Padding

| Strategy | Example | Best for |
|---|---|---|
| Left | `[0, 0, hello, world]` | Many-to-one (sentiment) — meaningful tokens end at `h_T` |
| Right | `[hello, world, 0, 0]` | Many-to-many, or with `pack_padded_sequence` |

### Better alternatives in production

1. **`pack_padded_sequence`** — tells `nn.RNN`/`nn.LSTM` to skip pad positions
2. **Attention masking** — zero out attention at pad positions
3. **Bucketing** — group similar-length sequences in same batch

### Key Concept

**Padding exists because GPUs need rectangular tensors for batch parallelism.** Everything else (left vs right, masking) is refinement on top of this core requirement.

---

## Question 12: Fixing Vanishing Gradients in a Vanilla RNN

**Q:** Which modification would be most effective in alleviating the vanishing gradient problem in a vanilla RNN?

- Using the sigmoid activation function instead of tanh.
- Reduce the learning rate.
- Increasing the number of stacked hidden layers.
- ✅ **None of the above.**

### Explanation

Vanishing gradients are caused by **repeated multiplication** of gradients by `W_hh` and `tanh'(z) ≤ 1` across many time steps:
```
∂L/∂h_0 = ∂L/∂h_T · ∏_{t} (tanh'(z) · W_hh)
```
If `W_hh`'s largest eigenvalue < 1, the gradient shrinks exponentially with `T`. **None of the proposed fixes addresses this root cause** — they either make it worse or don't help.

### Why each option fails

#### ❌ Sigmoid instead of tanh — makes it WORSE

| Activation | Max derivative |
|---|---|
| `tanh'(z)` | **1** (at z=0) |
| `sigmoid'(z)` | **0.25** (at z=0) |

Sigmoid shrinks gradients **4x faster per step**. After 10 steps: `0.25^10 ≈ 9.5e-7` vs `1^10 = 1`. This is why tanh is the standard — it's the best you can do without architectural change. Sigmoid is also not zero-centered → zig-zagging optimization.

#### ❌ Reduce learning rate — irrelevant

Learning rate controls **step size**, not gradient magnitude. Vanishing gradients aren't "too big to handle safely" — they're effectively **zero**, so smaller steps don't help. This actually makes practical convergence even slower. Reducing LR is the fix for **exploding** gradients (along with clipping), not vanishing.

#### ❌ Stacking more layers — makes it WORSE

Adds depth to the gradient flow path. Now gradients must pass through:
1. **Time** (T steps of `W_hh`)
2. **Depth** (L layers)

Result: `T × L` multiplications instead of `T`. Deeper networks **exacerbate** vanishing unless you add skip connections (which LSTM does internally via the cell state).

### What actually fixes vanishing gradients

1. ✅ **LSTM / GRU** — additive cell state creates gradient highway (the main solution)
2. ✅ **Skip / residual connections** — let gradients bypass layers (ResNet)
3. ✅ **Attention** — directly connects distant time steps
4. ✅ **Orthogonal init** of `W_hh` — keeps eigenvalues near 1 (partial fix)
5. ❌ **Gradient clipping** — only fixes *exploding*, not vanishing

For this lab, the answer the lecture cares about is **LSTM**.

### Why this is a trap question

Three plausible-sounding modifications are listed, but each is either:
- **Irrelevant** (learning rate)
- **Counterproductive** (sigmoid, more layers)
- Addresses a different problem entirely

### Key Concept

**Vanishing gradients in vanilla RNNs cannot be fixed with hyperparameter tuning or activation swaps within the same architecture.** The only effective fix is an architectural change — typically to LSTM or GRU, which use **additive** (not multiplicative) state updates.

---

## Question 13: Embedding Layer Output Shape

**Q:** Vocab size = 10000, embedding dim = 256, hidden size = 512. Batch of 32 sequences × 20 tokens. What's the shape of the embedding layer output (= LSTM input)?

- ✅ **(32, 20, 256)**
- (512, 32, 20)
- (32, 20, 512)
- (32, 20, 10000)

### Explanation

The embedding layer is a **lookup table** that replaces each integer token with a dense `embed_dim`-vector. It doesn't change the batch size or sequence length.

### Shape transformation

| Stage | Shape | Notes |
|---|---|---|
| Token indices (input) | `(32, 20)` | Integers in `[0, 9999]` |
| **After embedding** | **`(32, 20, 256)`** | Each token → 256-dim vector |
| After LSTM | `(32, 20, 512)` | Each step → 512-dim hidden state |
| FC output | `(32, num_classes)` | Final predictions |

### Why each dimension is what it is

| Dim | Value | Meaning |
|---|---|---|
| 1st | 32 | Batch size |
| 2nd | 20 | Sequence length |
| 3rd | 256 | Embedding dimension |

Note that **vocab size (10000)** determines how many rows the embedding table has, but does **not** appear in the output. Similarly, hidden size (512) is what comes *after* the embedding.

### Why the other options are wrong

- ❌ **(512, 32, 20)**: 512 is LSTM hidden size, not embed dim. Also wrong order — `batch_first=True` uses `(batch, seq, features)`
- ❌ **(32, 20, 512)**: This is the LSTM's *output* shape, not the embedding's. LSTM transforms `(32, 20, 256)` → `(32, 20, 512)`
- ❌ **(32, 20, 10000)**: This would be **one-hot encoding** — exactly what embeddings exist to avoid (see Q4)

### Key Concept

**Embedding adds a feature dimension; doesn't change batch or sequence length.**
```
(batch, seq_len)  →  embedding  →  (batch, seq_len, embed_dim)
   integers                          dense vectors
```
Vocab_size = number of rows in lookup table. Output vector size = `embedding_dim`.

---

## Question 14: Increasing max_len for a Vanilla RNN

**Q:** Original `max_len = 100`. What's the most likely result of using full 200-300 token sentences with a single-layer vanilla RNN (128 hidden neurons)?

- The number of learnable parameters will increase.
- ✅ **Minimal improvement or a decrease in accuracy, as the gradient vanishes as it travels back to early time steps.**
- The accuracy will significantly increase because the model can now capture the entire context.
- The model will converge faster.

### Explanation

In BPTT, gradients flow backward through every time step, multiplied by `tanh'(z) · W_hh` at each step:
```
∂L/∂h_0 ∝ ∏_{k=1}^{T} tanh'(z_k) · W_hh
```

If `||tanh'(z) · W_hh|| ≈ 0.9` per step:
- After 100 steps: `0.9^100 ≈ 2.7e-5` — already tiny
- After 300 steps: `0.9^300 ≈ 1.8e-14` — effectively zero

So the model **cannot learn** from tokens 1-200 of a 300-token review. It only effectively uses the **last ~30-50 tokens** regardless of `max_len`.

### Why accuracy doesn't improve (and may drop)

1. **More tokens, same effective context window** — paying compute for no benefit
2. **More padding waste** for short reviews
3. **Accumulated noise** in `h_T` from many small noisy contributions
4. **More training instability** (longer BPTT chains amplify gradient noise)
5. **Slower training** (2-3x more compute per batch)

### Predicted results

| max_len | Vanilla RNN | LSTM |
|---|---|---|
| 50 | ~78% | ~84% |
| 100 (our lab) | ~80% | ~86% |
| 200 | ~78-79% (no improvement) | ~87-88% (improvement) |
| 300 | ~76-78% (slightly worse) | ~88% (more improvement) |

### Why the other options are wrong

- ❌ **"Number of parameters will increase"**: **False.** Sequence length has **no effect** on parameters. `W_hh`, `W_hx`, `b_h` are fixed. Same weights reused at every step (weight sharing).
- ❌ **"Accuracy will significantly increase"**: Intuitive but wrong for vanilla RNN — it can't *use* the longer context due to vanishing gradients. Would be true for LSTM/Transformer.
- ❌ **"Converge faster"**: Opposite — more compute per pass, more gradient instability, slower convergence.

### Direct quote from Lab 4 manual (section 3b)

> "Standard RNNs often struggle to capture long-term dependencies due to vanishing gradients... Since these models are practically limited in how much historical context they can retain, each review is truncated to a fixed length..."

The `max_len = 100` is a **workaround** for the RNN's limitations, not an optimum. Increasing it doesn't fix the underlying problem.

### What WOULD help

1. **Switch to LSTM/GRU** — cell state preserves early info
2. **Bidirectional RNN/LSTM** — process forward + backward
3. **Attention** — direct access to all hidden states
4. **Transformer** — no recurrence at all

### Key Concept

**Vanilla RNNs have a hard practical limit on effective context length (~30-100 tokens) regardless of `max_len`.** Increasing `max_len` adds compute without adding usable information, because early-token gradients vanish before they can influence learning. This is precisely why LSTMs exist.

---

## Question 15: Effect of Sequence Length on Parameters vs Computation

**Q:** Which statement is correct?

- Increasing sequence length reduces model capacity.
- ✅ **Increasing sequence length increases computation but not number of parameters.**
- Increasing sequence length changes weight dimensions.
- Increasing sequence length increases the number of parameters.

### Explanation

This is the core consequence of **weight sharing** in RNNs/LSTMs. The same weight matrices (`W_hh`, `W_hx`, gate matrices) are reused at every time step:

- **Parameters**: fixed regardless of sequence length
- **Computation**: scales **linearly** with sequence length
- **Activation memory** (for BPTT): also scales linearly

### Why parameters stay fixed

```
h_t = tanh(W_hh * h_{t-1} + W_hx * x_t + b_h)
```

Whether `t = 1`, `t = 100`, or `t = 1000`, the **same** `W_hh`, `W_hx`, `b_h` are used. Going from `T=100` to `T=300` adds zero parameters — it just unrolls the same weights more times.

Same principle as kernel sharing in CNNs: the same conv filter is applied at every spatial location, so parameters don't depend on image size.

### Why computation scales linearly

Each time step requires one matrix multiply (or 4 for LSTM). With sequence length `T`:
- T forward steps
- T backward steps (BPTT)
- Each step ~`O(h²)` work

Total ≈ `O(T × h²)`. Double `T` → double compute.

### Concrete example

Vanilla RNN, embed=128, hidden=128, batch=128:

| seq_len | Parameters | FLOPs/forward |
|---|---|---|
| 50 | 32,896 | ~210M |
| 100 | **32,896** (same) | ~420M (2×) |
| 200 | **32,896** (same) | ~840M (4×) |
| 500 | **32,896** (same) | ~2100M (10×) |

### Why the other options are wrong

- ❌ **"Reduces model capacity"**: Capacity = parameters, which is unchanged. (Effective context might be capped by vanishing gradients, but that's a learning issue, not capacity)
- ❌ **"Changes weight dimensions"**: Weight shapes depend only on `input_size`, `hidden_size`, and gate count — never on sequence length
- ❌ **"Increases parameters"**: Most common misconception. Unrolled diagrams make it look like each step has its own weights, but they all share the **same** weights. If parameters scaled with `T`, RNNs couldn't generalize to new sequence lengths

### Why weight sharing matters

1. Generalization across sequence lengths
2. Constant parameter memory
3. Translation invariance in time
4. Enables BPTT (gradients accumulated for *same* weights across steps)

### Summary Table

| Property | Scales with seq length? |
|---|---|
| # Parameters | ❌ No (constant) |
| FLOPs per forward pass | ✅ Yes (linear) |
| Activation memory for BPTT | ✅ Yes (linear) |
| Effective context (vanilla RNN) | ❌ No (capped by vanishing gradients) |
| Effective context (LSTM) | ✅ Yes (cell state highway) |

### Key Concept

**Weight sharing decouples model size from sequence length.** RNNs have fixed parameters but linear compute scaling. This is what makes recurrent networks efficient AND generalizable across arbitrary sequence lengths.

---

## Question 16: Replacing tanh with Sigmoid in Vanilla RNN

**Q:** What's the most likely consequence of switching the hidden state activation from tanh to sigmoid for long sequences in a vanilla RNN?

- It helps the model avoid the "zigzagging" update problem.
- ✅ **The gradient vanishes more rapidly as it travels back to early time steps.**
- The model will converge faster due to restricted output range.
- The model will become highly susceptible to exploding gradients.

### Explanation

Sigmoid makes vanishing gradients **dramatically worse** because of one fact:

| Activation | Max derivative |
|---|---|
| `tanh'(z)` | **1.0** (at z=0) |
| `sigmoid'(z)` | **0.25** (at z=0) |

Sigmoid's derivative is **4x smaller** at every point. Since BPTT multiplies gradients by `activation'(z) · W_hh` at every step, this 4x gap compounds exponentially.

### The math

```
∂L/∂h_0 ∝ ∏_{t=1}^{T} activation'(z_t) · W_hh
```

Assume `||W_hh|| ≈ 1`. Best-case gradient magnitude after `T` steps:

| Activation | After 10 steps | After 50 steps | After 100 steps |
|---|---|---|---|
| tanh | 1.0 | 1.0 | 1.0 |
| sigmoid | `0.25^10 ≈ 1e-6` | `0.25^50 ≈ 8e-31` | `0.25^100 ≈ 6e-61` |

Even in the **best** case, sigmoid effectively zeros out gradients after ~10 steps. With `max_len=100`, the model couldn't learn from anything beyond the last ~10 tokens.

### Sigmoid's derivative formula

```
sigmoid'(z) = sigmoid(z) · (1 - sigmoid(z))   ≤   0.25
```
Maximum is at `s=0.5`, giving `0.5 × 0.5 = 0.25`. Always smaller in saturated regions.

### Why the other options are wrong

- ❌ **"Avoids zig-zagging"**: **Backwards.** Sigmoid output is `[0, 1]` (always positive, NOT zero-centered) → suffers same zig-zagging as ReLU. Tanh's `[-1, 1]` zero-centered range is what *avoids* zig-zagging.
- ❌ **"Converges faster"**: Opposite — 4x smaller derivatives → 4x smaller weight updates → slower convergence
- ❌ **"Exploding gradients"**: Sigmoid is **bounded** in `[0, 1]`. That's the ReLU failure mode (unbounded output). Sigmoid's failure is **vanishing**, not exploding.

### Comparison: tanh vs sigmoid vs ReLU

| Property | tanh | sigmoid | ReLU |
|---|---|---|---|
| Output range | [-1, 1] | [0, 1] | [0, ∞) |
| Max derivative | 1.0 | **0.25** | 1.0 (active) |
| Zero-centered? | ✅ Yes | ❌ No | ❌ No |
| Forward stability | ✅ Bounded | ✅ Bounded | ❌ Explodes |
| Vanishing resistance | ✅ Best | ❌ Worst | ✅ Best (active) |
| Zig-zagging? | ✅ Avoids | ❌ Causes | ❌ Causes |

Tanh is the only activation that simultaneously (1) bounds the forward pass, (2) has reasonable derivative magnitude, AND (3) is zero-centered.

### Lecture connection

Tanh sits at a sweet spot. Replacing it with:
- **Sigmoid** → vanishing gradients (this question)
- **ReLU** → exploding gradients (Question 7)

Both break the vanilla RNN, just in different ways.

### Key Concept

**Sigmoid is strictly worse than tanh for RNN hidden states** because its max derivative (0.25) is 4x smaller, accelerating vanishing gradients. The only place sigmoid belongs in an RNN is **inside LSTM gates**, where `[0,1]` output is *desirable* for gating (not for hidden states).

---

## Question 17: LSTM Output Shape in Many-to-Many (Synced) Architecture

**Q:** Batch size B, sequence length T, vocab V, embed dim d, LSTM hidden size h. In a many-to-many (synchronized) LSTM, what is the shape of the LSTM's output tensor?

- (B, h)
- ✅ **(B, T, h)**
- (B, T, d)
- (B, T, V)

### Explanation

In a **many-to-many synchronized** architecture, the LSTM produces an output at **every** time step. This is exactly what `nn.LSTM`'s `output` tensor returns by default.

| Dim | Value | Meaning |
|---|---|---|
| 1st | B | Batch size |
| 2nd | T | One output per input step (synchronized) |
| 3rd | h | Hidden state dimension |

### Synchronized many-to-many

```
Time:    t=1   t=2   t=3  ...  t=T
Input:   x_1   x_2   x_3        x_T
LSTM:    h_1 → h_2 → h_3 → ... → h_T
Output:  y_1   y_2   y_3        y_T
```

Each input token → its own output. Examples: POS tagging, NER, frame-level speech recognition.

### Pipeline shape progression

| Stage | Shape | Notes |
|---|---|---|
| Token indices | `(B, T)` | Integer tokens |
| After embedding | `(B, T, d)` | LSTM input |
| **After LSTM** | **`(B, T, h)`** | ← this question |
| After per-step FC | `(B, T, num_classes)` | One prediction per step |

The FC layer is applied to each step (`nn.Linear` broadcasts over the leading dimensions):
```python
output = self.fc(lstm_output)   # (B, T, h) → (B, T, num_classes)
```

### Why the other options are wrong

- ❌ **(B, h)**: Many-to-**one** output (collapses time). Used for sentiment classification (our lab), not synced many-to-many
- ❌ **(B, T, d)**: This is the embedding's *output* (LSTM's *input*) — uses embed dim `d`, not hidden size `h`
- ❌ **(B, T, V)**: Output of a *classification head* that projects to vocab size (e.g., language modeling). Not the LSTM itself

### `output` vs `h_n`

`nn.LSTM` returns two tensors:
```python
output, (h_n, c_n) = lstm(input, (h_0, c_0))
```

| Tensor | Shape | Use |
|---|---|---|
| `output` | `(B, T, h)` | All hidden states — for many-to-many |
| `h_n` | `(num_layers, B, h)` | Final hidden state — for many-to-one |

`output[:, -1, :] == h_n[-1]` (last step = top layer's final hidden state).

### Sequence configuration summary

| Config | Tensor used | Output shape | Example |
|---|---|---|---|
| Many-to-one | `h_n` | `(B, h)` | Sentiment (our lab) |
| **Many-to-many synced** | **`output`** | **`(B, T, h)`** | POS tagging |
| Many-to-many async | encoder + decoder | Variable | Machine translation |
| One-to-many | decoder `output` | `(B, T_out, h)` | Image captioning |

### Key Concept

**LSTM `output` tensor is always `(B, T, h)`.** The configuration determines which slice you consume:
- Many-to-one: `output[:, -1, :]` → `(B, h)`
- Many-to-many synced: full `output` → `(B, T, h)`

Synchronized many-to-many means **input length = output length** with one-to-one alignment.

---

## Question 18: Pre-padding vs Post-padding (FALSE statement)

**Q:** Padding can be applied at the beginning (pre-padding) or end (post-padding). Which statement is FALSE?

- Post-padding ensures the model begins processing meaningful information at the first time step.
- Pre-padding is preferred in many-to-one tasks.
- Pre-padding keeps meaningful tokens closer to the final hidden state.
- ✅ **Pre-padding is often preferred in many-to-many tasks.** (FALSE)

### Explanation

**Pre-padding is preferred for many-to-ONE tasks**, not many-to-many. For many-to-many, **post-padding** is the standard choice.

### Why pre-padding helps many-to-one tasks

In many-to-one (e.g., sentiment), only `h_T` is used for prediction. We want meaningful tokens **as close to t=T as possible**:

```
Pre-padding for sentiment:
[<PAD>, <PAD>, <PAD>, the, movie, was, great]
                                            ↑ h_T influenced by "great"
```

The meaningful tokens occupy the latest time steps, where vanishing gradients haven't yet attenuated them. This is what our lab uses:
```python
return [0] * (self.max_len - len(seq)) + seq   # pre-padding
```

If post-padded instead:
```
[the, movie, was, great, <PAD>, <PAD>, <PAD>]
                                          ↑ h_T influenced by <PAD> noise
```
By time `t=T`, real content is far away (vanishing gradients hurt) and `h_T` is diluted by pad processing.

### Why post-padding is preferred for many-to-many tasks

In many-to-many synced (e.g., POS tagging), every step produces an output:

```
Pre-padding for POS tagging (BAD):
Input:   [<PAD>, <PAD>, <PAD>, the, dog, ran]
Output:  [<PAD>, <PAD>, <PAD>, DET, NOUN, VERB]
```
Hidden state gets polluted by processing garbage at t=1,2,3 before real content arrives.

```
Post-padding for POS tagging (GOOD):
Input:   [the, dog, ran, <PAD>, <PAD>, <PAD>]
Output:  [DET, NOUN, VERB, <PAD>, <PAD>, <PAD>]
```
Model immediately processes meaningful tokens, builds clean hidden state. Pairs naturally with `pack_padded_sequence`.

### Why the other three statements are TRUE

- ✅ **Post-padding starts with meaningful info at t=1** — correct
- ✅ **Pre-padding preferred for many-to-one** — correct (meaningful tokens end at `h_T`)
- ✅ **Pre-padding keeps meaningful tokens closer to final hidden state** — correct, this is the *mechanism*

### Summary

| Padding | Best for | Why |
|---|---|---|
| **Pre-padding** (left) | Many-to-one | Meaningful tokens end at `h_T` |
| **Post-padding** (right) | Many-to-many | Meaningful tokens at t=1; works with packing |

### Key Concept

**Pad placement should match task type:**
- Many-to-one → **pre-pad** (real tokens near `h_T`)
- Many-to-many → **post-pad** (real content first, clean outputs from t=1)

Our lab does many-to-one sentiment with pre-padding — the optimal combination.

---

## Question 19: Frozen Pre-trained Word Embeddings (Word2Vec)

**Q:** Which statement is TRUE when using a pre-trained word embedding (like Word2Vec) that is kept fixed during RNN training?

- The GPU memory required for gradient storage increases.
- ✅ **The word embedding cannot adapt to domain-specific semantic shifts.**
- The word embedding causes dimension mismatch with the hidden states.
- The model needs more epochs to converge.

### Explanation

When you freeze a pre-trained embedding (`requires_grad=False`), the vectors stay exactly as trained on the original corpus and never update during your task training:
```python
self.embedding = nn.Embedding.from_pretrained(word2vec_weights, freeze=True)
```

The model is stuck with whatever word meanings the original corpus encoded.

### Domain-specific semantic shifts

The same word means different things in different domains:

| Word | General (Word2Vec) | Domain-specific |
|---|---|---|
| "sick" | ill | "awesome" (slang) |
| "cell" | prison cell | mobile phone / biology |
| "bug" | insect | software defect |
| "jaguar" | animal | car brand |
| "python" | snake | programming language |
| "apple" | fruit | company |
| "killer" (movies) | murderer | "amazing" ("killer soundtrack") |

A frozen embedding can't learn that "python" means a programming language even if every training example uses it that way.

### Why the other options are wrong

- ❌ **"GPU memory increases"**: **Opposite** — freezing reduces memory because PyTorch doesn't allocate gradient buffers. For a 50k vocab × 300-dim = 15M parameters whose gradients you skip. **Saves** memory.
- ❌ **"Dimension mismatch with hidden states"**: **False.** Embed dim and hidden size are independent. `W_hx` is shaped `(h, embed_dim)` and projects between them automatically. 300-dim Word2Vec works fine with 128-hidden LSTM.
- ❌ **"Needs more epochs to converge"**: Usually **false.** Fewer trainable params → faster, more stable convergence (just lower ceiling).

### When freezing is good vs bad

**Helps when:**
- Small dataset (overfitting risk if fine-tuning)
- Pretraining domain matches your domain
- You want faster training / less memory

**Hurts when:**
- Specialized vocabulary
- Words have different meanings in your domain
- Large in-domain dataset available

### Trade-offs

| Property | Frozen | Fine-tuned |
|---|---|---|
| Domain adaptation | ❌ No | ✅ Yes |
| Trainable params | Fewer | More |
| GPU memory | Lower | Higher |
| Training speed | Faster | Slower |
| Overfitting risk | Lower | Higher |

### Common middle-ground: Gradual Unfreezing

1. **Phase 1**: Freeze embedding, train RNN/classifier
2. **Phase 2**: Unfreeze with very small learning rate (~10x smaller)

Used in ULMFiT, BERT fine-tuning, etc.

### Connection to the lab

We used `nn.Embedding(vocab_size, embed_size)` trained from scratch on IMDB — fully adapts to the movie domain but needs enough data to learn good vectors. Our 30k training samples worked well enough.

### Key Concept

**Freezing a pre-trained embedding trades adaptability for stability.** You inherit pre-trained knowledge but lose the ability to refine it. The fundamental cost: the embedding cannot adapt to domain-specific semantic shifts.

---

## Question 20: Identity-Init W_hh + ReLU (IRNN)

**Q:** Initialize `W_hh` as an identity matrix and use ReLU activation in a vanilla RNN trained on long sequences. Most likely outcome?

- The model will cease learning after a certain number of timesteps despite not reaching convergence.
- ✅ **The model will behave similarly to a ResNet, potentially mitigating vanishing gradients but risking exploding gradients.**
- The model will suffer from more severe vanishing gradients compared to vanilla RNN with tanh.
- None of the above.

### Explanation

This is the **IRNN (Identity-Recurrent Neural Network)** from Le, Jaitly, Hinton (2015). It's a clever trick that turns a vanilla RNN into something resembling ResNet.

### How it works

With `W_hh = I`, `b_h = 0`, ReLU activation:
```
h_t = ReLU(I * h_{t-1} + W_hx * x_t)
    = ReLU(h_{t-1} + W_hx * x_t)
```

This is **mathematically a residual (skip) connection**:
```
h_t ≈ h_{t-1} + (new info from x_t)
```

The previous hidden state passes through unchanged (when ReLU is active), and the input adds a perturbation. Same idea as ResNet's `f(x) + x`.

### Why this mitigates vanishing gradients

```
∂h_t/∂h_{t-1} = ReLU'(z) · W_hh = 1 · I = I
```

Gradient flows backward **unchanged** at every time step:
```
∂h_T/∂h_0 = I^T = I
```

No vanishing! Same mechanism as ResNet's skip connections.

### But: exploding gradient risk

Identity init only stops *vanishing*. It does nothing to prevent growth:
1. `W_hh` drifts away from identity as training progresses
2. If eigenvalues > 1, gradients explode exponentially
3. ReLU has no upper bound → forward pass can also explode

IRNNs typically need **gradient clipping** to train stably.

### Why the other options are wrong

- ❌ **"Cease learning after some timesteps"**: Identity init *preserves* gradient signal for many steps. This describes vanishing, which IRNNs *avoid*.
- ❌ **"More severe vanishing than tanh"**: **Opposite.** Identity init + ReLU is a known *fix* for vanishing gradients. ReLU's derivative = 1 in active region (vs `tanh' ≤ 1`).
- ❌ **"None of the above"**: There's a clear correct answer.

### The deep connection: ResNet ↔ LSTM ↔ IRNN

| Architecture | Skip mechanism | Effect |
|---|---|---|
| **ResNet** | `y = F(x) + x` (explicit) | Gradient highway through depth |
| **LSTM** | `c_t = f_t · c_{t-1} + i_t · g_t` (additive) | Gradient highway through time |
| **IRNN** | `h_t = ReLU(h_{t-1} + W_hx · x_t)` (identity init) | Gradient highway through time (fragile) |

All three solve vanishing via **additive update paths**. Lecture slide 63 makes this LSTM↔ResNet analogy explicit.

### Trade-offs

| Property | IRNN | Vanilla RNN (tanh) | LSTM |
|---|---|---|---|
| Vanishing | ✅ Mitigated | ❌ Severe | ✅ Solved |
| Exploding | ❌ Risky | ⚠️ Possible | ✅ Controlled |
| Training stability | ⚠️ Fragile | ⚠️ Unstable | ✅ Robust |
| Parameter count | Same as RNN | Baseline | 4x |
| Forward stability | ❌ Unbounded | ✅ Bounded | ✅ Gated |
| Long-range capability | ✅ Good | ❌ Poor | ✅ Excellent |

### Why IRNNs aren't mainstream

1. **Training instability** without careful clipping/LR tuning
2. **LSTMs are more robust** with much less hyperparameter sensitivity
3. **Transformers** came along and dominated

IRNNs are an elegant proof that gradient highways don't *require* gating, but gating provides much more reliable training in practice.

### Key Concept

**Identity-init ReLU creates a ResNet-like gradient highway through time**, mitigating vanishing gradients. But it inherits ReLU's unbounded range → exploding gradient risk. Beautiful in theory, fragile in practice — which is why LSTMs (with gated bounded design) won.
