# Temporal Dynamics Module

**PILLAR 3: Dynamics - Temporal Prediction and Belief Propagation**

This module implements temporal dynamics components for AIME's recurrent processing.

## Architecture Overview

```
    Temporal Dynamics Pipeline
              │
         LSTMLayer
              │
    [Recurrent Processing]
```

## Theoretical Foundation

AIME models temporal dynamics through recurrent neural networks that:
1. **Propagate beliefs over time**: h_t = f(h_{t-1}, z_t, c_t, a_t)
2. **Maintain temporal context**: Hidden states encode history
3. **Enable prediction**: Future states depend on past observations

The LSTM acts as the "working memory" of the system, integrating:
- **Latent states** (z_t): Current observations
- **Context** (c_t): Perceiver-extracted features
- **Actions** (a_t): Control signals
- **History** (h_{t-1}): Past temporal context

## Components

### LSTMLayer (`lstm.py`)

Orthogonal LSTM with layer normalization for stable temporal processing.

**Key Features:**
- Orthogonal weight initialization for gradient stability
- Layer normalization after LSTM output
- Mask support for variable-length sequences
- Efficient single-timestep processing

**Input:**
- x: [B, input_dim] - Concatenated input (z_t + c_t + a_t)
- hxs: [n_layers, B, hidden_dim] - Hidden states
- cxs: [n_layers, B, hidden_dim] - Cell states
- masks: [B] - Active sequence mask

**Output:**
- output: [B, hidden_dim] - Processed features
- (hxs_new, cxs_new): Updated hidden/cell states

**Usage:**
```python
from temporal_dynamics import LSTMLayer

# Initialize
lstm = LSTMLayer(
    input_size=298,      # z_dim + context_dim + action_dim
    hidden_size=512,
    n_lstm_layers=1,
    use_orthogonal=True
)

# Forward pass (single timestep)
h_t, (h_next, c_next) = lstm(
    x=torch.cat([z_t, c_t, a_t], dim=-1),
    hxs=h_prev,
    cxs=c_prev,
    masks=torch.ones(batch_size)
)
```

## Tensor Shape Reference

### LSTM Processing

```python
# Single timestep
x_t: [B, 298]               # Input: z (36) + context (256) + action (6)
h_prev: [1, B, 512]         # Previous hidden state (1 layer)
c_prev: [1, B, 512]         # Previous cell state
masks: [B]                  # Active sequences (1.0 = active, 0.0 = masked)

# Output
h_t: [B, 512]               # Processed features
h_next: [1, B, 512]         # Updated hidden state
c_next: [1, B, 512]         # Updated cell state
```

### Sequence Processing

```python
# Process full sequence (loop over T timesteps)
sequence_length = 16
h_t = h_0  # [1, B, 512]
c_t = c_0  # [1, B, 512]

for t in range(sequence_length):
    x_t = inputs[:, t, :]  # [B, 298]
    h_t, (h_t, c_t) = lstm(x_t, h_t, c_t, masks)
    # h_t: [B, 512] at each timestep
```

## Integration with AIME

The LSTM is used in the main VRNN model (`dpgmm_stickbreaking_prior_vrnn.py`):

```python
# In forward_sequence():
rnn_input = torch.cat([z_posterior, c_t, actions], dim=-1)  # [B, T, 298]

# Process sequence
for t in range(seq_len):
    h_t, (h_next, c_next) = self._rnn(
        x=rnn_input[:, t, :],
        hxs=h_prev,
        cxs=c_prev,
        masks=masks
    )
    h_sequence.append(h_t)
    h_prev, c_prev = h_next, c_next
```

## Design Principles

1. **Orthogonal Initialization**: Prevents vanishing/exploding gradients
2. **Layer Normalization**: Stabilizes training dynamics
3. **Mask Support**: Handles variable-length sequences
4. **Single-Step Interface**: Efficient for RL environments

## Benefits for AI Coders

1. **Focused Module**: LSTM logic isolated from model
2. **Clear Interface**: Simple forward() signature
3. **Well-Documented**: Tensor shapes and usage examples
4. **Easy Testing**: Can test LSTM independently

## Future Extensions

- [ ] Add GRU variant
- [ ] Add bidirectional processing
- [ ] Add attention over hidden states
- [ ] Add hierarchical temporal scales

## References

**Orthogonal Initialization:**
- Saxe et al. "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks" (2013)

**Layer Normalization:**
- Ba et al. "Layer Normalization" (2016)

---

**For questions about this module:**
- LSTM implementation: See inline comments in `lstm.py`
- Integration: See `VRNN/dpgmm_stickbreaking_prior_vrnn.py`
- Theory: See `docs/THEORY_AND_PHILOSOPHY.md`
