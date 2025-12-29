# Artificial-Humanity

A basic Large Language Model (LLM) implementation in C, demonstrating core concepts of transformer-based neural networks.

## Features

- **Pure C Implementation**: Written entirely in C with no external dependencies except standard math library
- **Transformer Architecture**: Implements key components including:
  - Token and positional embeddings
  - Multi-head self-attention mechanism
  - Feed-forward neural networks
  - Layer normalization
  - Residual connections
- **Forward Pass**: Complete forward propagation through the network
- **Text Generation**: Auto-regressive token generation with configurable parameters
- **Configurable**: Easy to adjust model dimensions, layers, and vocabulary size

## Project Structure

```
.
├── llm.h           # Header file with LLM API and data structures
├── llm.c           # Core LLM implementation
├── example.c       # Demo program showing usage
├── Makefile        # Build configuration
└── README.md       # This file
```

## Building

Requirements:
- GCC compiler
- Make
- Standard C library and math library

Build the project:

```bash
make
```

This will compile the LLM library and example program.

## Running

Execute the example program:

```bash
make run
```

Or run directly:

```bash
./example
```

## Usage

### Basic Example

```c
#include "llm.h"

// Configure the model
LLMConfig config = {
    .vocab_size = 100,      // Size of vocabulary
    .n_layers = 1,          // Number of transformer layers
    .n_heads = 4,           // Number of attention heads
    .d_model = 64,          // Model dimension
    .d_ff = 256,            // Feed-forward dimension
    .max_seq_len = 32       // Maximum sequence length
};

// Initialize model
LLM* model = llm_init(config);

// Forward pass
int tokens[] = {1, 5, 10, 15, 20};
float* logits = malloc(config.vocab_size * sizeof(float));
llm_forward(model, tokens, 5, logits);

// Generate text
int prompt[] = {1, 2, 3};
int generated[10];
llm_generate(model, prompt, 3, generated, 10);

// Cleanup
llm_free(model);
```

## API Reference

### Data Structures

- **LLMConfig**: Configuration parameters for the model
- **LLM**: Main model structure containing weights and state
- **LLMWeights**: All model parameters (embeddings, attention, feed-forward)
- **LLMState**: Runtime activations and intermediate values

### Functions

- `LLM* llm_init(LLMConfig config)`: Initialize a new model with given configuration
- `void llm_free(LLM* model)`: Free all memory associated with the model
- `void llm_forward(LLM* model, int* tokens, int n_tokens, float* output)`: Run forward pass
- `int llm_sample(float* logits, int vocab_size)`: Sample next token from logits (greedy)
- `void llm_generate(LLM* model, int* prompt, int prompt_len, int* output, int max_new_tokens)`: Generate sequence

## Implementation Details

This is a simplified educational implementation that demonstrates:

1. **Embeddings**: Combines token and positional embeddings
2. **Self-Attention**: Computes Q, K, V matrices and attention scores
3. **Feed-Forward Network**: Two-layer network with ReLU activation
4. **Layer Normalization**: Normalizes activations for stable training
5. **Residual Connections**: Helps with gradient flow

The implementation uses:
- Matrix multiplication for linear transformations
- Softmax for attention score normalization
- ReLU for non-linear activation
- Greedy sampling for token selection

## Limitations

This is a basic educational implementation and has several limitations:

- No training code (weights are randomly initialized)
- Single layer only (can be extended)
- Simplified attention (no masking for causal generation)
- No optimization (e.g., no KV caching)
- Greedy sampling only (no temperature, top-k, or top-p)
- No tokenization (works directly with token IDs)

## License

Open source - feel free to use and modify.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.