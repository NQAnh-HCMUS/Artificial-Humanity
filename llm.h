#ifndef LLM_H
#define LLM_H

#include <stddef.h>

// Configuration structure for the LLM
typedef struct {
    int vocab_size;      // Size of vocabulary
    int n_layers;        // Number of transformer layers
    int n_heads;         // Number of attention heads
    int d_model;         // Model dimension
    int d_ff;            // Feed-forward dimension
    int max_seq_len;     // Maximum sequence length
} LLMConfig;

// Model weights structure
typedef struct {
    float* token_embedding;    // Token embedding weights
    float* position_embedding; // Position embedding weights
    float* wq;                 // Query weights
    float* wk;                 // Key weights
    float* wv;                 // Value weights
    float* wo;                 // Output weights
    float* w1;                 // Feed-forward layer 1 weights
    float* w2;                 // Feed-forward layer 2 weights
    float* ln1_gamma;          // Layer norm 1 gamma
    float* ln1_beta;           // Layer norm 1 beta
    float* ln2_gamma;          // Layer norm 2 gamma
    float* ln2_beta;           // Layer norm 2 beta
    float* output_weights;     // Final output layer weights
} LLMWeights;

// Runtime state/activations
typedef struct {
    float* x;          // Input embeddings
    float* q;          // Query
    float* k;          // Key
    float* v;          // Value
    float* att;        // Attention scores
    float* att_out;    // Attention output
    float* ff;         // Feed-forward intermediate
    float* logits;     // Output logits
} LLMState;

// Model structure
typedef struct {
    LLMConfig config;
    LLMWeights weights;
    LLMState state;
} LLM;

// Function declarations
LLM* llm_init(LLMConfig config);
void llm_free(LLM* model);
void llm_forward(LLM* model, int* tokens, int n_tokens, float* output);
int llm_sample(float* logits, int vocab_size);
void llm_generate(LLM* model, int* prompt, int prompt_len, int* output, int max_new_tokens);

#endif // LLM_H
