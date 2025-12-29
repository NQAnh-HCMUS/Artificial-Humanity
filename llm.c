#include "llm.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// Helper function: matrix multiplication
// C = A * B where A is (m x k) and B is (k x n)
static void matmul(float* C, float* A, float* B, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// Helper function: softmax
static void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// Helper function: ReLU activation
static void relu(float* x, int size) {
    for (int i = 0; i < size; i++) {
        if (x[i] < 0) x[i] = 0;
    }
}

// Helper function: layer normalization
static void layer_norm(float* out, float* x, float* gamma, float* beta, int size) {
    float mean = 0.0f;
    for (int i = 0; i < size; i++) {
        mean += x[i];
    }
    mean /= size;
    
    float var = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= size;
    
    float std = sqrtf(var + 1e-5f);
    for (int i = 0; i < size; i++) {
        out[i] = gamma[i] * (x[i] - mean) / std + beta[i];
    }
}

// Initialize model
LLM* llm_init(LLMConfig config) {
    LLM* model = (LLM*)malloc(sizeof(LLM));
    if (!model) return NULL;
    
    model->config = config;
    
    // Initialize all pointers to NULL for safe cleanup on error
    model->weights.token_embedding = NULL;
    model->weights.position_embedding = NULL;
    model->weights.wq = NULL;
    model->weights.wk = NULL;
    model->weights.wv = NULL;
    model->weights.wo = NULL;
    model->weights.w1 = NULL;
    model->weights.w2 = NULL;
    model->weights.ln1_gamma = NULL;
    model->weights.ln1_beta = NULL;
    model->weights.ln2_gamma = NULL;
    model->weights.ln2_beta = NULL;
    model->weights.output_weights = NULL;
    model->state.x = NULL;
    model->state.q = NULL;
    model->state.k = NULL;
    model->state.v = NULL;
    model->state.att = NULL;
    model->state.att_out = NULL;
    model->state.ff = NULL;
    model->state.logits = NULL;
    
    // Allocate weights (initialized to random small values)
    // Note: Seed random once at program startup for better behavior
    static int seeded = 0;
    if (!seeded) {
        srand(time(NULL));
        seeded = 1;
    }
    
    int d = config.d_model;
    int vocab = config.vocab_size;
    int seq_len = config.max_seq_len;
    
    model->weights.token_embedding = (float*)malloc(vocab * d * sizeof(float));
    if (!model->weights.token_embedding) goto cleanup;
    
    model->weights.position_embedding = (float*)malloc(seq_len * d * sizeof(float));
    if (!model->weights.position_embedding) goto cleanup;
    
    model->weights.wq = (float*)malloc(d * d * sizeof(float));
    if (!model->weights.wq) goto cleanup;
    
    model->weights.wk = (float*)malloc(d * d * sizeof(float));
    if (!model->weights.wk) goto cleanup;
    
    model->weights.wv = (float*)malloc(d * d * sizeof(float));
    if (!model->weights.wv) goto cleanup;
    
    model->weights.wo = (float*)malloc(d * d * sizeof(float));
    if (!model->weights.wo) goto cleanup;
    
    model->weights.w1 = (float*)malloc(d * config.d_ff * sizeof(float));
    if (!model->weights.w1) goto cleanup;
    
    model->weights.w2 = (float*)malloc(config.d_ff * d * sizeof(float));
    if (!model->weights.w2) goto cleanup;
    
    model->weights.ln1_gamma = (float*)malloc(d * sizeof(float));
    if (!model->weights.ln1_gamma) goto cleanup;
    
    model->weights.ln1_beta = (float*)malloc(d * sizeof(float));
    if (!model->weights.ln1_beta) goto cleanup;
    
    model->weights.ln2_gamma = (float*)malloc(d * sizeof(float));
    if (!model->weights.ln2_gamma) goto cleanup;
    
    model->weights.ln2_beta = (float*)malloc(d * sizeof(float));
    if (!model->weights.ln2_beta) goto cleanup;
    
    model->weights.output_weights = (float*)malloc(d * vocab * sizeof(float));
    if (!model->weights.output_weights) goto cleanup;
    
    // Initialize weights with small random values
    for (int i = 0; i < vocab * d; i++) {
        model->weights.token_embedding[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    }
    for (int i = 0; i < seq_len * d; i++) {
        model->weights.position_embedding[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    }
    for (int i = 0; i < d * d; i++) {
        model->weights.wq[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
        model->weights.wk[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
        model->weights.wv[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
        model->weights.wo[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    }
    for (int i = 0; i < d * config.d_ff; i++) {
        model->weights.w1[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    }
    for (int i = 0; i < config.d_ff * d; i++) {
        model->weights.w2[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    }
    for (int i = 0; i < d; i++) {
        model->weights.ln1_gamma[i] = 1.0f;
        model->weights.ln1_beta[i] = 0.0f;
        model->weights.ln2_gamma[i] = 1.0f;
        model->weights.ln2_beta[i] = 0.0f;
    }
    for (int i = 0; i < d * vocab; i++) {
        model->weights.output_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    }
    
    // Allocate state buffers
    model->state.x = (float*)malloc(seq_len * d * sizeof(float));
    if (!model->state.x) goto cleanup;
    
    model->state.q = (float*)malloc(seq_len * d * sizeof(float));
    if (!model->state.q) goto cleanup;
    
    model->state.k = (float*)malloc(seq_len * d * sizeof(float));
    if (!model->state.k) goto cleanup;
    
    model->state.v = (float*)malloc(seq_len * d * sizeof(float));
    if (!model->state.v) goto cleanup;
    
    model->state.att = (float*)malloc(seq_len * seq_len * sizeof(float));
    if (!model->state.att) goto cleanup;
    
    model->state.att_out = (float*)malloc(seq_len * d * sizeof(float));
    if (!model->state.att_out) goto cleanup;
    
    model->state.ff = (float*)malloc(seq_len * config.d_ff * sizeof(float));
    if (!model->state.ff) goto cleanup;
    
    model->state.logits = (float*)malloc(vocab * sizeof(float));
    if (!model->state.logits) goto cleanup;
    
    return model;

cleanup:
    // Free any allocated memory before returning NULL
    llm_free(model);
    return NULL;
}

// Free model
void llm_free(LLM* model) {
    if (!model) return;
    
    free(model->weights.token_embedding);
    free(model->weights.position_embedding);
    free(model->weights.wq);
    free(model->weights.wk);
    free(model->weights.wv);
    free(model->weights.wo);
    free(model->weights.w1);
    free(model->weights.w2);
    free(model->weights.ln1_gamma);
    free(model->weights.ln1_beta);
    free(model->weights.ln2_gamma);
    free(model->weights.ln2_beta);
    free(model->weights.output_weights);
    
    free(model->state.x);
    free(model->state.q);
    free(model->state.k);
    free(model->state.v);
    free(model->state.att);
    free(model->state.att_out);
    free(model->state.ff);
    free(model->state.logits);
    
    free(model);
}

// Forward pass
void llm_forward(LLM* model, int* tokens, int n_tokens, float* output) {
    int d = model->config.d_model;
    int vocab = model->config.vocab_size;
    
    // Bounds check on n_tokens
    if (n_tokens > model->config.max_seq_len) {
        n_tokens = model->config.max_seq_len;
    }
    if (n_tokens <= 0) {
        return;  // Invalid input
    }
    
    // Embedding: token + position
    for (int i = 0; i < n_tokens; i++) {
        int token = tokens[i];
        // Clamp token to valid range
        if (token < 0) token = 0;
        if (token >= vocab) token = vocab - 1;
        
        for (int j = 0; j < d; j++) {
            model->state.x[i * d + j] = 
                model->weights.token_embedding[token * d + j] +
                model->weights.position_embedding[i * d + j];
        }
    }
    
    // Single transformer layer (simplified)
    // Self-attention
    matmul(model->state.q, model->state.x, model->weights.wq, n_tokens, d, d);
    matmul(model->state.k, model->state.x, model->weights.wk, n_tokens, d, d);
    matmul(model->state.v, model->state.x, model->weights.wv, n_tokens, d, d);
    
    // Attention scores: Q * K^T / sqrt(d)
    float scale = 1.0f / sqrtf((float)d);
    for (int i = 0; i < n_tokens; i++) {
        for (int j = 0; j < n_tokens; j++) {
            float score = 0.0f;
            for (int k = 0; k < d; k++) {
                score += model->state.q[i * d + k] * model->state.k[j * d + k];
            }
            model->state.att[i * n_tokens + j] = score * scale;
        }
        // Softmax over attention scores
        softmax(&model->state.att[i * n_tokens], n_tokens);
    }
    
    // Attention output: att * V
    for (int i = 0; i < n_tokens; i++) {
        for (int j = 0; j < d; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n_tokens; k++) {
                sum += model->state.att[i * n_tokens + k] * model->state.v[k * d + j];
            }
            model->state.att_out[i * d + j] = sum;
        }
    }
    
    // Output projection
    float* att_proj = (float*)malloc(n_tokens * d * sizeof(float));
    if (!att_proj) return;  // Early return on allocation failure
    
    matmul(att_proj, model->state.att_out, model->weights.wo, n_tokens, d, d);
    
    // Residual connection and layer norm
    float* ln1_out = (float*)malloc(n_tokens * d * sizeof(float));
    if (!ln1_out) {
        free(att_proj);
        return;
    }
    
    for (int i = 0; i < n_tokens; i++) {
        for (int j = 0; j < d; j++) {
            model->state.x[i * d + j] += att_proj[i * d + j];
        }
        layer_norm(&ln1_out[i * d], &model->state.x[i * d], 
                   model->weights.ln1_gamma, model->weights.ln1_beta, d);
    }
    
    // Feed-forward network
    matmul(model->state.ff, ln1_out, model->weights.w1, n_tokens, model->config.d_ff, d);
    relu(model->state.ff, n_tokens * model->config.d_ff);
    
    float* ff_out = (float*)malloc(n_tokens * d * sizeof(float));
    if (!ff_out) {
        free(att_proj);
        free(ln1_out);
        return;
    }
    
    matmul(ff_out, model->state.ff, model->weights.w2, n_tokens, d, model->config.d_ff);
    
    // Residual connection and layer norm
    for (int i = 0; i < n_tokens; i++) {
        for (int j = 0; j < d; j++) {
            model->state.x[i * d + j] = ln1_out[i * d + j] + ff_out[i * d + j];
        }
        layer_norm(&model->state.x[i * d], &model->state.x[i * d],
                   model->weights.ln2_gamma, model->weights.ln2_beta, d);
    }
    
    // Output layer: project to vocabulary
    // Use the last token's representation
    float* last_hidden = &model->state.x[(n_tokens - 1) * d];
    for (int i = 0; i < vocab; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            sum += last_hidden[j] * model->weights.output_weights[j * vocab + i];
        }
        model->state.logits[i] = sum;
    }
    
    // Copy logits to output
    memcpy(output, model->state.logits, vocab * sizeof(float));
    
    free(att_proj);
    free(ln1_out);
    free(ff_out);
}

// Sample next token from logits
int llm_sample(float* logits, int vocab_size) {
    // Simple greedy sampling (pick argmax)
    int max_idx = 0;
    float max_val = logits[0];
    
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

// Generate tokens
void llm_generate(LLM* model, int* prompt, int prompt_len, int* output, int max_new_tokens) {
    // Validate inputs
    if (prompt_len <= 0 || max_new_tokens <= 0) return;
    if (prompt_len > model->config.max_seq_len) {
        prompt_len = model->config.max_seq_len;
    }
    
    int* context = (int*)malloc((prompt_len + max_new_tokens) * sizeof(int));
    if (!context) return;  // Early return on allocation failure
    
    memcpy(context, prompt, prompt_len * sizeof(int));
    
    int context_len = prompt_len;
    float* logits = (float*)malloc(model->config.vocab_size * sizeof(float));
    if (!logits) {
        free(context);
        return;
    }
    
    for (int i = 0; i < max_new_tokens; i++) {
        // Forward pass
        llm_forward(model, context, context_len, logits);
        
        // Sample next token
        int next_token = llm_sample(logits, model->config.vocab_size);
        output[i] = next_token;
        
        // Add to context
        if (context_len < model->config.max_seq_len) {
            context[context_len++] = next_token;
        } else {
            // Slide window
            memmove(context, context + 1, (context_len - 1) * sizeof(int));
            context[context_len - 1] = next_token;
        }
    }
    
    free(context);
    free(logits);
}
