#include <stdio.h>
#include <stdlib.h>
#include "llm.h"

int main(void) {
    printf("=== Basic LLM in C ===\n\n");
    
    // Configure a small model
    LLMConfig config = {
        .vocab_size = 100,      // Small vocabulary
        .n_layers = 1,          // Single layer
        .n_heads = 4,           // 4 attention heads
        .d_model = 64,          // 64-dimensional embeddings
        .d_ff = 256,            // Feed-forward dimension
        .max_seq_len = 32       // Max sequence length
    };
    
    printf("Initializing LLM with configuration:\n");
    printf("  Vocabulary size: %d\n", config.vocab_size);
    printf("  Number of layers: %d\n", config.n_layers);
    printf("  Attention heads: %d\n", config.n_heads);
    printf("  Model dimension: %d\n", config.d_model);
    printf("  Feed-forward dimension: %d\n", config.d_ff);
    printf("  Max sequence length: %d\n\n", config.max_seq_len);
    
    // Initialize model
    LLM* model = llm_init(config);
    if (!model) {
        fprintf(stderr, "Failed to initialize model\n");
        return 1;
    }
    printf("Model initialized successfully!\n\n");
    
    // Example 1: Single forward pass
    printf("Example 1: Forward pass with input tokens\n");
    int input_tokens[] = {1, 5, 10, 15, 20};
    int n_tokens = 5;
    
    printf("Input tokens: ");
    for (int i = 0; i < n_tokens; i++) {
        printf("%d ", input_tokens[i]);
    }
    printf("\n");
    
    float* logits = (float*)malloc(config.vocab_size * sizeof(float));
    llm_forward(model, input_tokens, n_tokens, logits);
    
    printf("Top 5 predicted next tokens:\n");
    // Find top 5 tokens
    for (int k = 0; k < 5; k++) {
        int max_idx = 0;
        float max_val = logits[0];
        for (int i = 1; i < config.vocab_size; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = i;
            }
        }
        printf("  %d. Token %d (score: %.4f)\n", k+1, max_idx, max_val);
        logits[max_idx] = -1e9;  // Mark as used
    }
    printf("\n");
    
    // Example 2: Text generation
    printf("Example 2: Generate sequence from prompt\n");
    int prompt[] = {1, 2, 3};
    int prompt_len = 3;
    int max_new_tokens = 10;
    int* generated = (int*)malloc(max_new_tokens * sizeof(int));
    
    printf("Prompt tokens: ");
    for (int i = 0; i < prompt_len; i++) {
        printf("%d ", prompt[i]);
    }
    printf("\n");
    
    llm_generate(model, prompt, prompt_len, generated, max_new_tokens);
    
    printf("Generated tokens: ");
    for (int i = 0; i < max_new_tokens; i++) {
        printf("%d ", generated[i]);
    }
    printf("\n\n");
    
    // Example 3: Different prompt
    printf("Example 3: Generate from different prompt\n");
    int prompt2[] = {10, 20, 30, 40};
    int prompt2_len = 4;
    
    printf("Prompt tokens: ");
    for (int i = 0; i < prompt2_len; i++) {
        printf("%d ", prompt2[i]);
    }
    printf("\n");
    
    llm_generate(model, prompt2, prompt2_len, generated, max_new_tokens);
    
    printf("Generated tokens: ");
    for (int i = 0; i < max_new_tokens; i++) {
        printf("%d ", generated[i]);
    }
    printf("\n\n");
    
    // Cleanup
    free(logits);
    free(generated);
    llm_free(model);
    
    printf("Model freed. Demo complete!\n");
    return 0;
}
