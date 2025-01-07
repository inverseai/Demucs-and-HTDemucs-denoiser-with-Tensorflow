import numpy as np

# Example dimensions for demonstration (e.g., for a multi-head attention scenario)
embed_dim = 2
num_heads = 2
seq_length = 5
batch_size = 1

# Generate random weights for Q, K, V, and projection layers in PyTorch style (out_features, in_features)
np.random.seed(42)  # Ensure reproducibility
weights_q = np.random.rand(embed_dim, embed_dim).astype(np.float32)
weights_k = np.random.rand(embed_dim, embed_dim).astype(np.float32)
weights_v = np.random.rand(embed_dim, embed_dim).astype(np.float32)
weights_proj = np.random.rand(embed_dim, embed_dim).astype(np.float32)

# Save weights
np.savez('initial_weights.npz', weights_q=weights_q, weights_k=weights_k, weights_v=weights_v, weights_proj=weights_proj)

# Generate a random input batch
input_data = np.random.rand(batch_size, seq_length, embed_dim).astype(np.float32)

# Save input data
np.save('initial_input.npy', input_data)
