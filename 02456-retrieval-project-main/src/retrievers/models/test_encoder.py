import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from math import log
import matplotlib.pyplot as plt
import seaborn as sns
import encoder
import numpy as np

def plot_positional_encoding(x, d_model, max_len=1000, dropout=0.0):
    """ Plot the positional encodings as a 2D heatmap """
    pe = encoder.PositionalEncoding(d_model, dropout, max_len=max_len)
    y = pe(x)

    plt.figure(figsize=(10, 4))
    sns.heatmap(y[0].numpy().T, cmap="viridis")  # 2-D: d_model x seq_len
    plt.ylabel("Dimension")
    plt.xlabel("Position in the sentence")
    plt.gca().invert_yaxis()
    plt.show()

def plot_attention_map(attention_map, queries_labels, keys_labels, print_values:bool=False, ax=None, color_bar:bool=True):
    """Plot the attention weights as a 2D heatmap"""
    if ax is None:
        fig, ax = plt.subplots(figsize = (10,6), dpi=300)
    else:
        fig = plt.gcf()
    im = ax.imshow(attention_map, cmap=sns.color_palette("viridis", as_cmap=True))
    ax.grid(False)
    ax.set_ylabel("$\mathbf{Q}$")
    ax.set_xlabel("$\mathbf{K}$")
    ax.set_yticks(np.arange(len(queries_labels)))
    ax.set_yticklabels(queries_labels)
    ax.set_xticks(np.arange(len(keys_labels)))
    ax.set_xticklabels(keys_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    if print_values:
        for i in range(len(queries_labels)):
            for j in range(len(keys_labels)):
                text = ax.text(j, i, f"{attention_map[i, j]:.2f}",
                            ha="center", va="center", color="w")

    if color_bar:
      fig.colorbar(im, fraction=0.02, pad=0.04)
    fig.tight_layout()
    plt.show()

def test_encoder():
    ## Generate a random tensor of shape (batch_size, sequence_length, emb_dim)=(2, 1, 128) to test the encoder
    
    batch = torch.rand(64, 1, 128)
    model = encoder.TransformerEncoder(n_layers=6, n_heads=8, emb_dim=128, hidden_dim=128, feed_forward_dim=4)
    output = model(batch)
    print(f"Output shape of the encoder: {output.shape}")

    # Test positional encoding map
    # let's assume we havee a sequence of length 11 and embedding of length 128
    # how their positional encoding looks like
    plot_positional_encoding(torch.zeros(1, 512, 128), d_model=128, max_len=1000, dropout=0.0)
    plot_positional_encoding(torch.zeros(1, 512, 128), d_model=128, max_len=1000, dropout=0.2)
    plot_positional_encoding(batch.transpose(0,1), d_model=128, max_len=1000, dropout=0.2)
    

    # Test the parallel neural network layer
    # As we can see from the printing value, because the input shape [B, L, d_model] is preserved
    #  N identical Transformer encoder blocks can be stacked sequentially.
    # Usually for Lineary Layers we would have to flatten the input, but here we can keep the shape of the input and output the same.
    pnn = encoder.ParallelNeuralNetwork(emb_dim=128, hidden_dim=128, feed_forward_dim=4, dropout=0.1)
    output_pnn = pnn(batch)
    print(f"Output shape of the parallel neural network layer: {output_pnn.shape}")
    assert output_pnn.shape == batch.shape

def test_attention():
    # Test the attention function
    # Q, V and K is a sequence of length 8 and embedding of length 128
    query = torch.rand(16, 128)
    key = torch.rand(16, 128)
    value = torch.rand(16, 128)
    output_attention, p_attn = encoder.attention(query, key, value)
    print(f"Output shape of the attention function: {output_attention.shape}")
    print(f"Attention weights shape: {p_attn.shape}")
    plot_attention_map(p_attn.numpy(), queries_labels=[""], keys_labels=[""], print_values=False, color_bar=True)

def test_layernorm():
    x = torch.rand(4, 10, 128) * 5 + 3
    layer_norm = encoder.LayerNorm(128)
    y = layer_norm(x)
    print(f"Output shape of the layer norm: {y.shape}")
    print(f"Max mean of the normalized embeddings (should be close to 0): {y.mean(-1).abs().max():.2e}")
    print(f"Mean std of the normalized embeddings (should be close to 1): {y.std(-1).mean():.4f}")
    assert y.shape == x.shape
    assert y.mean(-1).abs().max() < 1e-4
    assert torch.allclose(y.std(-1), torch.ones(4, 10), atol=1e-3)

def test_encoder_layer():
    # One encoder layer must keep the shape [B, L, d_model] in the forward pass
    x = torch.rand(8, 10, 128)
    layer = encoder.EncoderLayer(n_heads=8, emb_dim=128, hidden_dim=128, feed_forward_dim=4, dropout=0.1)
    output_layer = layer(x)
    print(f"Output shape of the encoder layer: {output_layer.shape}")
    assert output_layer.shape == x.shape
 
    model = encoder.TransformerEncoder(n_layers=6, n_heads=8, emb_dim=128, hidden_dim=128, feed_forward_dim=4)
    output = model(x)
    print(f"Output shape of the encoder with sequence length 10: {output.shape}")
    assert output.shape == x.shape


if __name__ == '__main__':
    test_encoder()
    test_attention()
    test_layernorm()
    test_encoder_layer()