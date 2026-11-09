import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from math import log

## from " Attention is all you need "
# The encoder is composed of a stack of N = 6 identical layers. Each layer has two
# sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-
# wise fully connected feed-forward network. We employ a residual connection [11] around each of
# the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is
# LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer
# itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding
# layers, produce outputs of dimension dmodel = 512.
# The embeddings are trained through contrastive learning during the training phase

class TransformerEncoder(nn.Module):
    def __init__(self, n_layers: int, n_heads: int, emb_dim: int, hidden_dim: int, feed_forward_dim: int, pooling_type: str = 'mean', dropout: float = 0.1):
        super(TransformerEncoder, self).__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.emb_dim =    emb_dim
        self.hidden_dim = hidden_dim
        self.feed_forward_dim = feed_forward_dim
        self.dropout = dropout
        self.pooling_type = pooling_type
        self.setup_model()

    def setup_model(self):
        """
        Instantiate the model.
        """
        self.layers = nn.ModuleList([EncoderLayer(self.n_heads, self.emb_dim, self.hidden_dim, self.feed_forward_dim, self.dropout) for _ in range(self.n_layers)])

    # Note: I modified this, so when initializing the transformerEncoder we need to pass cfg.model.pooling_type as pooling argument.
    def pool(self, x):
        """
        Given a batch_size x sequence_length x emb_dim tensor, pool the input along the sequence_length dimension 
        Returns: A pooled vector of shape batch_size x emb_dim
        """
        if self.pooling_type == 'cls':
            return x[:,0,:]
        
        elif self.pooling_type == 'max':
            # returns a tensor. If both indeces and values are needed as output, remove [0]
            return torch.max(x, dim=1)[0]
        
        elif self.pooling_type == 'mean':
            return torch.mean(x,dim=1)

    # Assume that x is already embedded
    def forward(self, x: torch.Tensor, mask: torch.Tensor =None) -> torch.tensor:
        """
        Run a forward pass on the model
        """
        for layer in self.layers:
            x = layer(x)
        return x

class LayerNorm(nn.Module):
    """LayerNorm Layer. Normalizes each embedding in the batch separately"""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# Takes as input the query, key and values that are
class MultiHeadAttentionLayer(nn.Module):
    """
    Multi Head Attention Layer. Takes as input query, key and value and computes in parallel the attention on a subset of the embedding dimension
    """
    def __init__(self, n_heads: int, emb_dim: int,dropout=0.1):
        super(MultiHeadAttentionLayer, self).__init__()
        # remember that the shape of the input is [batch_size, embedding_dimension]
        assert emb_dim % n_heads == 0
        self.emb_dim_k = emb_dim // n_heads
        self.n_heads = n_heads
        # Linear layers used to compute the query, keys, values and attention output
        # The output of the linear layer is shaped [*, emb_dim]
        self.linear_q = nn.Linear(emb_dim, emb_dim)
        self.linear_v = nn.Linear(emb_dim, emb_dim)
        self.linear_k = nn.Linear(emb_dim, emb_dim)
        self.linear_y = nn.Linear(emb_dim, emb_dim)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask = None):
        batch_num = query.size(0)
        # The output of the linear layers giving the query, key and values need to be shaped [batch_size, heads, embedding_size / number_heads]
        query = self.linear_q(query).view(batch_num, -1, self.n_heads, self.emb_dim_k).transpose(1, 2)
        key = self.linear_k(key).view(batch_num, -1, self.n_heads, self.emb_dim_k).transpose(1, 2)
        value = self.linear_v(value).view(batch_num, -1, self.n_heads, self.emb_dim_k).transpose(1, 2)
        if mask is not None:
            mask = mask[:, None, None, :]
        
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        
        # Concatenation step
        x = self.linear_y(x.transpose(1,2).reshape(batch_num, -1, self.n_heads*self.emb_dim_k))
        return x


class EncoderLayer(nn.Module):
    """
    Encoder Layer. Composed by a multihead self-attention, LayerNorm and a FeedForward Layer, with residual connections.
    """
    def __init__(self, n_heads: int, emb_dim: int, hidden_dim: int, feed_forward_dim: int, dropout: float):
        super(EncoderLayer, self).__init__()
        self.layerNorm1 = LayerNorm(emb_dim)
        self.layerNorm2 = LayerNorm(emb_dim)
        self.multiHead = MultiHeadAttentionLayer(n_heads, emb_dim, dropout=dropout)
        self.pnn = ParallelNeuralNetwork(emb_dim, hidden_dim, feed_forward_dim, dropout = dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        y = self.multiHead(x, x, x, mask)
        x = self.layerNorm1(self.dropout(y) + x)
        y = self.pnn(x)
        x = self.layerNorm2(self.dropout(y)+x)
        return x

        

# Parallel neural networks
class ParallelNeuralNetwork(nn.Module):
    """
    Parallel Neural Network Layer used in the encoder architecture. Takes as input all embedded sequences in parallel
    """
    def __init__(self, emb_dim: int, hidden_dim: int, feed_forward_dim: int,  dropout: float):
        super(ParallelNeuralNetwork, self).__init__()
        self.W_1 = nn.Linear(emb_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(feed_forward_dim)])
        self.W_N = nn.Linear(hidden_dim, emb_dim)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        x = F.relu(self.W_1(x))
        x = self.dropout(x)
        
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
            
        x = self.W_N(x)
        return x

def attention(query, key, value, dropout=None, mask =None): # General attention or just for the encoder? otherwise we need the mask
    """
    Compute the Scaled Dot Product attention
    """

    d_k = query.size(-1)
    # Normalized attention
    scores = torch.matmul(query, key.transpose(-2, -1)) / sqrt(d_k)
    if mask is not None:
        # set the scores of the masked elements to a very low value so that they are not considered in the softmax
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding implementation
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
       
        positional_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(log(10000.0) / d_model))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer('pe', positional_encoding)
        
    # register_buffer tells PyTorch to not update the positional encoding during training
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

