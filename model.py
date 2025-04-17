import torch
import torch.nn as nn
import math

class InputEmbeddingBlock(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncodingBlock(nn.Module):
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Matrix shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Vector shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # Paper formula
        #div_term = torch.exp(torch.arange(0, d_model, 2).float()) * (-math.log(10000.0) / d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class AddAndNormalizationBlock(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x):
        # (Batch, seq_len, d_model) > (batch, seq_len, d_ff) > (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h

        # Make sure the model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h

        # Matrix q, k, v
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]

        query = self.w_q(q) # (Batch, seq_len, d_model) > (Batch, seq_len, d_model)
        key = self.w_k(k)
        value = self.w_v(v)

        # (Batch, seq_len, d_model) > (Batch, seq_len, h, d_k) > (Batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_w = torch.softmax(scores, dim = -1)
        attention_w = self.dropout(attention_w)

        output = torch.matmul(attention_w, value)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.w_o(output)

class Connection(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout
        self.norm = AddAndNormalizationBlock()

    def forward(self, x, previous_layer):
        return x + self.dropout(previous_layer(self.norm(x)))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttentionBlock(d_model, h, dropout)
        self.ff = FeedForwardBlock(d_model, d_ff, dropout)
        self.norm1 = AddAndNormalizationBlock(1e-6)
        self.norm2 = AddAndNormalizationBlock(1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.attn(x, x, x, mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)

        return x

class DNATransformer(nn.Module):
    def __init__(self, d_model, vocab_size, seq_len, dropout, h, d_ff, n_layers):
        super().__init__()
        
        # Input Embeddings
        self.embedding = InputEmbeddingBlock(d_model, vocab_size)
        
        # Positional Embeddings
        self.positional = PositionalEncodingBlock(d_model, seq_len, dropout)
        
        # Create Encoder layers by the number of heads. Example h = 6, create 6 encoders
        # Sublayer 1 = Multihead Attention Block + Add and Normalizing Block
        # Sublayer 2 = Feed Forward Block + Add and Normalizing Block
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, h, d_ff, dropout) for _ in range(n_layers)
        ])

        # FOr binary classification task
        self.classifier = nn.Linear(d_model, 1)  # Binary classification

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional(x)
        for layer in self.layers:
            x = layer(x, mask)

        # Mean pooling over sequence
        x = x.mean(dim=1)  # shape: (batch_size, d_model)
        logits = self.classifier(x)
        return logits

        # Logits = [a, b, c, ...]
        # Return the probabilities