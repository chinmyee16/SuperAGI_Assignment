import torch
import torch.nn as nn
from torch.nn import functional as F

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, embed_size, max_len):
        super(RotaryPositionalEmbedding, self).__init__()
        self.embed_size = embed_size
        self.positional_embeddings = nn.Parameter(torch.zeros(max_len, embed_size))

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).float()
        sinusoids = torch.ger(positions, torch.exp(torch.arange(0, self.embed_size, 2, device=x.device) * -(math.log(10000.0) / self.embed_size)))

        pos_emb = torch.cat([sinusoids.sin(), sinusoids.cos()], dim=-1)
        pos_emb = pos_emb.unsqueeze(0).expand(x.size(0), -1, -1)

        return pos_emb

class GPT2Layer(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_size, dropout):
        super(GPT2Layer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.attn = MultiHeadAttention(embed_size, heads)
        self.ff = FeedForward(embed_size, ff_hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        x = self.attn(value, key, query, mask)

        # Add skip connection and pass through normalization
        x = self.dropout(x)
        x = self.norm1(x + query)

        # Feed forward
        ff_out = self.ff(x)

        # Add skip connection and pass through normalization
        ff_out = self.dropout(ff_out)
        x = self.norm2(ff_out + x)
        return x

class GPT2(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_size, num_layers, max_len, vocab_size):
        super(GPT2, self).__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.token_embeddings = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = RotaryPositionalEmbedding(embed_size, max_len)  # Change here
        self.layers = nn.ModuleList(
            [
                GPT2Layer(embed_size, heads, ff_hidden_size, dropout=0.1)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x, mask):
        N, seq_len = x.shape
        x = self.token_embeddings(x) + self.positional_embedding(x)  # Change here

        # ... (unchanged code)
        # x = self.token_embeddings(x) + self.positional_embedding(positions)

        for layer in self.layers:
            x = layer(x, x, x, mask)

        x = self.fc_out(x)
        return x

        x = self.fc_out(x)
        return x
