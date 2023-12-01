import torch
import torch.nn as nn

class GroupQueryAttention(nn.Module):
    def __init__(self, embed_size, heads, group_size=4):
        super(GroupQueryAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.group_size = group_size

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # Group queries
        queries = queries.reshape(N, query_len, self.heads, self.group_size, -1)
        queries = queries.mean(dim=-2)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Fix the mask dimension
        mask = mask.unsqueeze(1).unsqueeze(1)  # Add dimensions for heads and query_len
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.nn.functional.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


# Update the MultiHeadAttention class to use GroupQueryAttention
class GPT2Layer(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_size, dropout, group_size=4):
        super(GPT2Layer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.attn = GroupQueryAttention(embed_size, heads, group_size)  # Use GroupQueryAttention
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
    def __init__(self, embed_size, heads, ff_hidden_size, num_layers, max_len, vocab_size, group_size=4):
        super(GPT2, self).__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.token_embeddings = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(max_len, embed_size)
        self.layers = nn.ModuleList(
            [
                GPT2Layer(embed_size, heads, ff_hidden_size, dropout=0.1, group_size=group_size)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x, mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(x.device)
        x = self.token_embeddings(x) + self.positional_embedding(positions)

        for layer in self.layers:
            x = layer(x, x, x, mask)

        x = self.fc_out(x)
        return x
