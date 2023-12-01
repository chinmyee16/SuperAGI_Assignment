import torch
import torch.nn as nn
from torch.nn import functional as F

class SlidingWindowAttention(nn.Module):
    def __init__(self, embed_size, heads, window_size):
        super(SlidingWindowAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.window_size = window_size

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

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Apply sliding window attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        mask = mask.unsqueeze(1).unsqueeze(1)  # Add dimensions for heads and query_len

        # Implement sliding window mask
        sliding_mask = torch.zeros((1, 1, query_len, key_len), device=mask.device)
        for i in range(query_len):
            start = max(0, i - self.window_size // 2)
            end = min(key_len, i + self.window_size // 2 + 1)
            sliding_mask[:, :, i, start:end] = 1

        # Apply sliding window mask
        mask = mask * sliding_mask
        energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Compute attention
        attention = torch.nn.functional.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

# Modify the GPT2Layer to use the SlidingWindowAttention
class GPT2Layer(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_size, dropout, window_size):
        super(GPT2Layer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.attn = SlidingWindowAttention(embed_size, heads, window_size)  # Change here
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
    def __init__(self, embed_size, heads, ff_hidden_size, num_layers, max_len, vocab_size, window_size):
        super(GPT2, self).__init__()
        # ... (unchanged code)
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.token_embeddings = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(max_len, embed_size)
        self.layers = nn.ModuleList(
            [
                GPT2Layer(embed_size, heads, ff_hidden_size, dropout=0.1, window_size=window_size)  # Change here
                for _ in range(num_layers)
            ]
        )
        # ... (unchanged code)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x, mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(x.device)
        x = self.token_embeddings(x) + self.positional_embedding(positions)

        for layer in self.layers:
            x = layer(x, x, x, mask)

        x = self.fc_out(x)
        return x

    def generate_text(self, tokenizer, prompt, mask=None):
        self.eval()  # Set the model to evaluation mode
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Create a default mask if not provided
        if mask is None:
            mask = torch.ones_like(input_ids)

        # Generate output
        with torch.no_grad():
            output = self(input_ids, mask=mask)

        # Fix the decoding step to handle potential None tokens
        decoded_output = tokenizer.decode(output[0][0], skip_special_tokens=True)

        # Skip None tokens in the decoded output
        filtered_tokens = [token for token in decoded_output if token is not None]

        # Concatenate the non-None tokens
        generated_text = "".join(filtered_tokens)
        return generated_text

# Instantiate your GPT-2 model with the required parameters
# model = GPT2(embed_size=256, heads=8, ff_hidden_size=512, num_layers=6, max_len=100, vocab_size=10000)



# Instantiate your GPT-2 model with the required parameters
model = GPT2(embed_size=256, heads=8, ff_hidden_size=512, num_layers=6, max_len=100, vocab_size=10000, window_size=5)

# Sample input for testing
sample_input = torch.randint(0, 10000, (1, 10))  # Batch size of 1, sequence length of 10
mask = torch.ones_like(sample_input)

# Uncomment the following line to get the output from your model
output = model(sample_input, mask)
print("GPT-2 Output Shape:", output.shape)

# Example for text generation
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # Replace with the actual tokenizer you're using
# prompt = "Once upon a time"
# generated_text = model.generate_text(tokenizer, prompt)
# print("Generated Text:", generated_text)
