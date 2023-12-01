import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class YourDatasetClass(Dataset):
    def __init__(self, max_len=100, vocab_size=10000, num_samples=1000):
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Replace this with your actual data loading logic
        input_sequence = torch.randint(0, self.vocab_size, (self.max_len,))
        target_sequence = torch.randint(0, self.vocab_size, (self.max_len,))
        return input_sequence, target_sequence

def collate_fn(batch):
    # Assuming each sample in the batch is a tuple (input_sequence, target_sequence)
    # Pad sequences to have the same length
    input_sequences, target_sequences = zip(*batch)
    padded_inputs = nn.utils.rnn.pad_sequence(input_sequences, batch_first=True, padding_value=0)
    padded_targets = nn.utils.rnn.pad_sequence(target_sequences, batch_first=True, padding_value=0)
    return padded_inputs, padded_targets

# Instantiate your GPT-2 model
model = GPT2(embed_size=256, heads=8, ff_hidden_size=512, num_layers=6, max_len=100, vocab_size=10000)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Instantiate your dataset and DataLoader
train_dataset = YourDatasetClass()
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Training loop
for epoch in range(10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Assuming your model expects a mask as well, create it accordingly
        mask = torch.ones_like(data).to(device)

        # Forward pass
        output = model(data, mask)
        loss = criterion(output.view(-1, model.vocab_size), target.view(-1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Save your trained model if needed
torch.save(model.state_dict(), 'gpt2_model.pth')
