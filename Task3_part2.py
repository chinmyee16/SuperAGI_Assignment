import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel

from transformers import GPT2Tokenizer

# Define your GPT-2 model class here (the one you provided in the previous message)

# Define your dataset and dataloader
# For demonstration purposes, let's create a simple dataset
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, vocab_size=10000, seq_len=50, num_samples=1000):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Create the dataset and dataloader
dataset = DummyDataset()
train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Define GPT-2 model
# Replace this line with the instantiation of your GPT-2 model
model = GPT2(embed_size=256, heads=8, ff_hidden_size=512, num_layers=6, max_len=100, vocab_size=10000)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0.0

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        mask = torch.ones_like(data)

        # Forward pass
        output = model(data, mask)
        loss = criterion(output.view(-1, model.vocab_size), data.view(-1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Average Loss: {average_loss}")

# Save the trained model
torch.save(model.state_dict(), "gpt2_model.pth")
