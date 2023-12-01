import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

def train_step(model, data, target, criterion, optimizer, device):
    model.train()
    data, target = data.to(device), target.to(device)

    # Create attention mask (you need to adjust this based on your model requirements)
    mask = torch.ones_like(data)

    # Ensure data is of type torch.long
    data = data.long()

    # Forward pass
    output = model(data, mask)
    loss = criterion(output, target)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def validate_step(model, data, target, criterion, device):
    model.eval()
    with torch.no_grad():
        data, target = data.to(device), target.to(device)

        # Create attention mask for validation
        val_mask = torch.ones_like(data)

        val_output = model(data, val_mask)
        val_loss = criterion(val_output, target)

    return val_loss.item()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assume your model is already loaded externally
# Assume your model is already loaded externally
model = GPT2(embed_size=256, heads=8, ff_hidden_size=512, num_layers=6, max_len=100, vocab_size=10000).to(device)


# Sample data and target (modify this based on your dataset)
# Sample data and target (modify this based on your dataset)
train_data, train_target = torch.randn(32, 50).to(device), torch.randint(0, 10000, (32,), dtype=torch.long).to(device)
val_data, val_target = torch.randn(16, 50).to(device), torch.randint(0, 10000, (16,), dtype=torch.long).to(device)


# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# DataLoader setup (modify this based on your dataset)
# DataLoader setup (modify this based on your dataset)
import torch
from torch.utils.data import Dataset, DataLoader

# Define a dummy dataset (replace this with your actual dataset)
class RandomDataset(Dataset):
    def __init__(self, num_samples, input_size, target_size):
        self.data = torch.randn(num_samples, input_size)
        self.targets = torch.randn(num_samples, target_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Instantiate the dummy dataset
train_dataset = RandomDataset(num_samples=1000, input_size=32, target_size=10)
val_dataset = RandomDataset(num_samples=200, input_size=32, target_size=10)

# DataLoader setup (modify this based on your dataset)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False)



# Training loop
for epoch in range(10):
    for batch_data, batch_target in train_loader:
        train_loss = train_step(model, batch_data, batch_target, criterion, optimizer, device)
        print(f'Epoch {epoch}, Training Loss: {train_loss}')

    # Validation (example with random validation data)
    if epoch % 2 == 0:
        for batch_val_data, batch_val_target in val_loader:
            val_loss = validate_step(model, batch_val_data, batch_val_target, criterion, device)
            print(f'Epoch {epoch}, Validation Loss: {val_loss}')

# Distributed Data Parallel (DDP) Training Loop
# Ensure you run this script using torch.distributed.launch
# Example: python -m torch.distributed.launch --nproc_per_node=2 your_script.py
if torch.cuda.device_count() > 1:
    model = DistributedDataParallel(model)
    # Additional DDP-specific configuration goes here

    # Training loop (similar to the single GPU loop)
    for epoch in range(10):
        for batch_data, batch_target in train_loader:
            train_loss = train_step(model, batch_data, batch_target, criterion, optimizer, device)
            print(f'Epoch {epoch}, Training Loss: {train_loss}')
