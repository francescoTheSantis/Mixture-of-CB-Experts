# Basic usage sketch
from kan.KANLayer import *
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Example: synthetic regression data
X = torch.linspace(-1, 1, steps=200).unsqueeze(1)
y = (X ** 3) + 0.1 * torch.randn_like(X)

# Set up dataset
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Instantiate a simple KAN model
params = {
        'in_dim': [1],
        'out_dim': [1],
        'num': 5,
        'k': 3,
}

model = KANLayer(**params)

# Choose optimizer & loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(1000):
    for xb, yb in loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: loss = {loss.item():.4f}')

# After training, extract symbolic form (if supported)
equation = model.symbolize()  # or similar
print("Learned equation:", equation)