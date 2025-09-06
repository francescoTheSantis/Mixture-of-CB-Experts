# Basic usage sketch
from kan import KAN
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Example: synthetic regression data
X = torch.linspace(-1, 1, steps=2000).unsqueeze(1)
Z = torch.linspace(-1, 1, steps=2000).unsqueeze(1)
#y = (X ** 3) + 0.01 * torch.randn_like(X) + (Z ** 2) + 0.01 * torch.randn_like(Z)
y = (X ** 3) + (Z ** 2) 

# Set up dataset
dataset = TensorDataset(X, Z, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

## Tutorial for KAN model
input_size = X.shape[1] + Z.shape[1]
output_size = y.shape[1]

# Instantiate a simple KAN model
params = {
        'width': [2, 3, 1],
        'grid':5, 
        'k':4, 
        'auto_save': False,
}

model = KAN(**params)

# Choose optimizer & loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(200):
    for xb, zb, yb in loader:
        optimizer.zero_grad()
        preds = model(torch.cat([xb, zb], dim=1))
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
    if epoch % 50 == 0:
        print(f'Epoch {epoch}: loss = {loss.item():.4f}')

# Prune model
# model.prune()

# lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
lib = ['x', 'x^2', 'x^3']
model.auto_symbolic(lib=lib)

# Show discovered symbolic formulas
print("\nDiscovered symbolic formula(s):")
expr = model.symbolic_formula()[0]  
print(expr)
