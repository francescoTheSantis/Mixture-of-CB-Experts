"""
We just test whether a linear layer can learn the nor function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_concepts.nn as pyc_nn
from torch_concepts.nn import functional as CF
from torch_concepts.semantic import ProductTNorm

def nor(x):
    return (1 - torch.max(x, dim=-1)[0] > 0.5).float()

class LinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = nn.Sigmoid()()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

if __name__ == "__main__":
    # set random seed for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Define the model
    input_size = 2
    output_size = 1
    model = LinearModel(input_size, output_size)

    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    # Generate some training data for the NOR function
    x_train = torch.rand([1000, 2], dtype=torch.float32)
    print(x_train[:10])
    y_train = nor(x_train).unsqueeze(1)  # Reshape to match output size
    print(y_train[:10])
    # input("Press Enter to continue...")

    # Train the model
    for epoch in range(10000):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 100 == 0:
            acc = ((outputs > 0.5).squeeze() == y_train.squeeze()).float().mean().item()
            print(f'Epoch [{epoch+1}/10000], '
                  f'Loss: {loss.item():.4f}, Accuracy: {acc:.4f}')

    # Print first ten predictions and labels on training data
    print("Model predictions")
    print(outputs[:10] > 0.5)
    print("Labels")
    print(y_train[:10])

    # Test the model
    with torch.no_grad():
        print("\nTesting the model\n")
        test_data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        predictions = model(test_data)
        y_test = nor(test_data)
        print("Predictions:")
        print(predictions > 0.5)
        print("Labels:")
        print(y_test)

        print(f"Accuracy: {((predictions > 0.5).squeeze() == y_test).float().mean().item():.4f}")

        print("Weights:")
        print(model.linear.weight)
        print("Bias:")
        print(model.linear.bias)
