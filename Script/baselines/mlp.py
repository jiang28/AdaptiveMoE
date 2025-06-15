import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Initialization
input_size = X_train.shape[2]
hidden_size = 64
output_size = y_train.shape[1]
num_experts = 3

# Define the base MLP expert (same as your provided code)
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the Gating Network
class GatingNetwork(nn.Module):
    def __init__(self, input_size, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, num_experts)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        return self.softmax(x)

class MoE(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_experts):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList([MLP(input_size, hidden_size, output_size) for _ in range(num_experts)])
        self.gating_network = GatingNetwork(input_size, num_experts)

    def forward(self, x):
        gating_outputs = self.gating_network(x)
        expert_outputs = [expert(x) for expert in self.experts]

        # Transpose gating outputs to match expert outputs
        gating_outputs_expanded = gating_outputs.unsqueeze(-1)

        # Expand dimensions of expert outputs for broadcasting
        expert_outputs_stacked = torch.stack(expert_outputs, dim=2)

        # Element-wise multiplication
        combined_output = torch.sum(gating_outputs_expanded * expert_outputs_stacked, dim=1)

        return combined_output

# Define the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model, loss function, and optimizer
model = MoE(input_size, hidden_size, output_size, num_experts).to(device)
criterion = nn.L1Loss()  # MAE loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop with validation
best_val_loss = float('inf')
best_epoch = 0
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (data, targets) in enumerate(train_dataloader):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        # Aggregate predictions across the expert dimension
        outputs_aggregated = torch.mean(outputs, dim=1)
        loss = criterion(outputs_aggregated, targets)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(train_dataloader)

    # Validation
    model.eval()  # Set the model to evaluation mode
    total_val_loss = 0
    total_mse = 0
    with torch.no_grad():
        for data, targets in val_dataloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            # Aggregate predictions across the expert dimension
            outputs_aggregated = torch.mean(outputs, dim=1)
            val_loss = criterion(outputs_aggregated, targets)
            total_val_loss += val_loss.item()
            # Calculate MSE
            total_mse += mean_squared_error(targets.cpu().numpy(), outputs_aggregated.cpu().numpy())
    avg_val_loss = total_val_loss / len(val_dataloader)
    avg_val_mse = total_mse / len(val_dataloader)

    print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_loss:.4f} - Validation Loss: {avg_val_loss:.4f} - Validation MSE: {avg_val_mse:.4f}")

    # Check for improvement in validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch
        # Optionally, save the model checkpoint

    # Optionally implement early stopping
    # if early_stopping_criteria:
    #     break

print(f"Best Validation Loss: {best_val_loss:.4f} at Epoch {best_epoch+1}")

# Testing loop
model.eval()  # Set the model to evaluation mode
total_mae = 0
total_mse = 0
with torch.no_grad():
    for data, targets in test_dataloader:
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        # Aggregate predictions across the expert dimension
        outputs_aggregated = torch.mean(outputs, dim=1)
        total_mae += mean_absolute_error(targets.cpu().numpy(), outputs_aggregated.cpu().numpy())
        total_mse += mean_squared_error(targets.cpu().numpy(), outputs_aggregated.cpu().numpy())

avg_mae = total_mae / len(test_dataloader)
avg_mse = total_mse / len(test_dataloader)

print(f"Test MAE: {avg_mae:.4f}")
print(f"Test MSE: {avg_mse:.4f}")
print(f"Test RMSE: {np.sqrt(avg_mse):.4f}")