#LSTM baseline

import numpy as np
# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Initialize model parameters
input_size = X_train_tensor.shape[-1] # Features per time step
hidden_size = 128
num_layers = 2
output_size = y_train_tensor.shape[-1] # Number of output features
num_epochs = 50
learning_rate = 0.001

# Create the LSTM model
lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    lstm_model.train()
    total_loss = 0.0
    for batch_X, batch_y in train_dataloader:
        # batch_X shape: (batch_size, sequence_length, input_size)
        # batch_y shape: (batch_size, output_size)

        optimizer.zero_grad()
        outputs = lstm_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Print the loss for the current epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_dataloader):.4f}')

# Evaluation phase
lstm_model.eval()
predictions = []
actual_values = []

with torch.no_grad():
    for batch_X, batch_y in test_dataloader:
        outputs = lstm_model(batch_X)
        predictions.extend(outputs.numpy())
        actual_values.extend(batch_y.numpy())

predictions = np.array(predictions)
actual_values = np.array(actual_values)

print("Shape of actual_values:", actual_values.shape)
print("Shape of predictions:", predictions.shape)

# Calculate evaluation metrics
mae = mean_absolute_error(actual_values, predictions)
mse = mean_squared_error(actual_values, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(actual_values, predictions)
mape = modified_mape(actual_values, predictions) # Assuming modified_mape is defined earlier

print("Evaluation Results (LSTM):")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")