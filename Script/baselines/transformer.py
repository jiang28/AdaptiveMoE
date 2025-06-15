# transformer baseline

import numpy as np
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_dim = model_dim
        self.input_linear = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(model_dim * 10, output_dim)  # Assuming sequence length is 10

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        x = self.input_linear(x) * torch.sqrt(torch.tensor(self.model_dim, dtype=torch.float32))
        x = self.positional_encoding(x)
        # Transformer expects input shape (sequence_length, batch_size, model_dim)
        x = x.permute(1, 0, 2)
        output = self.transformer_encoder(x)
        # Permute back to (batch_size, sequence_length, model_dim)
        output = output.permute(1, 0, 2)
        # Flatten the output for the final linear layer (batch_size, sequence_length * model_dim)
        output = output.reshape(output.size(0), -1)
        output = self.output_linear(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# Initialize model parameters
input_dim = X_train_tensor.shape[-1] # Features per time step
model_dim = 128 # Dimension of the model embeddings
num_heads = 8   # Number of attention heads
num_layers = 2  # Number of transformer encoder layers
output_dim = y_train_tensor.shape[-1] # Number of output features
num_epochs = 50
learning_rate = 0.001

# Create the Transformer model
transformer_model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(transformer_model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    transformer_model.train()
    total_loss = 0.0
    for batch_X, batch_y in train_dataloader:
        # batch_X shape: (batch_size, sequence_length, input_dim)
        # batch_y shape: (batch_size, output_dim)

        optimizer.zero_grad()
        outputs = transformer_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Print the loss for the current epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_dataloader):.4f}')

# Evaluation phase
transformer_model.eval()
predictions = []
actual_values = []

with torch.no_grad():
    for batch_X, batch_y in test_dataloader:
        outputs = transformer_model(batch_X)
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

print("Evaluation Results (Transformer):")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")