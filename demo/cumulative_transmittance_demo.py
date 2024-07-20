import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
""" Demonstrate prediction of model optimizing from predicting transmittance"""
# Define the neural network model
class SigmoidFittingModel(nn.Module):
    def __init__(self):
        super(SigmoidFittingModel, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # Input layer to hidden layer
        self.fc2 = nn.Linear(10, 1)  # Hidden layer to output layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation function

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Generate synthetic data
def generate_data(num_points=100):
    x = np.linspace(-10, 10, num_points)
    y = 1 / (1 + np.exp(-x))  # Sigmoid function
    return x, y

# Prepare data
x_data, y_data = generate_data()
x_train = torch.tensor(x_data, dtype=torch.float32).view(-1, 1)
y_train = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)

# Initialize the model, loss function, and optimizer
model = SigmoidFittingModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 5000
for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plotting the results
model.eval()
output = model(x_train)
predicted = output.detach().numpy()

plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data, 'ro', label='Original data')
plt.plot(x_data, predicted, 'b-', label='Fitted line')
plt.legend()
plt.show()





def render_ray(outputs):
    t = torch.tensor(np.linspace(-10, 10, 100))
    delta = torch.cat((t[1:] - t[:-1], torch.tensor([100])), -1)
    mid = (t[:-1] + t[1:]) / 2.
    h = outputs * (1 - outputs)     # h is the termination probability
    h = rearrange(h, 'b 1 -> b')
    rendered_value = torch.sum(h*delta*t)
    return rendered_value

temp = render_ray(output)
print(temp)