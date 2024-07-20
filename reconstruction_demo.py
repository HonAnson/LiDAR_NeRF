import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


""" Demonstrate how a neural network fits a sigmoid function"""


# Define the neural network model
class SigmoidFittingModel(nn.Module):
    def __init__(self):
        super(SigmoidFittingModel, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # Input layer to hidden layer
        self.fc2 = nn.Linear(10, 1)  # Hidden layer to output layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation function

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = self.sigmoid(self.fc2(x))
        x = self.fc2(x)
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
optimizer = optim.Adam(model.parameters(), lr=0.01)


def render_ray(outputs):
    t = torch.tensor(np.linspace(-10, 10, 100))
    delta = torch.cat((t[1:] - t[:-1], torch.tensor([1e10])), -1)
    mid = (t[:-1] + t[1:]) / 2.
    sigma = outputs * (1 - outputs)
    alpha = 1 - torch.exp(-sigma * delta)  # [nb_bins]
    breakpoint()
    accumulated_transmittance = torch.cumprod(alpha, 1)
    # weights = compute_accumulated_transmittance(1 - alpha)





    rendered_value = 0
    return rendered_value



# Training loop
num_epochs = 5000
for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    sigmoid_ = nn.Sigmoid()
    outputs = sigmoid_(model(x_train))
    # use output to conduct reconstruction
    rendered_value = render_ray(y_train)



    loss = (outputs, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plotting the results
model.eval()
predicted_tensor = sigmoid_(model(x_train))
predicted_np = predicted_tensor.detach().numpy()

# try to also plot the gradient 
occupancy = predicted_tensor* (1 -predicted_tensor)
occupancy_np = occupancy.detach().numpy()




plt.figure(figsize=(10, 6))
plt.plot(x_data, occupancy_np, 'ro', label='Original data')
plt.plot(x_data, predicted_np, 'b-', label='Fitted line')
plt.legend()
plt.show()