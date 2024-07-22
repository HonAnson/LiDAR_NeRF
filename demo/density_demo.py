import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange

""" Demosntrate model optimizing by using reconstrucition loss to optimize density prediction"""
# NOTE: sometimes it stuck at local minimal because I did not use batch normalization


# Define the neural network model
class DensityFittingModel(nn.Module):
    def __init__(self):
        super(DensityFittingModel, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # Input layer to hidden layer
        self.fc2 = nn.Linear(10, 1)  # Hidden layer to output layer
        self.fc3 = nn.Linear(10,10)


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc3(x))
        # x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc2(x))
        return x

def render_cumtran(density):
    density = rearrange(density, 'a 1 -> a')
    t = torch.linspace(0, 20, 100)
    delta = torch.cat((t[1:] - t[:-1], torch.tensor([1e2])), -1)
    local_transmittance = torch.exp(-density * delta)
    cumulative_transmittance = torch.cumprod(local_transmittance, 0)
    return cumulative_transmittance, local_transmittance


def render_h(cum_tran, local_tran):
    h = (cum_tran * (1 - local_tran))
    return h

def render_ray(h):
    t = torch.linspace(0, 20, 100)
    E_depth = (h*t).sum()
    return E_depth


# Generate synthetic data
def generate_data(num_points=100):
    x = np.linspace(0, 20, num_points)
    y = 1 - (1 / (1 + np.exp(-(x-10))))  # Mirrored Sigmoid function
    return x, y

# Prepare data
x_data, y_data = generate_data()
x_train = torch.tensor(x_data, dtype=torch.float32).view(-1, 1)
target_cumtrans = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)
target_cumtrans = rearrange(target_cumtrans, 'a 1 -> a')
target_depth = torch.tensor(10, dtype = torch.float32)
target_h = (target_cumtrans ) * (1 - target_cumtrans) * 0.2
# Initialize the model, loss function, and optimizer
model = DensityFittingModel()
MSE_loss= nn.MSELoss()
KL_loss = nn.KLDivLoss()
BCE_loss = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 10000
for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    predicted_density = model(x_train)
    rendered_cumtrans, rendered_localtrans = render_cumtran(predicted_density)
    rendered_h = render_h(rendered_cumtrans, rendered_localtrans)
    rendered_depth = render_ray(rendered_h)

    loss_depth = MSE_loss(rendered_depth, target_depth)
    # loss_h = MSE_loss(rendered_h, target_h)
    loss_h = KL_loss(rendered_h, target_h)
    loss_cumtran = MSE_loss(rendered_cumtrans, target_cumtrans)
    # loss_togehter = loss_depth + loss_cumtran + loss_h
    # loss_togehter = loss_depth
    # loss_togehter = loss_cumtran + loss_h
    loss_togehter = loss_h

    # Backward pass and optimization
    optimizer.zero_grad()
    loss_togehter.backward()
    optimizer.step()
    
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss_h: {loss_h.item():.4f}, Loss_depth: {loss_depth.item():.4f}, Loss_cumtran: {loss_cumtran.item():.4f}')
        # print(predicted_density)

# Plotting the results
model.eval()
output = model(x_train)
predicted_cumtran, predicted_localtran = render_cumtran(output)
predicted_h = render_h(predicted_cumtran, predicted_localtran)

output_np = output.detach().numpy()
predicted_cumtran_np = predicted_cumtran.detach().numpy()
predicted_h_np = predicted_h.detach().numpy()
target_h_np = target_h.detach().numpy()
plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data,  label='Targeted Cumulative Transmittance')
plt.plot(x_data, output_np,  label='Predicted Density')
plt.plot(x_data, target_h_np,  label='Target Termination Distribution')
plt.plot(x_data, predicted_h_np,  label='Predicted Termination Distribution')
plt.plot(x_data, predicted_cumtran_np,  label='Predicted Cumulative Transmittance')
plt.legend()
plt.show()


evel_cumtran, evel_local_tran = render_cumtran(output)
evel_h = render_h(evel_cumtran, evel_local_tran)
evel_depth = render_ray(evel_h)
print(evel_depth)