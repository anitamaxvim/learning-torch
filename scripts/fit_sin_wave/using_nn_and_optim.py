"""
This implementation uses the nn package from PyTorch to build the network.

Rather than manually updating the weights of the model as we have been
doing, we use the optim package to define an Optimizer that will
update the weights for us. The optim package defines many optimization
algorithms that are commonly used for deep learning, including SGD+momentum,
RMSProp, Adam, etc.
"""

import math

import matplotlib.pyplot as plt
import torch

# Create Tensors to hold input and outputs.
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# Prepare the input tensor (x, x^2, x^3)
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(torch.nn.Linear(3, 1), torch.nn.Flatten(0, 1))
loss_fn = torch.nn.MSELoss(reduction="sum")

# Set up the plot
plt.ion()
fig, ax = plt.subplots()
ax.plot(x, y, label="True Function (sin(x))", color="blue")
(fitted_line,) = ax.plot(x, torch.zeros_like(x), label="Polynomial Fit", color="red")
ax.legend()

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use RMSprop; the optim package contains many other
# optimization algorithms. The first argument to the RMSprop constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-3
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

for t in range(2000):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(xx)

    # Compute and print loss
    loss = loss_fn(y_pred, y)
    if t % 10 == 9:
        print(f"Iteration: {t}, Loss: {loss.item()}")

        # Update the plot
        fitted_line.set_ydata(y_pred.detach().numpy())
        plt.title(f"Iteration {t}, Loss: {loss:.4f}")
        plt.pause(0.01)  # Pause to update the plot

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss w.r.t the model params
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its parameters
    optimizer.step()


# Keep the final plot open
plt.ioff()
plt.show()

linear_layer = model[0]
print(
    f"Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3"
)
