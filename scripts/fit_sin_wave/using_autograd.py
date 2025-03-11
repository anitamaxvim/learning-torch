import math

import matplotlib.pyplot as plt
import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

torch.set_default_device(device)

# Create Tensors to hold input and outputs.
# By default, `requires_grad=False`, which indicates that we do not need to
# compute gradients with respect to these Tensors during the backward pass.

x = torch.linspace(-math.pi, math.pi, 2000, dtype=dtype)
y = torch.sin(x)

# Create random Tensors for weights. For a third order polynomial, we need:
# 4 weights: y = a + bx + cx^2 + dx^3
# Setting `requires_grad=True` indicates that we want to compute gradients
# with respect to these Tensors during the backward pass.
a = torch.randn((), dtype=dtype, requires_grad=True)
b = torch.randn((), dtype=dtype, requires_grad=True)
c = torch.randn((), dtype=dtype, requires_grad=True)
d = torch.randn((), dtype=dtype, requires_grad=True)

# Set up the plot
plt.ion()
fig, ax = plt.subplots()
ax.plot(x, y, label="True Function (sin(x))", color="blue")
(fitted_line,) = ax.plot(x, torch.zeros_like(x), label="Polynomial Fit", color="red")
ax.legend()

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y using operations on Tensors.
    y_pred = a + b * x + c * x**2 + d * x**3

    # Compute and print loss using operations on Tensors.
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the scalar value held in the loss.
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(f"Iteration {t}: Loss = {loss.item():.4f}")

        # Update the plot
        fitted_line.set_ydata(y_pred.detach().numpy())
        plt.title(f"Iteration {t}, Loss: {loss.item():.4f}")
        plt.pause(0.15)  # Pause to update the plot

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call a.grad, b.grad. c.grad and d.grad will be Tensors holding
    # the gradient of the loss with respect to a, b, c, d respectively.
    loss.backward()

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.

    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        # Manually zero the gradients after updating weights
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

# Keep the final plot open
plt.ioff()
plt.show()

print(f"Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3")
