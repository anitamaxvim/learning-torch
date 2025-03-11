import math

import matplotlib.pyplot as plt
import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU


# Create random input and output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Randomly initialize weights
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

# Set up the plot
plt.ion()
fig, ax = plt.subplots()
ax.plot(x, y, label="True Function (sin(x))", color="blue")
(fitted_line,) = ax.plot(x, torch.zeros_like(x), label="Polynomial Fit", color="red")
ax.legend()

learning_rate = 1e-6
for t in range(2000):
    # Forward pass
    # y = a + bx + cx^2 + dx^3
    y_pred = a + b * x + c * x**2 + d * x**3

    # Compute and print loss = (a - y)^2
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(f"Iteration {t}: Loss = {loss:.4f}")

        # Update the plot
        fitted_line.set_ydata(y_pred)
        plt.title(f"Iteration {t}, Loss: {loss:.4f}")
        plt.pause(0.15)  # Pause to update the plot

    # Backpropagation
    grad_y_pred = 2.0 * (y_pred - y)

    # Use the chain rule to compute gradients of the weights
    grad_a = (grad_y_pred * 1).sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x**2).sum()
    grad_d = (grad_y_pred * x**3).sum()

    # Update the weights using SGD
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

# Keep the final plot open
plt.ioff()
plt.show()

print(
    f"Result: y = {a.item():.4f} + {b.item():.4f} x + {c.item():.4f} x^2 + {d.item():.4f} x^3"
)
