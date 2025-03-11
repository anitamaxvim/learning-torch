import math

import matplotlib.pyplot as plt
import numpy as np

# Create random input and output data
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

# Randomly initialize weights
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

# Set up the plot
plt.ion()
fig, ax = plt.subplots()
ax.plot(x, y, label="True Function (sin(x))", color="blue")
(fitted_line,) = ax.plot(x, np.zeros_like(x), label="Polynomial Fit", color="red")
ax.legend()

learning_rate = 1e-6
for t in range(2000):
    # Forward pass
    # y = a + bx + cx^2 + dx^3
    y_pred = a + b * x + c * x**2 + d * x**3

    # Compute and print loss = (a - y)^2
    loss = np.square(y_pred - y).sum()
    if t % 10 == 9:
        print(f"Iteration {t}: Loss = {loss:.4f}")

        # Update the plot
        fitted_line.set_ydata(y_pred)
        plt.title(f"Iteration {t}, Loss: {loss:.4f}")
        plt.pause(0.01)  # Pause to update the plot

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

print(f"Result: y = {a:.4f} + {b:.4f} x + {c:.4f} x^2 + {d:.4f} x^3")
