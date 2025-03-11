"""
To showcase the power of PyTorch dynamic graphs, we will
implement a very strange model: a third-fifth order polynomial
that on each forward pass chooses a random number between
4 and 5 and uses that many orders, reusing the same weights
multiple times to compute the fourth and fifth order.
"""

import math
import random

import matplotlib.pyplot as plt
import torch


class DynamicNet(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate five parameters and assign
        them as members.
        """
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
        self.e = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        """
        For the forward pass of the model, we randomly choose either 4, 5
        and reuse the e parameter to compute the contribution of these orders.

        Since each forward pass builds a dynamic computation graph, we can use normal
        Python control-flow operators like loops or conditional statements when
        defining the forward pass of the model.

        Here we also see that it is perfectly safe to reuse the same parameter many
        times when defining a computational graph.
        """
        y = self.a + self.b * x + self.c * x**2 + self.d * x**3
        for exp in range(4, random.randint(4, 6)):
            y = y + self.e * x**exp
        return y

    def string(self):
        """
        Just like any class in python, you can also define custom method on PyTorch modules.
        """
        return f"y = {self.a.item():.4f} + {self.b.item():.4f} x + {self.c.item():.4f} x^2 + {self.d.item():.4f} x^3 + {self.e.item():.4f} x^4 ? + {self.e.item():.4f} x^5 ?"


# Create Tensors to hold inputs and outputs
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# Construct our model by instaniating the class defined above
model = DynamicNet()

# Construct our loss function and Optimizer. Training this strange model with
# vanilla stochastic gradient descent is tough, so we use momentum
criterion = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)


# Set up the plot
plt.ion()
fig, ax = plt.subplots()
ax.plot(x, y, label="True Function (sin(x))", color="blue")
(fitted_line,) = ax.plot(x, torch.zeros_like(x), label="Polynomial Fit", color="red")
ax.legend()

for t in range(30000):
    # Forwward pass
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    if t % 200 == 199:
        print(f"Iteration: {t}, Loss: {loss.item()}")

        # Update the plot
        fitted_line.set_ydata(y_pred.detach().numpy())
        plt.title(f"Iteration {t}, Loss: {loss.item():.4f}")
        plt.pause(0.001)  # Pause to update the plot

    # Zero gradients, peform backward pass and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Keep the final plot open
plt.ioff()
plt.show()

print(f"Result: {model.string()}")
