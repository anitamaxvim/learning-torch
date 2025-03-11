"""
This implementation defines the model as a custom Module subclass.
Whenever you want a model more complex than a simple sequence of
existing Modules you will need to define your model this way.
"""

import math

import matplotlib.pyplot as plt
import torch


class Polynomial3(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate four parameters
        and assign them as member parameters.
        """
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return self.a + self.b * x + self.c * x**2 + self.d * x**3

    def string(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        return f"y = {self.a.item():.4f} + {self.b.item():.4f} x + {self.c.item():.4f} x^2 + {self.d.item():.4f} x^3"


# Create Tensors to hold inputs and outputs.
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# Construct our model by instantiating the class defined above
model = Polynomial3()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters (defined
# with torch.nn.Parameter) which are members of the model.
criterion = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)


# Set up the plot
plt.ion()
fig, ax = plt.subplots()
ax.plot(x, y, label="True Function (sin(x))", color="blue")
(fitted_line,) = ax.plot(x, torch.zeros_like(x), label="Polynomial Fit", color="red")
ax.legend()

for t in range(2000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    if t % 10 == 9:
        print(f"Iteration: {t}, Loss: {loss.item()}")

        # Update the plot
        fitted_line.set_ydata(y_pred.detach().numpy())
        plt.title(f"Iteration {t}, Loss: {loss.item():.4f}")
        plt.pause(0.001)  # Pause to update the plot

    # Zero gradients, perform backward pass, and update the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Keep the final plot open
plt.ioff()
plt.show()

print(f"Result: {model.string()}")
