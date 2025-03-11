import math

import matplotlib.pyplot as plt
import torch


class LegendrePolynomial3(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions
    by subclassing `torch.autograd.Function` and
    implementing the forward and backward passes which operate
    on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we recieve a Tensor containing
        the input and return a Tensor containing the output.
        `ctx` is a context object that can be used to stash
        information for backward computation.
        You can cache arbitrary objects for use in the backward
        pass using ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return 0.5 * (5 * input**3 - 3 * input)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        """
        In the backward pass we recieve a Tensor containing
        the gradient of the loss with respect to the output,
        and we need to compute the gradient of the loss
        with respect to the input.
        """
        (input,) = ctx.saved_tensors
        return grad_output * 1.5 * (5 * input**2 - 1)


dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0")  # Uncomment this to run on GPU


# Create Tensors to hold input and outputs.
# By default, requires_grad=False, which indicates that we do not need to
# compute gradients with respect to these Tensors during the backward pass.
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)


# Create random Tensors for weights. For this example, we need
# 4 weights: y = a + b * P3(c + d * x), these weights need to be initialized
# not too far from the correct result to ensure convergence.
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.
a = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
b = torch.full((), -1.0, device=device, dtype=dtype, requires_grad=True)
c = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
d = torch.full((), 0.3, device=device, dtype=dtype, requires_grad=True)

# Set up the plot
plt.ion()
fig, ax = plt.subplots()
ax.plot(x, y, label="True Function (sin(x))", color="blue")
(fitted_line,) = ax.plot(x, torch.zeros_like(x), label="Polynomial Fit", color="red")
ax.legend()

learning_rate = 5e-6
for t in range(2000):
    # To apply our Function, we use Function.apply method.
    # We alias this as 'P3'.
    P3 = LegendrePolynomial3.apply

    # Forward pass: compute predicted y using operations:
    # we compute P3 using our custom autograd operation.
    y_pred = a + b * P3(c + d * x)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum()
    if t % 10 == 9:
        print(f"Iteration {t}: Loss = {loss.item():.4f}")

        # Update the plot
        fitted_line.set_ydata(y_pred.detach().numpy())
        plt.title(f"Iteration {t}, Loss: {loss.item():.4f}")
        plt.pause(0.01)  # Pause to update the plot

    # Use autograd to compute the backward pass.
    loss.backward()

    # Update weights using gradient descent
    with torch.no_grad():
        a -= learning_rate * a.grad  # type: ignore
        b -= learning_rate * b.grad  # type: ignore
        c -= learning_rate * c.grad  # type: ignore
        d -= learning_rate * d.grad  # type: ignore

        # Manually zero the gradients after updating weights
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

# Keep the final plot open
plt.ioff()
plt.show()

print(f"Result: y = {a.item()} + {b.item()} * P3({c.item()} + {d.item()} x)")
