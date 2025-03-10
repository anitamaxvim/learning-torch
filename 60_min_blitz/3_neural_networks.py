import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input):
        # Convolution layer C1: 1 input image, 6 output channels,
        # 5x5 square convolution, it uses RELU activation function,
        # and outputs a Tensor with size (N, 6, 28, 28), where N
        # is the siz of the batch
        c1 = F.relu(self.conv1(input))

        # Subsampling layer S2: 2x2 grid, purely functional.
        # This layer does not have any parameters, and outputs a (N, 6, 14, 14) Tensor
        s2 = F.max_pool2d(c1, (2, 2))

        # Convolution layer C3: 6 input channels, 16 output channels
        # 5x5 square convolution, it uses RELU activation function and
        # outputs a (N, 16, 10, 10) Tensor
        c3 = F.relu(self.conv2(s2))

        # Subsampling layer S4: 2x2 grid, purely functional,
        # this layer has no paramters and outputs a (N, 16, 5, 5) Tensor
        s4 = F.max_pool2d(c3, 2)

        # Flatten operation: purely functional, outputs a (N, 400) Tensor
        s4 = torch.flatten(s4, 1)

        # Fully connected layer F5: (N, 400) Tensor input, and
        # ouputs a (N, 120) Tensor, and uses RELU activation
        f5 = F.relu(self.fc1(s4))

        # Fully connected layer F6: (N, 120) Tensor input,
        # and outputs a (N, 84) Tensor, it uses RELU activation function
        f6 = F.relu(self.fc2(f5))

        # Gaussian layer OUTPUT: (N, 84) Tensor input, and
        # outputs a (N, 10) Tensor
        output = self.fc3(f6)
        return output


net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's weight

# Try a random 32x32 input
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

# Zero the gradint buffers of all params and backprops with random gradients:
net.zero_grad()
out.backward(torch.randn(1, 10))

# ---- Loss Function ----
output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

# ---- Backprop ----
net.zero_grad()  # zeroes the gradient buffers on all parameters

print("conv1.bias.grad before backward")
print(net.conv1.bias.grad)  # type: ignore

loss.backward()

print("conv1.bias.grad after backward")
print(net.conv1.bias.grad)  # type: ignore

# ---- Updating the weights ----

# SGD: weight = weight - learning_rate * gradient

# learning_rate = 0.01
# for f in net.parameters():
#     f.data.sub_(f.grad.data * learning_rate)


# Create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# In your training loop:
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()
