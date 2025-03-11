import torch
from torch import nn, optim
from torchvision.models import ResNet18_Weights, resnet18

# ---- A Single Training Step ----

model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

prediction = model(data)  # forward pass

loss = (prediction - labels).sum()
loss.backward()  # backward pass

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

optimizer.step()  # gradient descent


# ---- Differentiation in Autograd ----

# 'requires_grad=True' signals to autograd that every operation on the tensor should be tracked
a = torch.tensor([2.0, 3.0], requires_grad=True)
b = torch.tensor([6.0, 4.0], requires_grad=True)

Q = 3 * a**3 - b**2

external_grad = torch.tensor([1.0, 1.0])
Q.backward(gradient=external_grad)

# Check if collected gradients are correct

print(a.grad == 9 * a**2)
print(b.grad == -2 * b)


# ---- Fine-tuning and Freezing Paramaeters ----

model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Freeze all paramaters in the network
for param in model.parameters():
    param.requires_grad = False

# Replace the classifier (last linar layer) with a new linear layer
# This layer is unfrozen by default.
# So now all the params in the model, except the paramaters of the the final layer, are frozen.
model.fc = nn.Linear(512, 10)

# Optimize only the classifier
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
