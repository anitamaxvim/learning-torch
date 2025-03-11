import numpy as np
import torch

# ---- Tensor Initialization ----

# Directly from data.
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(f"Tensor directly from data: \n {x_data} \n")

# Directly from a NumPy array.
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"Tensor from a NumPy array: \n {x_np} \n")

# From another tensor.
x_ones = torch.ones_like(x_data)  # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# With random or constant values
shape = (
    2,
    3,
)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


# ---- Tensor Attributes ----

tensor = torch.rand(3, 4)


print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


# ---- Tensor Operations ----

# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
    print(f"Device tensor is stored on: {tensor.device}")

# Standard numpy-like indexing and slicing
tensor = torch.ones(4, 4)
tensor[:, 1] = 0
print(tensor, "\n")

# Joining tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)
t2 = torch.cat([tensor, tensor, tensor], dim=0)
print(t1, "\n")
print(t2, "\n")

# Multiplying tensors

# Element-wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax:
print(f"tensor * tensor \n {tensor * tensor}")

# Matrix multiplication
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# Alternative syntax:
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

# In-place operations (use _ suffix)
print(tensor, "\n")
tensor.add_(5)
print(tensor, "\n")

# ---- Bridge with NumPy ----

# Tensor to NumPy array
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n} \n")

# A change in the tensor reflects in the NumPy array
t.add_(1)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n} \n")

# NumPy array to Tensor
n = np.ones(5)
t = torch.from_numpy(n)

# Changes in the NumPy array reflect in the Tensor
np.add(n, 5, out=n)
print(f"t: {t}")
print(f"n: {n}")
