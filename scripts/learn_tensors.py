import numpy as np
import torch

print("-" * 40)

data = [[1, 2], [3, 4]]
print(f"Raw data: \n {data} \n")

x_data = torch.tensor(data)
print(f"Direct Tensor: \n {x_data} \n")

np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"From NumPy Tensor: \n {x_np} \n")

x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")

print("-" * 40)

shape = (
    2,
    3,
)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")


print("-" * 40)

tensor = torch.rand(3, 4)
print(f"Random Tensor: \n {tensor} \n")

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

print("-" * 40)

tensor = torch.ones(4, 4)
print(f"Tensor: \n {tensor} \n")
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")

tensor[:, 1] = 0
print(tensor)

print("-" * 40)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)


print("-" * 40)

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
print(torch.matmul(tensor, tensor.T, out=y3))

z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
print(torch.mul(tensor, tensor, out=z3))


print("-" * 40)

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))


print("-" * 40)

print(f"{tensor} \n")
tensor.add_(5)
print(tensor)


print("-" * 40)

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")


print("-" * 40)

n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
