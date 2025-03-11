import torch
from torchvision import datasets
from torchvision.transforms import Lambda, ToTensor

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(
        lambda y: torch.zeros(10, dtype=torch.float).scatter_(
            dim=0, index=torch.tensor(y), value=1
        )
    ),
)

first_img, first_label = ds[0]
print(f"Image shape: {first_img.size()}")
print(f"Label shape: {first_label.size()}")
