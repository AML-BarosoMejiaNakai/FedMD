import torch
from torchvision.datasets import Subset


class CustomSubset(Subset):
    def __init__(self, dataset, indices) -> None:
        super().__init__(dataset, indices)
        self.targets = self.dataset.targets[self.indices]
