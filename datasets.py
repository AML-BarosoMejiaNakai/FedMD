import torch
from torch.utils.data.dataset import Subset


class CustomSubset(Subset):
    def __init__(self, dataset, indices) -> None:
        super().__init__(dataset, indices)
        if torch.is_tensor(indices):
            self.targets = torch.tensor([self.dataset.targets[idx] for idx in indices.long()], dtype=torch.long)
        else:
            self.targets = torch.tensor([self.dataset.targets[int(idx)] for idx in indices], dtype=torch.long)