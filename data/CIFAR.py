import torch
import torchvision
import torchvision.transforms as transforms

# Table taken from github.com/ryanchankh/cifar100coarse
# Mapping of the 100 CIFAR100 classes to the 20 superclasses (categories)
coarse_labels = torch.tensor([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                                3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
                                0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                                5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                                16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                                10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
                                2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                                16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                                18,  1,  2, 15,  6,  0, 17,  8, 14, 13], dtype=torch.long)


def load_CIFAR10(train_transform = None, root_dir='./data/cifar10'):
    if train_transform is None:
        train_transform = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
                    ])

    train_dataset = torchvision.datasets.CIFAR10(root_dir, transform=train_transform, download=True)
    test_dataset  = torchvision.datasets.CIFAR10(root_dir, train=False, transform=train_transform, download=True)
    return train_dataset, test_dataset

def load_CIFAR100(train_transform = None, root_dir='./data/cifar100', granularity='fine'):
    if train_transform is None:
        train_transform = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
                    ])

    train_dataset = torchvision.datasets.CIFAR100(root_dir, transform=train_transform, download=True)
    test_dataset  = torchvision.datasets.CIFAR100(root_dir, train=False, transform=train_transform, download=True)
    
    if granularity == 'coarse':
        train_dataset.targets = coarse_labels[train_dataset.targets]
        test_dataset.targets  = coarse_labels[test_dataset.targets ]
    
    return train_dataset, test_dataset