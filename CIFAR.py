import torch
import torchvision
import torchvision.transforms as transforms
from datasets import CustomSubset as Subset

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

def load_CIFAR100(train_transform = None, root_dir='./data/cifar100'):
    if train_transform is None:
        train_transform = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
                    ])

    train_dataset = torchvision.datasets.CIFAR100(root_dir, transform=train_transform, download=True)
    test_dataset  = torchvision.datasets.CIFAR100(root_dir, train=False, transform=train_transform, download=True)
    return train_dataset, test_dataset

def generate_class_subset(dataset, classes):
    dataset_classes = torch.tensor(dataset.targets)
    idxs = torch.cat([torch.nonzero(dataset_classes == i) for i in classes])
    return Subset(dataset, idxs)

def split_dataset(dataset, N_agents, N_samples_per_class, classes_in_use = None, seed = None):
    if classes_in_use is None:
        classes_in_use = list(set(dataset.targets))
    if seed is not None:
        rand_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    labels = torch.tensor(dataset.targets)
    private_idxs = [torch.tensor([], dtype=torch.long)]*N_agents
    all_idxs = torch.tensor([], dtype=torch.long)
    for cls_ in classes_in_use:
        idxs = torch.nonzero(labels == cls_).flatten()
        samples = torch.multinomial(torch.ones(idxs.size()), N_agents * N_samples_per_class)
        all_idxs = torch.cat((all_idxs, idxs[samples]))
        for i in range(N_agents):
            idx_agent = idxs[samples[i*N_samples_per_class : (i+1)*N_samples_per_class]]
            private_idxs[i] = torch.cat((private_idxs[i], idx_agent))
    if seed is not None:
        torch.random.set_rng_state(rand_state)

    private_data = [Subset(dataset, private_idx) for private_idx in private_idxs]
    all_private_data = Subset(dataset, all_idxs)
    
    return private_data, all_private_data

def stratified_sampling(dataset, size = 3000):
    import sklearn.model_selection
    idxs = sklearn.model_selection.train_test_split([i for i in range(len(dataset))], \
        train_size = size, stratify = dataset.targets)[0]
    return Subset(dataset, idxs)