import torch
from data.custom_subset import CustomSubset as Subset

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

def split_dataset_imbalanced(dataset, super_classes, N_agents, N_samples_per_class, classes_per_agent, seed = None):
    if seed is not None:
        rand_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    labels = torch.tensor(dataset.targets)
    private_idxs = [torch.tensor([], dtype=torch.long)]*N_agents
    all_idxs = torch.tensor([], dtype=torch.long)
    for i, agent_classes in enumerate(classes_per_agent):
        for cls_ in agent_classes:
            idxs = torch.nonzero(labels == cls_).flatten()
            samples = torch.multinomial(torch.ones(idxs.size()), N_samples_per_class)
            idx_agent = idxs[samples]
            private_idxs[i] = torch.cat((private_idxs[i], idx_agent))
        all_idxs = torch.cat((all_idxs, private_idxs[i]))
    if seed is not None:
        torch.random.set_rng_state(rand_state)
    dataset.targets = super_classes
    private_data = [Subset(dataset, private_idx) for private_idx in private_idxs]
    all_private_data = Subset(dataset, all_idxs)
    
    return private_data, all_private_data


def stratified_sampling(dataset, size = 3000):
    import sklearn.model_selection
    idxs = sklearn.model_selection.train_test_split([i for i in range(len(dataset))], \
        train_size = size, stratify = dataset.targets)[0]
    return Subset(dataset, idxs)