import torch
import torch.nn as nn
import torchvision 
import torch.nn.functional as F
import torch.distributions as D
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

import os
import pickle
import pandas as pd 
import numpy as np
import cvxpy as cp
import math
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from random import sample
import random
from tdigest import TDigest
from collections import defaultdict
from scipy.special import logsumexp
from numpy import logaddexp
from copy import deepcopy
from ddsketch import DDSketch
from ddsketch.ddsketch import DDSketch
from ddsketch.mapping import LogarithmicMapping
from scipy.interpolate import interp1d

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader

from torch.utils.data import Subset
from collections import defaultdict
from sklearn.model_selection import train_test_split
from conformal_utils2 import *

from dataset.rxrx1 import RxRx1TorchDataset

def stripPrefix(stateDict):
    returnDict = {}
    for key in stateDict:
        newKey = key[6:]
        returnDict[newKey] = stateDict[key]
    return returnDict

def myLoad(module, path, device=None):
    if device is not None:
        state = torch.load(path, map_location=device)
    else:
        state = torch.load(path)
    state = stripPrefix(state['algorithm'])
    
    module.load_state_dict(state)

    return 

def initializeTransform(dataset_name: str):
    
    def standardize(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2))
        std = x.std(dim=(1, 2))
        std[std == 0.] = 1.
        return TF.normalize(x, mean, std)

    t_standardize = transforms.Lambda(lambda x: standardize(x))
    
    if dataset_name.lower() == 'rxrx1':
        transform = transforms.Compose([
            transforms.ToTensor(),
            t_standardize,
        ])
    
    elif dataset_name == 'fmow' or dataset_name == 'iwildcam':
        # ImageNet-style normalization (RGB natural images)
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])

    elif dataset_name == 'amazon':
        # Amazon is text → transforms not needed here (handled via tokenizer)
        # Return identity function or raise to remind user to use tokenizer
        transform = lambda x: x  # or raise NotImplementedError
        # raise ValueError("Amazon Reviews is a text dataset; use tokenizer instead.")

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return transform

def load_model(dataset, device):
 
    if dataset == 'rxrx1':
        dimOut = 1139
        constructor = torchvision.models.resnet50
        model = constructor()
        dimFeatures = model.fc.in_features
        lastLayer = nn.Linear(dimFeatures,dimOut)
        model.d_out = dimFeatures
        model.fc = lastLayer

        Model = model.to(device)
        myLoad(Model,
            'data/rxrx1_seed_0_epoch_best_model.pth',
            device=device)
        featurizer = Model
        classifier = Model.fc
        featurizer.fc = nn.Identity()

        Model = nn.Sequential(*(featurizer,classifier))
    
    else:
    	raise Exception('Undefined dataset')

    Model.eval()
    featurizer.eval()
    classifier.eval()

    return Model, featurizer, classifier

def load_dataset(dataset):

    myTransform = initializeTransform(dataset)

    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    Data = get_dataset(dataset=dataset, download=False)
    TrainImages = Data.get_subset(
        "train",
        transform = myTransform
    )
    TestImages = Data.get_subset(
        "test",
        transform = myTransform
    )

    if dataset == 'rxrx1':
        metaData = pd.read_csv('data/rxrx1_v1.0/metadata.csv')
    else:
    	raise Exception('Undefined dataset')

    metaData_train = metaData[metaData['dataset'] == 'train']
    metaData_test = metaData[metaData['dataset'] == 'test']

    train_dataset = RxRx1TorchDataset(metaData_train, image_root = 'data/rxrx1_v1.0/images', transform=myTransform, label_col='sirna')
    test_dataset  = RxRx1TorchDataset(metaData_test,  image_root = 'data/rxrx1_v1.0/images', transform=myTransform, label_col='sirna')
    
    return train_dataset, test_dataset

def load_feature(dataset, indices, featurizer, device, tokenizer=None):
    """
    Loads features for WILDS datasets using a featurizer.
    Supports rxrx1, fmow, iwildcam, and amazon.
    """
    dataset = dataset.lower()
    if dataset in ['rxrx1', 'fmow', 'iwildcam']:
        # Image datasets
        myTransform = initializeTransform(dataset)
        Data = get_dataset(dataset=dataset, download=False)
        TestImages = Data.get_subset("test", transform=myTransform)

        # Infer feature dimension dynamically
        with torch.no_grad():
            sample_tensor = TestImages[0][0].unsqueeze(0).to(device)
            feat = featurizer(sample_tensor)
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            feature_dim = feat.shape[-1]

        featureMat = np.zeros((len(indices), feature_dim))

        featurizer.eval()
        with torch.no_grad():
            for i, idx in enumerate(indices):
                x_tensor = TestImages[idx][0].unsqueeze(0).to(device)
                feat = featurizer(x_tensor)
                if isinstance(feat, (tuple, list)):
                    feat = feat[0]
                featureMat[i, :] = feat.cpu().numpy()[0]

        return featureMat

    elif dataset == 'amazon':
        # Text dataset: needs tokenizer + BERT-style featurizer
        assert tokenizer is not None, "Tokenizer must be provided for Amazon Reviews."

        Data = get_dataset(dataset="amazon", download=False)
        TestSubset = Data.get_subset("test")
        texts = [TestSubset[idx][0] for idx in indices]  # raw strings

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        featurizer.eval()
        with torch.no_grad():
            outputs = featurizer(**encoded)
            # Use [CLS] token representation
            if hasattr(outputs, "last_hidden_state"):
                cls_features = outputs.last_hidden_state[:, 0, :]  # (B, hidden_size)
            else:
                cls_features = outputs[0][:, 0, :]

        return cls_features.cpu().numpy()

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

def load_soft_dataset(dataset, data_folder='dataset'):
    '''
    Load softmax scores and labels for a dataset
    
    Input:
        - dataset: string specifying dataset. Options are 'imagenet', 'cifar-100', 'places365', 'inaturalist'
        - data_folder: string specifying folder containing the <dataset name>.npz files

    Output: softmax_scores, labels
        
    '''
    # assert dataset in ['imagenet', 'cifar100', 'places365', 'inaturalist']
    
    
    data = np.load(f'{data_folder}/{dataset}.npz')
    # data = np.load(f'{data_folder}/{dataset}.npz')
    softmax_scores = data['softmax']
    labels = data['labels']
    indices = data['indices'] 
    
    return softmax_scores, labels, indices

def partition_iid(dataset, num_clients):
    """Evenly split the data across clients (i.i.d.)."""
    indices = np.random.permutation(len(dataset))
    split_sizes = np.array_split(indices, num_clients)
    return {i: split.tolist() for i, split in enumerate(split_sizes)}

def partition_iid_2(data_size, num_clients, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(data_size)
    split_sizes = np.array_split(indices, num_clients)
    return {i: split.tolist() for i, split in enumerate(split_sizes)}


# def partition_non_iid(dataset, num_clients, num_bins=5):
#     """Partition using quantile-binned target for non-i.i.d. simulation."""
#     targets = np.array([dataset[i][1].item() for i in range(len(dataset))])
#     bins = np.quantile(targets, np.linspace(0, 1, num_bins + 1)[1:-1])
#     bin_labels = np.digitize(targets, bins)

#     bins_to_indices = defaultdict(list)
#     for idx, b in enumerate(bin_labels):
#         bins_to_indices[b].append(idx)

#     client_indices = defaultdict(list)
#     for i in range(num_clients):
#         for b in range(num_bins):
#             indices = bins_to_indices[b]
#             chunk_size = len(indices) // num_clients
#             start = i * chunk_size
#             end = (i + 1) * chunk_size if i != num_clients - 1 else len(indices)
#             client_indices[i].extend(indices[start:end])

#     return client_indices

# def partition_civilcomments_by_subgroup(dataset, num_clients):
#     """
#     Partition CivilComments dataset by protected subgroup attributes to induce concept shift.
#     Assumes dataset is from WILDS and includes 'subgroup' metadata.
#     """
#     subgroup_fields = ['male', 'female', 'black', 'white', 'christian', 'muslim', 'lgbtq']  # or use dataset.metadata_fields filtering
#     subgroup_to_indices = defaultdict(list)

#     for i in range(len(dataset)):
#         metadata = dataset.metadata_array[i]
#         for subgroup in subgroup_fields:
#             subgroup_idx = dataset.metadata_fields.index(subgroup)
#             if metadata[subgroup_idx] == 1:
#                 subgroup_to_indices[subgroup].append(i)

#     # assign each client to a subgroup (cyclically or uniquely)
#     client_indices = defaultdict(list)
#     subgroups = list(subgroup_to_indices.keys())
#     for i in range(num_clients):
#         subgroup = subgroups[i % len(subgroups)]
#         client_indices[i].extend(subgroup_to_indices[subgroup])

#     return client_indices


def partition_non_iid(dataset, metadata, num_clients):
    """Partition using quantile-binned target for non-i.i.d. simulation."""


    if dataset == 'rxrx1':
        client_indices = partition_rxrx1_by_experiment(metadata, num_clients)
    elif dataset == 'camelyon': 
        client_indices = partition_camelyon17_by_hospital(metadata, num_clients)
    elif dataset == 'civil': 
        client_indices = partition_civilcomments_by_subgroup(metadata, num_clients)
    elif dataset == 'amazon':
        client_indices = partition_amazon_by_domain(metadata, num_clients)
    else:
        raise Exception('Undefined dataset')

    return client_indices

def partition_non_iid_2(dataset, metadata, args):
    """Partition using quantile-binned target for non-i.i.d. simulation."""
    if dataset == 'rxrx1':
        client_indices = partition_rxrx1_disjoint_labels(metadata, args.num_clients, args.rho)
    elif dataset == 'iwild': 
        client_indices = partition_iwild_disjoint_labels(metadata, args.num_clients, args.rho)
    elif dataset == 'fmow': 
        client_indices = partition_fmow_disjoint_labels(metadata, args.num_clients, args.rho)
    elif dataset == 'amazon':
        client_indices = partition_amazon_joint_dirichlet(metadata, args.num_clients, args.rho)
    else:
        raise Exception('Undefined dataset')

    return client_indices

def partition_rxrx1_by_experiment(metadata, num_clients, seed=42):
    """
    metadata['experiment'] should be a length-N array.
    Returns: client_id -> list of sample indices
    """
    np.random.seed(seed) 
    experiment_to_indices = defaultdict(list)
    
    for i, exp in enumerate(metadata['experiment']):
        experiment_to_indices[exp].append(i)

    experiments = list(experiment_to_indices.keys())
    np.random.shuffle(experiments)

    client_indices = defaultdict(list)
    for i, exp in enumerate(experiments):
        cid = i % num_clients
        client_indices[cid].extend(experiment_to_indices[exp])

    return client_indices

# def partition_rxrx1_by_joint(metadata, num_clients, rho=0.3, seed=42):
#     """
#     Combined strategy: Assign experiments to clients (domain shift),
#     and within each client use Dirichlet to sample labels (label imbalance).

#     Args:
#         metadata: pandas.DataFrame with columns 'experiment' and 'label'
#         num_clients: number of clients
#         rho: Dirichlet concentration parameter (smaller = more non-iid)
#         seed: random seed

#     Returns:
#         dict: client_id -> list of sample indices
#     """
#     np.random.seed(seed)
#     experiment_col = 'experiment'
#     label_col = 'label'

#     min_samples_per_class = 10

#     # Step 1: Group indices by (experiment, label)
#     group_dict = defaultdict(list)
#     for i, (exp, y) in enumerate(zip(metadata[experiment_col], metadata[label_col])):
#         group_dict[(exp, y)].append(i)

#     # Step 2: Assign experiments to clients evenly (preserve domain shift)
#     experiment_to_indices = defaultdict(list)
#     for (exp, y), idxs in group_dict.items():
#         experiment_to_indices[exp].extend(idxs)

#     experiments = list(experiment_to_indices.keys())
#     np.random.shuffle(experiments)
#     client_to_experiments = defaultdict(list)
#     for i, exp in enumerate(experiments):
#         cid = i % num_clients
#         client_to_experiments[cid].append(exp)

#     # Step 3: Get all labels
#     all_labels = sorted(set(metadata[label_col]))
#     num_classes = len(all_labels)

#     # Step 4: Generate Dirichlet distribution over classes for each client
#     class_prop = np.random.dirichlet(alpha=np.ones(num_classes) * rho, size=num_clients)  # shape: [client, class]

#     # Step 5: Assign samples from each (experiment, label) to corresponding client
#     client_indices = defaultdict(list)

#     for cid in range(num_clients):
#         for exp in client_to_experiments[cid]:
#             for y in all_labels:
#                 key = (exp, y)
#                 if key not in group_dict:
#                     continue
#                 indices = group_dict[key]
#                 np.random.shuffle(indices)
#                 # sample proportion for this label from this client's Dirichlet vector
#                 prop = class_prop[cid, y]
#                 count = max(int(round(prop * len(indices))), min_samples_per_class)
#                 # count = int(round(prop * len(indices)))  # soft allocation
#                 client_indices[cid].extend(indices[:count])

#     return client_indices

def partition_rxrx1_disjoint_labels(metadata, num_clients, rho=0.3, seed=42, min_samples_per_class=10):
    """
    Construct a strong joint shift:
    - Client experiments are disjoint (P(X) shift)
    - Client labels are disjoint (P(Y) shift)
    - Within allowed labels, use Dirichlet distribution to control imbalance

    Args:
        metadata: DataFrame with 'experiment' and 'label'
        num_clients: number of clients
        rho: Dirichlet parameter for label imbalance
        seed: for reproducibility
        min_samples_per_class: min class samples per client to include

    Returns:
        dict[cid] -> list of sample indices
    """
    np.random.seed(seed)

    experiment_col = 'experiment'
    label_col = 'label'

    # Group by (experiment, label)
    group_dict = defaultdict(list)
    for i, (exp, y) in enumerate(zip(metadata[experiment_col], metadata[label_col])):
        group_dict[(exp, y)].append(i)

    # Assign experiments to clients
    experiment_to_indices = defaultdict(list)
    for (exp, y), idxs in group_dict.items():
        experiment_to_indices[exp].extend(idxs)

    experiments = list(experiment_to_indices.keys())
    np.random.shuffle(experiments)
    client_to_experiments = defaultdict(list)
    for i, exp in enumerate(experiments):
        cid = i % num_clients
        client_to_experiments[cid].append(exp)

    # Assign disjoint label subsets to clients
    all_labels = sorted(set(metadata[label_col]))
    num_classes = len(all_labels)
    labels_per_client = np.array_split(all_labels, num_clients)  # disjoint assignment

    client_indices = defaultdict(list)

    for cid in range(num_clients):
        allowed_labels = set(labels_per_client[cid])
        class_prop = np.random.dirichlet(alpha=np.ones(len(allowed_labels)) * rho)
        label_list = sorted(allowed_labels)

        for exp in client_to_experiments[cid]:
            for j, y in enumerate(label_list):
                key = (exp, y)
                if key not in group_dict:
                    continue
                indices = group_dict[key]
                np.random.shuffle(indices)
                prop = class_prop[j]
                count = max(int(round(prop * len(indices))), min_samples_per_class)
                client_indices[cid].extend(indices[:count])

    # Optional: filter too-small clients
    client_indices = {cid: idxs for cid, idxs in client_indices.items() if len(idxs) >= 4}

    return client_indices

def partition_rxrx1_joint_dirichlet(metadata, num_clients, rho=0.3, seed=42, min_samples=0):
    """
    Jointly non-IID partition of FMoW using (country, label) groups + Dirichlet allocation.
    """
    np.random.seed(seed)
    domain_col = 'experiment'   # or 'region' if preferred
    label_col = 'label'

    # 1. Group by (country, label)
    group_dict = defaultdict(list)
    for i, (d, y) in enumerate(zip(metadata[domain_col], metadata[label_col])):
        group_dict[(d, y)].append(i)

    client_indices = defaultdict(list)

    # 2. For each (domain, label), use Dirichlet to split into clients
    for group_key, idxs in group_dict.items():
        np.random.shuffle(idxs)
        proportions = np.random.dirichlet([rho] * num_clients)
        proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        splits = np.split(idxs, proportions)

        for cid, subset in enumerate(splits):
            if len(subset) >= min_samples:
                client_indices[cid].extend(subset)

    return dict(client_indices)

def partition_camelyon17_by_hospital(metadata, num_clients):
    """
    Partition Camelyon17 using hospital metadata.
    
    Args:
        metadata: dict containing 'hospital' as a key, whose value is a length-N array (int or str)
        num_clients: number of clients to simulate
    
    Returns:
        client_indices: dict mapping client ID to list of sample indices
    """
    from collections import defaultdict
    hospital_to_indices = defaultdict(list)

    for i, h in enumerate(metadata['hospital']):
        hospital_to_indices[h].append(i)

    hospitals = list(hospital_to_indices.keys())
    np.random.shuffle(hospitals)

    client_indices = defaultdict(list)
    for i, h in enumerate(hospitals):
        cid = i % num_clients
        client_indices[cid].extend(hospital_to_indices[h])

    return client_indices

def partition_camelyon17_disjoint_labels(metadata, num_clients, rho=0.3, seed=42, min_samples_per_class=10):
    """
    Partition Camelyon17 clients by hospital (domain shift) and disjoint labels (P(Y) shift).
    """

    np.random.seed(seed)
    hospital_col = 'hospital'
    label_col = 'label'

    group_dict = defaultdict(list)
    for i, (h, y) in enumerate(zip(metadata[hospital_col], metadata[label_col])):
        group_dict[(h, y)].append(i)

    hospitals = sorted(set(metadata[hospital_col]))
    np.random.shuffle(hospitals)
    client_to_hospitals = defaultdict(list)
    for i, h in enumerate(hospitals):
        cid = i % num_clients
        client_to_hospitals[cid].append(h)

    all_labels = sorted(set(metadata[label_col]))
    labels_per_client = np.array_split(all_labels, num_clients)

    client_indices = defaultdict(list)
    for cid in range(num_clients):
        allowed_labels = set(labels_per_client[cid])
        class_prop = np.random.dirichlet(alpha=np.ones(len(allowed_labels)) * rho)
        label_list = sorted(allowed_labels)

        for h in client_to_hospitals[cid]:
            for j, y in enumerate(label_list):
                key = (h, y)
                if key not in group_dict:
                    continue
                indices = group_dict[key]
                np.random.shuffle(indices)
                count = max(int(round(class_prop[j] * len(indices))), min_samples_per_class)
                client_indices[cid].extend(indices[:count])

    return dict(client_indices)

def partition_civilcomments_by_subgroup(metadata, num_clients, subgroup_fields=None):
    """
    Each client assigned to a different subgroup (e.g., male, female, black, etc.)
    metadata[field] must be np.array of length N with binary 0/1.
    """
    from collections import defaultdict
    if subgroup_fields is None:
        # you can set your default fields
        subgroup_fields = ['male', 'female', 'black', 'white', 'christian', 'muslim', 'lgbtq']

    subgroup_to_indices = defaultdict(list)
    for i in range(len(metadata[subgroup_fields[0]])):
        for field in subgroup_fields:
            if metadata[field][i] == 1:
                subgroup_to_indices[field].append(i)

    subgroups = list(subgroup_to_indices.keys())
    np.random.shuffle(subgroups)

    client_indices = defaultdict(list)
    for i in range(num_clients):
        group = subgroups[i % len(subgroups)]
        client_indices[i].extend(subgroup_to_indices[group])

    return client_indices

def partition_civilcomments_disjoint_labels(metadata, num_clients, rho=0.3, seed=42,
                                            subgroup_fields=None, min_samples_per_class=10):
    """
    Partition CivilComments clients by subgroup and disjoint label subsets.
    """

    if subgroup_fields is None:
        subgroup_fields = ['male', 'female', 'black', 'white', 'christian', 'muslim', 'lgbtq']
    
    np.random.seed(seed)
    label_col = 'label'  # assume binary label 0/1

    group_dict = defaultdict(list)
    for i in range(len(metadata[label_col])):
        for field in subgroup_fields:
            if metadata[field][i] == 1:
                y = metadata[label_col][i]
                group_dict[(field, y)].append(i)

    subgroups = list({key[0] for key in group_dict.keys()})
    np.random.shuffle(subgroups)
    client_to_groups = defaultdict(list)
    for i, g in enumerate(subgroups):
        cid = i % num_clients
        client_to_groups[cid].append(g)

    all_labels = sorted(set(metadata[label_col]))
    labels_per_client = np.array_split(all_labels, num_clients)

    client_indices = defaultdict(list)
    for cid in range(num_clients):
        allowed_labels = set(labels_per_client[cid])
        class_prop = np.random.dirichlet(alpha=np.ones(len(allowed_labels)) * rho)
        label_list = sorted(allowed_labels)

        for g in client_to_groups[cid]:
            for j, y in enumerate(label_list):
                key = (g, y)
                if key not in group_dict:
                    continue
                indices = group_dict[key]
                np.random.shuffle(indices)
                count = max(int(round(class_prop[j] * len(indices))), min_samples_per_class)
                client_indices[cid].extend(indices[:count])

    return dict(client_indices)

def partition_amazon_by_domain(metadata, num_clients):
    """
    metadata['domain'] is an array of domain names (strings or ints).
    """
    from collections import defaultdict
    domain_to_indices = defaultdict(list)
    
    for i, dom in enumerate(metadata['domain']):
        domain_to_indices[dom].append(i)

    domains = list(domain_to_indices.keys())
    np.random.shuffle(domains)

    client_indices = defaultdict(list)
    for i, dom in enumerate(domains):
        cid = i % num_clients
        client_indices[cid].extend(domain_to_indices[dom])

    return client_indices

def partition_amazon_disjoint_labels(metadata, num_clients, rho=0.3, seed=42, min_samples_per_class=10):
    """
    Partition Amazon clients by domain with disjoint labels and Dirichlet-controlled imbalance.
    """

    np.random.seed(seed)
    domain_col = 'category'
    label_col = 'label'

    group_dict = defaultdict(list)
    for i, (d, y) in enumerate(zip(metadata[domain_col], metadata[label_col])):
        group_dict[(d, y)].append(i)

    domains = sorted(set(metadata[domain_col]))
    np.random.shuffle(domains)
    client_to_domains = defaultdict(list)
    for i, d in enumerate(domains):
        cid = i % num_clients
        client_to_domains[cid].append(d)

    all_labels = sorted(set(metadata[label_col]))
    labels_per_client = np.array_split(all_labels, num_clients)

    client_indices = defaultdict(list)
    for cid in range(num_clients):
        allowed_labels = set(labels_per_client[cid])
        class_prop = np.random.dirichlet(alpha=np.ones(len(allowed_labels)) * rho)
        label_list = sorted(allowed_labels)

        for d in client_to_domains[cid]:
            for j, y in enumerate(label_list):
                key = (d, y)
                if key not in group_dict:
                    continue
                indices = group_dict[key]
                np.random.shuffle(indices)
                count = max(int(round(class_prop[j] * len(indices))), min_samples_per_class)
                client_indices[cid].extend(indices[:count])

    return dict(client_indices)

def partition_amazon_joint_dirichlet(metadata, num_clients, rho=0.3, seed=42, min_samples=10):
    """
    Jointly non-IID partition on (domain, label) using Dirichlet allocation.
    
    Args:
        metadata: DataFrame with 'domain' and 'label' columns
        num_clients: number of clients
        rho: Dirichlet concentration parameter
        seed: random seed
        min_samples: minimum samples assigned per (client, group)
        
    Returns:
        client_indices: dict[client_id] = list of data indices
    """
    np.random.seed(seed)
    domain_col = 'category'
    label_col = 'label'

    # Step 1: Group data by (domain, label)
    group_dict = defaultdict(list)
    for i, (d, y) in enumerate(zip(metadata[domain_col], metadata[label_col])):
        group_dict[(d, y)].append(i)

    # Step 2: Allocate each (domain, label) group to clients via Dirichlet
    client_indices = defaultdict(list)
    for group_key, idxs in group_dict.items():
        np.random.shuffle(idxs)
        proportions = np.random.dirichlet([rho] * num_clients)
        proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        splits = np.split(idxs, proportions)

        for cid, subset in enumerate(splits):
            if len(subset) >= min_samples:
                client_indices[cid].extend(subset)

    return dict(client_indices)

def partition_fmow_disjoint_labels(metadata, num_clients, rho=0.3, seed=42, min_samples_per_class=10):
    """
    Partition FMoW clients by country with disjoint labels and Dirichlet-controlled imbalance.
    """
    np.random.seed(seed)

    group_dict = defaultdict(list)
    for idx, (reg, yr, y) in enumerate(zip(metadata['region'],
                                           metadata['year'],
                                           metadata['label'])):
        dom = (reg, int(yr))                
        group_dict[(dom, y)].append(idx)

    all_domains = sorted({dom for dom, _ in group_dict})
    np.random.shuffle(all_domains)

    client_to_domains = defaultdict(list)
    for i, dom in enumerate(all_domains):
        cid = i % num_clients
        client_to_domains[cid].append(dom)

    all_labels = sorted(set(metadata['label']))
    labels_per_client = np.array_split(all_labels, num_clients)

    client_indices = defaultdict(list)
    for cid in range(num_clients):
        allowed_labels = labels_per_client[cid]
        if len(allowed_labels) == 0:
            continue

        class_prop = np.random.dirichlet(alpha=np.ones(len(allowed_labels)) * rho)
        label_list = sorted(map(int, allowed_labels))

        for dom in client_to_domains[cid]:
            for j, y in enumerate(label_list):
                bucket = (dom, y)
                if bucket not in group_dict:
                    continue
                indices = group_dict[bucket]
                np.random.shuffle(indices)
                n_pick = max(int(round(class_prop[j] * len(indices))),
                             min_samples_per_class)
                client_indices[cid].extend(indices[:n_pick])

    return dict(client_indices)

def partition_iwildcam_disjoint_labels(metadata, num_clients, rho=0.3, seed=42, min_samples_per_class=10):
    """
    Partition iWildCam clients by camera location with disjoint labels and Dirichlet-controlled imbalance.
    """
    np.random.seed(seed)
    domain_col = 'location'
    label_col = 'label'

    group_dict = defaultdict(list)
    for i, (loc, y) in enumerate(zip(metadata[domain_col], metadata[label_col])):
        group_dict[(loc, y)].append(i)

    locations = sorted(set(metadata[domain_col]))
    np.random.shuffle(locations)
    client_to_locations = defaultdict(list)
    for i, loc in enumerate(locations):
        cid = i % num_clients
        client_to_locations[cid].append(loc)

    all_labels = sorted(set(metadata[label_col]))
    labels_per_client = np.array_split(all_labels, num_clients)

    client_indices = defaultdict(list)
    for cid in range(num_clients):
        allowed_labels = set(labels_per_client[cid])
        class_prop = np.random.dirichlet(alpha=np.ones(len(allowed_labels)) * rho)
        label_list = sorted(allowed_labels)

        for loc in client_to_locations[cid]:
            for j, y in enumerate(label_list):
                key = (loc, y)
                if key not in group_dict:
                    continue
                indices = group_dict[key]
                np.random.shuffle(indices)
                count = max(int(round(class_prop[j] * len(indices))), min_samples_per_class)
                client_indices[cid].extend(indices[:count])

    return dict(client_indices)

def partition_fmow_joint_dirichlet(metadata, num_clients, rho=0.3, seed=42, min_samples=0):
    """
    Jointly non-IID partition of FMoW using (country, label) groups + Dirichlet allocation.
    """
    np.random.seed(seed)
    domain_col = 'region'   # or 'region' if preferred
    label_col = 'label'

    # 1. Group by (country, label)
    group_dict = defaultdict(list)
    for i, (d, y) in enumerate(zip(metadata[domain_col], metadata[label_col])):
        group_dict[(d, y)].append(i)

    client_indices = defaultdict(list)

    # 2. For each (domain, label), use Dirichlet to split into clients
    for group_key, idxs in group_dict.items():
        np.random.shuffle(idxs)
        proportions = np.random.dirichlet([rho] * num_clients)
        proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        splits = np.split(idxs, proportions)

        for cid, subset in enumerate(splits):
            if len(subset) >= min_samples:
                client_indices[cid].extend(subset)

    return dict(client_indices)

def partition_iwild_joint_dirichlet(metadata, num_clients, rho=0.3, seed=42, min_samples=0):
    """
    Jointly non-IID partition of FMoW using (country, label) groups + Dirichlet allocation.
    """
    np.random.seed(seed)
    domain_col = 'location'   # or 'region' if preferred
    label_col = 'label'

    # 1. Group by (country, label)
    group_dict = defaultdict(list)
    for i, (d, y) in enumerate(zip(metadata[domain_col], metadata[label_col])):
        group_dict[(d, y)].append(i)

    client_indices = defaultdict(list)

    # 2. For each (domain, label), use Dirichlet to split into clients
    for group_key, idxs in group_dict.items():
        np.random.shuffle(idxs)
        proportions = np.random.dirichlet([rho] * num_clients)
        proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        splits = np.split(idxs, proportions)

        for cid, subset in enumerate(splits):
            if len(subset) >= min_samples:
                client_indices[cid].extend(subset)

    return dict(client_indices)

def save_client_data(softmax, labels, client_indices, output_dir, prefix="client"):
    os.makedirs(output_dir, exist_ok=True)
    failed_clients = []
    written_files = []

    for client_id, indices in client_indices.items():
        path = os.path.join(output_dir, f"{prefix}_{client_id}.npz")

        try:
            client_softmax = softmax[indices]
            client_labels = labels[indices]

            np.savez(path, softmax=client_softmax, labels=client_labels)
            written_files.append(path)

            if not os.path.exists(path) or os.path.getsize(path) == 0:
                raise IOError(f"File {path} not written correctly.")

            print(f"[✓] Saved client {client_id} ({len(indices)} samples)")

        except Exception as e:
            print(f"[✗] Failed to save client {client_id}: {e}")
            failed_clients.append(client_id)

    if failed_clients:
        print(f"[Error] {len(failed_clients)} clients failed to save: {failed_clients}")
        print("[Action] Cleaning up all written files.")
        for f in written_files:
            if os.path.exists(f):
                os.remove(f)
        raise RuntimeError("Client data saving failed. Aborting.")

def load_client_data(output_dir, num_clients, prefix="client"):
    client_data = {}

    for client_id in range(num_clients):
        path = os.path.join(output_dir, f"{prefix}_{client_id}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"[load_client_data] Missing file: {path}")
        if os.path.getsize(path) == 0:
            raise ValueError(f"[load_client_data] Empty file: {path}")

        data = np.load(path)
        client_data[client_id] = {
            "softmax": data['softmax'],
            "labels": data['labels']
        }

    return client_data

def prepare_or_load_client_data(
    softmax, labels, metadata, dataset,
    output_dir, num_clients, args, partition_type='iid', 
    seed=42, prefix='client'
):
    """
    Prepare or load client-partitioned data and return both client data and indices.

    Returns:
        client_data (dict): {client_id: {'softmax': ..., 'labels': ...}}
        client_indices (dict): {client_id: list of indices into original softmax/labels arrays}
    """
    index_path = os.path.join(output_dir, "client_indices.pkl")

    # ---------- Check if all files and index exist ----------
    all_client_files_exist = True
    for i in range(num_clients):
        path = os.path.join(output_dir, f"{prefix}_{i}.npz")
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"[Warn] Missing or empty file: {path}")
            all_client_files_exist = False
            break

    if all_client_files_exist and os.path.exists(index_path):
        print(f"[✓] Loading existing client files and indices from {output_dir}")
        client_data = load_client_data(output_dir, num_clients, prefix)
        with open(index_path, "rb") as f:
            client_indices = pickle.load(f)
        return client_data, client_indices

    # ---------- Otherwise: partition + save ----------
    print(f"[⚙] Partitioning and saving client files to {output_dir}")
    if partition_type == 'iid':
        client_indices = partition_iid_2(len(labels), num_clients, seed)
    elif partition_type == 'non-iid':
        client_indices = partition_non_iid_2(dataset, metadata, args)
    else:
        raise ValueError("partition_type must be 'iid' or 'non-iid'")

    # Save .npz files and check success
    try:
        save_client_data(softmax, labels, client_indices, output_dir, prefix)
    except RuntimeError as e:
        print(f"[Fatal] save_client_data() failed: {e}")
        if os.path.exists(index_path):
            os.remove(index_path)
        raise

    # Save partition index
    with open(index_path, "wb") as f:
        pickle.dump(client_indices, f)

    client_data = load_client_data(output_dir, num_clients, prefix)
    return client_data, client_indices


# def prepare_or_load_client_data(
#     softmax, labels, metadata, dataset,
#     output_dir, num_clients, partition_type='iid',
#     seed=42, prefix='client'
# ):
#     """
#     Prepare or load client-partitioned data and return both client data and indices.

#     Returns:
#         client_data (dict): {client_id: {'softmax': ..., 'labels': ...}}
#         client_indices (dict): {client_id: list of indices into original softmax/labels arrays}
#     """
#     index_path = os.path.join(output_dir, "client_indices.pkl")
#     ready = all(os.path.exists(os.path.join(output_dir, f"{prefix}_{i}.npz")) for i in range(num_clients)) \
#             and os.path.exists(index_path)
    
#     for i in range(num_clients):
#         path = os.path.join(output_dir, f"{prefix}_{i}.npz")
#         if not os.path.exists(path) or os.path.getsize(path) == 0:
#             ready = False
#             print(f"[Warn] Missing or empty file: {path}")
#             break


#     if ready:
#         print(f"Loading existing client files and indices from {output_dir}")
#         client_data = load_client_data(output_dir, num_clients, prefix)
#         with open(index_path, "rb") as f:
#             client_indices = pickle.load(f)
#         return client_data, client_indices

#     # Partition and save
#     print(f"Partitioning and saving client files to {output_dir}")
#     if partition_type == 'iid':
#         client_indices = partition_iid_2(len(labels), num_clients, seed)
#     elif partition_type == 'non-iid':
#         client_indices = partition_non_iid(dataset, metadata, num_clients)
#     else:
#         raise ValueError("partition_type must be 'iid' or 'non-iid'")

#     # Save data and indices
#     save_client_data(softmax, labels, client_indices, output_dir, prefix)
#     with open(index_path, "wb") as f:
#         pickle.dump(client_indices, f)

#     client_data = load_client_data(output_dir, num_clients, prefix)
#     return client_data, client_indices

def split_features_by_client(features, client_indices):
    """
    Given all features and client index mapping, return client-wise features.

    Args:
        features: numpy array or torch tensor of shape (N, d)
        client_indices: dict mapping client_id to list of indices

    Returns:
        client_features: dict {client_id: features_subset}
    """
    if isinstance(features, torch.Tensor):
        return {cid: features[torch.tensor(idxs)] for cid, idxs in client_indices.items()}
    else:  # numpy fallback
        return {cid: features[idxs] for cid, idxs in client_indices.items()}

def load_federated_data(train_dataset, test_dataset, num_clients=10, iid=True, num_bins=5, batch_size=64):
    """Returns {client_id: (train_subset, test_subset)} mapping."""
    
    # Partition train dataset
    if iid:
        train_partitions = partition_iid(train_dataset, num_clients)
        test_partitions = partition_iid(test_dataset, num_clients)
    else:
        train_partitions = partition_non_iid(train_dataset, num_clients, num_bins)
        test_partitions = partition_non_iid(test_dataset, num_clients, num_bins)


    # Construct per-client datasets
    client_data_loaders = {}

    for cid in range(num_clients):
        client_train_dataset = Subset(train_dataset, train_partitions[cid])
        client_test_dataset = Subset(test_dataset, test_partitions[cid])
        # client_datasets[cid] = (client_train_dataset, client_test_dataset)

        train_loader = DataLoader(client_train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(client_test_dataset, batch_size=500, shuffle=False)

        client_data_loaders[cid] = (train_loader, test_loader)

    return client_data_loaders

# def load_softmax_scores(model, client_data_loaders, output_dir='client_softmax_scores', device='cuda'):
#     os.makedirs(output_dir, exist_ok=True)
#     model.eval()

#     client_softmax_data = {}

#     for client_id, (train_loader, test_loader) in client_data_loaders.items():
#         client_dir = os.path.join(output_dir, f'client_{client_id}')
#         os.makedirs(client_dir, exist_ok=True)
#         client_softmax_data[client_id] = {}

#         for split_name, loader in [('train', train_loader), ('test', test_loader)]:
#             score_path = os.path.join(client_dir, f'{split_name}_scores.pt')
#             label_path = os.path.join(client_dir, f'{split_name}_labels.pt')

#             # Load if exists
#             if os.path.exists(score_path) and os.path.exists(label_path):
#                 scores = torch.load(score_path)
#                 labels = torch.load(label_path)
#                 print(f"[Client {client_id}] Loaded {split_name} scores from cache.")
#             else:
#                 # Compute softmax scores
#                 all_scores = []
#                 all_labels = []

#                 with torch.no_grad():
#                     for x, y in loader:
#                         x = x.to(device)
#                         logits = model(x)
#                         probs = F.softmax(logits, dim=1).cpu()
#                         all_scores.append(probs)
#                         all_labels.append(y)

#                 scores = torch.cat(all_scores, dim=0)
#                 labels = torch.cat(all_labels, dim=0)

#                 # Save
#                 torch.save(scores, score_path)
#                 torch.save(labels, label_path)
#                 print(f"[Client {client_id}] Computed and saved {split_name} scores.")

#             # Store in dict
#             client_softmax_data[client_id][f'{split_name}_scores'] = scores
#             client_softmax_data[client_id][f'{split_name}_labels'] = labels

#     return client_softmax_data

def load_softmax_scores(model, client_data_loaders, output_dir='client_softmax_scores', device='cuda'):
    """
    Compute or load softmax scores on the test set only for each client.
    
    Args:
        model: classification model
        client_data_loaders: dict[client_id] = (train_loader, test_loader)
        output_dir: base directory to save/load scores
        device: torch device

    Returns:
        client_softmax_data: dict[client_id] = {'test_scores': Tensor, 'test_labels': Tensor}
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    client_softmax_data = {}

    for client_id, (_, test_loader) in client_data_loaders.items():
        client_dir = os.path.join(output_dir, f'client_{client_id}')
        os.makedirs(client_dir, exist_ok=True)
        client_softmax_data[client_id] = {}

        score_path = os.path.join(client_dir, 'test_scores.pt')
        label_path = os.path.join(client_dir, 'test_labels.pt')

        if os.path.exists(score_path) and os.path.exists(label_path):
            scores = torch.load(score_path)
            labels = torch.load(label_path)
            print(f"[Client {client_id}] Loaded test scores from cache.")
        else:
            all_scores = []
            all_labels = []

            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device)
                    logits = model(x)
                    probs = F.softmax(logits, dim=1).cpu()
                    all_scores.append(probs)
                    all_labels.append(y)

            scores = torch.cat(all_scores, dim=0)
            labels = torch.cat(all_labels, dim=0)

            torch.save(scores, score_path)
            torch.save(labels, label_path)
            print(f"[Client {client_id}] Computed and saved test scores.")

        client_softmax_data[client_id]['test_scores'] = scores
        client_softmax_data[client_id]['test_labels'] = labels

    return client_softmax_data

# def compute_conformal_scores_all_clients(
#     client_softmax_data,
#     score_function='APS',
#     args
# ):
#     """
#     Compute APS or RAPS scores per client for train and test sets.

#     Args:
#         client_softmax_data: dict of client_id -> dict with 'train_scores', 'train_labels', etc.
#         score_function: one of {'APS', 'RAPS', 'HPS', 'RHPS'}
#         lam: lambda parameter for RAPS/RHPS
#         k_r: k_reg parameter for RAPS/RHPS

#     Returns:
#         client_scores: dict[client_id][split] = score_tensor
#             where split ∈ {'train', 'test'}
#     """

#     client_scores = {}

#     for client_id, data in client_softmax_data.items():
#         train_softmax = data['train_scores']
#         train_labels = data['train_labels']
#         test_softmax = data['test_scores']
#         test_labels = data['test_labels']  

#         if score_function == 'HPS':
#             train_scores = get_HPS_scores(train_softmax, train_labels)
#             val_scores = get_HPS_scores(test_softmax, test_labels)
#             val_scores_all = 1 - test_softmax

#         elif score_function == 'APS':
#             train_scores = get_APS_scores(train_softmax, train_labels, randomize=True)
#             val_scores = get_APS_scores(test_softmax, test_labels, randomize=True)
#             val_scores_all = get_APS_scores_all(test_softmax, randomize=True)

#         elif score_function == 'RAPS':
#         	lam = args.lmbda_val
#         	k_r = args.k_reg
#             train_scores = get_RAPS_scores(train_softmax, train_labels, lam, k_r, randomize=True)
#             val_scores = get_RAPS_scores(test_softmax, test_labels, lam, k_r, randomize=True)
#             val_scores_all = get_RAPS_scores_all(test_softmax, lam, k_r, randomize=True)

#         else:
#             raise ValueError(f"Unknown score_function: {score_function}")

#         client_scores[client_id] = {
#             'train': train_scores,
#             'test': val_scores,
#             'test_all': val_scores_all,
#             'train_labels': train_labels,
#             'test_labels': test_labels,
#         }

#     return client_scores

def compute_conformal_scores_all_clients(
    client_softmax_data,
    seed=42, 
    score_function='APS',
    args=None
):
    """
    Split test data into calibration/test subsets per client, and compute scores.

    Args:
        client_softmax_data: dict[client_id] = {'test_scores': Tensor, 'test_labels': Tensor}
        score_function: str in {'APS', 'RAPS', 'HPS'}
        args: contains lam (lambda) and k_reg for RAPS
        seed: random seed for reproducibility

    Returns:
        client_scores: dict[client_id] = {
            'calib': ..., 'test': ..., 'test_all': ...,
            'calib_labels': ..., 'test_labels': ...
        }
    """
    client_scores = {}

    for client_id, data in client_softmax_data.items():
        test_softmax = data['test_scores']
        test_labels = data['test_labels']

        n = len(test_labels)
        idx_calib, idx_test = train_test_split(
            np.arange(n),
            train_size=0.5,
            random_state=seed + 1111
        )

        calib_softmax = test_softmax[idx_calib]
        calib_labels = test_labels[idx_calib]
        test_softmax_split = test_softmax[idx_test]
        test_labels_split = test_labels[idx_test]

        # Compute conformal scores
        if score_function == 'HPS':
            calib_scores = get_HPS_scores(calib_softmax, calib_labels)
            # test_scores = get_HPS_scores(test_softmax_split, test_labels_split)
            test_scores = 1 - test_softmax_split

        elif score_function == 'APS':
            calib_scores = get_APS_scores(calib_softmax, calib_labels, randomize=True)
            # test_scores = get_APS_scores(test_softmax_split, test_labels_split, randomize=True)
            test_scores = get_APS_scores_all(test_softmax_split, randomize=True)

        elif score_function == 'RAPS':
            lam = args.lmbda_val
            k_r = args.k_reg
            calib_scores = get_RAPS_scores(calib_softmax, calib_labels, lam, k_r, randomize=True)
            # test_scores = get_RAPS_scores(test_softmax_split, test_labels_split, lam, k_r, randomize=True)
            test_scores = get_RAPS_scores_all(test_softmax_split, lam, k_r, randomize=True)

        else:
            raise ValueError(f"Unknown score_function: {score_function}")

        client_scores[client_id] = {
            'calib': calib_scores,
            'test': test_scores,
            'calib_labels': calib_labels,
            'test_labels': test_labels_split,
        }

    return client_scores

#========================================
#   Homogeneous Prediction Sets (HPS)
#========================================

def get_HPS_scores_2(softmax_scores, labels):
    '''
    Compute homogeneous conformity score 
    
    Inputs:
        softmax_scores: n x num_classes
        labels: length-n array of class labels
    
    Output: 
        length-n array of HPS scores
    '''
    n = len(labels)
    labels = labels.astype(int)
    scores = softmax_scores[np.arange(n), labels]
    hps_scores = 1 - scores

    return hps_scores

#========================================
#   Adaptive Prediction Sets (APS)
#========================================

def get_APS_scores_2(softmax_scores, labels, randomize=True, seed=0):
    '''
    Compute conformity score defined in Romano et al, 2020
    (Including randomization, unless randomize is set to False)
    
    Inputs:
        softmax_scores: n x num_classes
        labels: length-n array of class labels
    
    Output: 
        length-n array of APS scores
    '''
    n = len(labels)
    labels = labels.astype(int)
    sorted, pi = torch.from_numpy(softmax_scores).sort(dim=1, descending=True) # pi is the indices in the original array
    scores = sorted.cumsum(dim=1).gather(1,pi.argsort(1))[range(n), labels]
    scores = np.array(scores)
    
    if not randomize:
        return scores - softmax_scores[range(n), labels]
    else:
        np.random.seed(seed)
        U = np.random.rand(n) # Generate U's ~ Unif([0,1])
        randomized_scores = scores - U * softmax_scores[range(n), labels]
        return randomized_scores
    
def get_APS_scores_all_2(softmax_scores, randomize=True, seed=0):
    '''
    Similar to get_APS_scores(), except the APS scores are computed for all 
    classes instead of just the true label
    
    Inputs:
        softmax_scores: n x num_classes
    
    Output: 
        n x num_classes array of APS scores
    '''
    n = softmax_scores.shape[0]
    sorted, pi = torch.from_numpy(softmax_scores).sort(dim=1, descending=True) # pi is the indices in the original array
    scores = sorted.cumsum(dim=1).gather(1,pi.argsort(1))
    scores = np.array(scores)
    
    if not randomize:
        return scores - softmax_scores
    else:
        np.random.seed(seed)
        U = np.random.rand(*softmax_scores.shape) # Generate U's ~ Unif([0,1])
        randomized_scores = scores - U * softmax_scores # [range(n), labels]
        return randomized_scores

#========================================
#   Regularized Adaptive Prediction Sets (RAPS)
#========================================

def get_RAPS_scores_2(softmax_scores, labels, lmbda=.01, kreg=5, randomize=True, seed=0):
    '''
    Essentially the same as get_APS_scores() except with regularization.
    See "Uncertainty Sets for Image Classifiers using Conformal Prediction" (Angelopoulos et al., 2021)
    
    Inputs:
        softmax_scores: n x num_classes
        labels: length-n array of class labels
        lmbda, kreg: regularization parameters
    Output: 
        length-n array of APS scores
    
    '''
    n = len(labels)
    labels = labels.astype(int)
    sorted, pi = torch.from_numpy(softmax_scores).sort(dim=1, descending=True) # pi is the indices in the original array
    scores = sorted.cumsum(dim=1).gather(1,pi.argsort(1))[range(n), labels]
    
    # Regularization
    # y_rank = pi.argsort(1)[range(labels_calib.shape[0]), labels_calib] + 1 # Compute softmax rank of true labels y
    y_rank = pi.argsort(1)[range(n), labels] + 1
    reg = torch.maximum(lmbda * (y_rank - kreg), torch.zeros(size=y_rank.shape))
    scores += reg
    
    scores = np.array(scores)
    
    if not randomize:
        return scores - softmax_scores[range(n), labels]
    else:
        np.random.seed(seed)
        U = np.random.rand(n) # Generate U's ~ Unif([0,1])
        randomized_scores = scores - U * softmax_scores[range(n), labels]
        return randomized_scores
        
def get_RAPS_scores_all_2(softmax_scores, lmbda, kreg, randomize=True, seed=0):
    '''
    Similar to get_RAPS_scores(), except the RAPS scores are computed for all 
    classes instead of just the true label
    
    Inputs:
        softmax_scores: n x num_classes
    
    Output: 
        n x num_classes array of APS scores
    '''
    n = softmax_scores.shape[0]
    sorted, pi = torch.from_numpy(softmax_scores).sort(dim=1, descending=True) # pi is the indices in the original array
    scores = sorted.cumsum(dim=1).gather(1,pi.argsort(1))
    
    # Regularization (pretend each class is true label)
    y_rank = pi.argsort(1) + 1 # Compute softmax rank of true labels y
    reg = torch.maximum(lmbda * (y_rank - kreg), torch.zeros(size=scores.shape))
 
    scores += reg
        
    if not randomize:
        return scores - softmax_scores
    else:
        np.random.seed(seed)
        U = np.random.rand(*softmax_scores.shape) # Generate U's ~ Unif([0,1])
        randomized_scores = scores - U * softmax_scores # [range(n), labels]
        return randomized_scores


def compute_conformal_scores_all_clients_2(
    client_softmax_data,
    seed=42, 
    score_function='APS',
    args=None
):
    """
    Split client softmax/labels into calibration/test and compute conformal scores.

    Args:
        client_softmax_data: dict[client_id] = {'softmax': ndarray, 'labels': ndarray}
        score_function: 'APS', 'RAPS', or 'HPS'
        args: config with args.lmbda_val and args.k_reg for RAPS
        seed: random seed

    Returns:
        client_scores: dict[client_id] = {
            'calib': ..., 'test': ...,
            'calib_labels': ..., 'test_labels': ...
        }
    """
    client_scores = {}

    for client_id, data in client_softmax_data.items():
        softmax = data['softmax']
        labels = data['labels']
        n = len(labels)

        # Split into calibration and test
        idx_calib, idx_test = train_test_split(
            np.arange(n),
            train_size=0.5,
            random_state=seed + 1111
        )
        
        calib_softmax = softmax[idx_calib]
        calib_labels = labels[idx_calib]
        test_softmax = softmax[idx_test]
        test_labels = labels[idx_test]

        # Compute conformal scores
        if score_function == 'HPS':
            calib_scores = get_HPS_scores_2(calib_softmax, calib_labels)
            test_scores = 1 - test_softmax

        elif score_function == 'APS':
            calib_scores = get_APS_scores_2(calib_softmax, calib_labels, randomize=True)
            test_scores = get_APS_scores_all_2(test_softmax, randomize=True)

        elif score_function == 'RAPS':
            lam = args.lmbda_val
            k_r = args.k_reg
            calib_scores = get_RAPS_scores_2(calib_softmax, calib_labels, lam, k_r, randomize=True)
            test_scores = get_RAPS_scores_all_2(test_softmax, lam, k_r, randomize=True)

        else:
            raise ValueError(f"Unknown score_function: {score_function}")

        client_scores[client_id] = {
            'calib': calib_scores,
            'test': test_scores,
            'calib_labels': calib_labels,
            'test_labels': test_labels,
        }

    return client_scores

def compute_conformal_scores_all_clients_3(
    client_softmax_data,
    seed=42, 
    score_function='APS',
    args=None,
    client_features_dict=None  
):
    """
    Split client softmax/labels/features into calibration/test and compute conformal scores.

    Returns:
        client_scores: dict[client_id] = {
            'calib': ..., 'test': ...,
            'calib_labels': ..., 'test_labels': ...,
            'calib_features': ..., 'test_features': ...
        }
    """
    client_scores = {}

    for client_id, data in client_softmax_data.items():
        softmax = data['softmax']
        labels = data['labels']
        n = len(labels)

        # Split into calibration and test
        idx_calib, idx_test = train_test_split(
            np.arange(n),
            train_size=0.5,
            random_state=seed + 1111
        )

        # Softmax + labels
        calib_softmax = softmax[idx_calib]
        calib_labels = labels[idx_calib]
        test_softmax = softmax[idx_test]
        test_labels = labels[idx_test]

        # Feature split (if available)
        if client_features_dict is not None:
            features = client_features_dict[client_id]
            calib_features = features[idx_calib]
            test_features = features[idx_test]
        else:
            calib_features = None
            test_features = None

        # Compute conformal scores
        if score_function == 'HPS':
            calib_scores = get_HPS_scores_2(calib_softmax, calib_labels)
            test_scores = 1 - test_softmax

        elif score_function == 'APS':
            calib_scores = get_APS_scores_2(calib_softmax, calib_labels, randomize=True)
            test_scores = get_APS_scores_all_2(test_softmax, randomize=True)

        elif score_function == 'RAPS':
            lam = args.lmbda_val
            k_r = args.k_reg
            calib_scores = get_RAPS_scores_2(calib_softmax, calib_labels, lam, k_r, randomize=True)
            test_scores = get_RAPS_scores_all_2(test_softmax, lam, k_r, randomize=True)

        else:
            raise ValueError(f"Unknown score_function: {score_function}")

        client_scores[client_id] = {
            'calib': calib_scores,
            'test': test_scores,
            'calib_labels': calib_labels,
            'test_labels': test_labels,
            'calib_features': calib_features,     
            'test_features': test_features        
        }

    return client_scores

def estimate_likelihood_ratios_scaled(X_train, X_test):
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((np.zeros(len(X_train)), np.ones(len(X_test))))

    clf = LogisticRegression()
    clf.fit(X_combined, y_combined)

    probs_test = clf.predict_proba(X_test)[:, 1]
    likelihood_ratios = probs_test / (1 - probs_test)

    capped_likelihood_ratios = np.clip(likelihood_ratios, 1e-6, 1 - 1e-6)

    # print(f"Capped Likelihood Ratios: {capped_likelihood_ratios}")

    return capped_likelihood_ratios

def estimate_empirical_likelihood_ratios(client_scores_dict, all_scores, min_cdf=1e-6):
    """
    Estimate importance weights omega_i^k = dF(V_i)/dF^k(V_i)
    using empirical CDFs.

    Args:
        client_scores_dict: dict[cid] = {'train': conformity_scores (Tensor)}
        min_cdf: minimum value to clip local CDFs to avoid division by zero

    Returns:
        importance_weights: dict[cid] = Tensor[N_k] (importance weights per score)
    """

    # Step 1: concatenate all global train scores
    # all_scores = torch.cat([v['train'] for v in client_scores_dict.values()], dim=0).numpy()
    N_global = len(all_scores)
    sorted_global = np.sort(all_scores)

    client_weights = {}

    for cid, data in client_scores_dict.items():
        local_scores = data['calib'].numpy()

        N_local = len(local_scores)
        sorted_local = np.sort(local_scores)

        # Step 2: Compute empirical CDFs at each V_i
        # Empirical CDF = proportion of values ≤ V_i
        F_global = np.searchsorted(sorted_global, local_scores, side='right') / N_global
        F_local = np.searchsorted(sorted_local, local_scores, side='right') / N_local

        # Step 3: Clip to avoid division by zero
        F_local_clipped = np.clip(F_local, min_cdf, None)

        # Step 4: Compute importance weights
        omega = F_global / F_local_clipped
        client_weights[cid] = torch.tensor(omega, dtype=torch.float32)

    return client_weights

# def nonconformity_scores_scaled(model, X_train, y_train):
#     model.fit(X_train, y_train)
#     y_pred_train = model.predict(X_train)
#     residuals = np.abs(y_train - y_pred_train)

#     # Apply an additional scaling factor if necessary
#     scaled_residuals = residuals / np.std(residuals)

#     print(f"Scaled Residuals (nonconformity scores): {scaled_residuals}")

#     return scaled_residuals

# # Function to calculate normalized nonconformity scores
# def nonconformity_scores_normalized(model, X_train, y_train):
#     model.fit(X_train, y_train)
#     y_pred_train = model.predict(X_train)
#     residuals = np.abs(y_train - y_pred_train)

#     # Normalize residuals
#     normalized_residuals = residuals / np.std(residuals)

#     print(f"Normalized Residuals (nonconformity scores): {normalized_residuals}")

#     return normalized_residuals

def normalize_client_weights(client_weights_dict):
    """
    Normalize weights per client: omega_i^k / sum_j omega_j^k

    Args:
        client_weights_dict: dict[cid] = Tensor[N_k]

    Returns:
        normalized_weights_dict: dict[cid] = Tensor[N_k]
    """
    normalized_weights = {}
    for cid, weights in client_weights_dict.items():
        total = weights.sum()
        if total.item() == 0:
            raise ValueError(f"Client {cid} has zero total weight, cannot normalize.")
        normalized_weights[cid] = weights / total
    return normalized_weights

def weighted_scores(client_scores_dict, all_scores, method):
    key = 'calib' 

    likelihood_ratios = estimate_empirical_likelihood_ratios(client_scores_dict, all_scores, min_cdf=1e-6)
    
    normalized_weights = normalize_client_weights(likelihood_ratios)

    weighted_scores_dict = {}

    for cid in client_scores_dict:
        scores = client_scores_dict[cid][key]        # shape: [N_k]
        weights = normalized_weights[cid]            # shape: [N_k]
        weighted = weights * scores                  # shape: [N_k]
        # weighted = 1.0 * scores                  # shape: [N_k]
        
        weighted_scores_dict[cid] = weighted

        if torch.all(weighted == 0):
            print(f"[Client {cid}] Warning: all weighted scores are zero.")

    return weighted_scores_dict

def estimate_empirical_likelihood_ratios_2(client_scores_dict, all_scores, t, min_cdf=1e-6):
    """
    Estimate importance weights omega_i^k = dF(V_i)/dF^k(V_i) using empirical CDFs.
    """
    N_global = len(all_scores)
    sorted_global = np.sort(all_scores)

    client_weights = {}

    for cid, data in client_scores_dict.items():
        local_scores = data['calib']  # assumed to be numpy array
        N_local = len(local_scores)
        sorted_local = np.sort(local_scores)

        F_global = np.searchsorted(sorted_global, local_scores, side='right') / N_global
        F_local = np.searchsorted(sorted_local, local_scores, side='right') / N_local
        F_local_clipped = np.clip(F_local, min_cdf, None)

        # omega = F_global / F_local_clipped
        omega = np.exp(t * np.log(F_global / F_local_clipped))
        client_weights[cid] = omega  # keep as numpy array

    return client_weights


def estimate_client_label_counts(client_scores_dict, num_classes):
    """
    Estimate count M^i_y and total M^i for each client using calibration data.
    Returns:
        label_counts: dict[client_id] = ndarray[num_classes]
        client_totals: dict[client_id] = scalar M^i
    """
    label_counts = {}
    client_totals = {}

    for cid, scores_dict in client_scores_dict.items():
        labels = scores_dict['calib_labels']
        count = np.bincount(labels, minlength=num_classes)
        label_counts[cid] = count
        client_totals[cid] = len(labels)

    return label_counts, client_totals

def compute_global_label_counts(label_counts):
    """
    Aggregate M_y (total count of label y across all clients) and global total M.
    Returns:
        global_counts: ndarray[num_classes]
        global_total: scalar
    """
    num_classes = len(next(iter(label_counts.values())))
    global_counts = np.zeros(num_classes)
    global_total = 0

    for count in label_counts.values():
        global_counts += count
        global_total += count.sum()

    return global_counts, global_total

# def compute_omega_y(target_client_id, label_counts, client_totals, global_counts, global_total, num_classes, sigma=0.0):
#     """
#     Compute ω_y for target client using Equation (10), optionally adding Gaussian noise to counts.
#     """
#     target_count = label_counts[target_client_id].astype(float)
#     if sigma > 0:
#         target_count += np.random.normal(0, sigma, size=target_count.shape)
#         target_count = np.clip(target_count, 1e-6, None)

#     M_target = client_totals[target_client_id]
#     omega_y = np.zeros(num_classes)

#     for y in range(num_classes):
#         if global_counts[y] >= 1:
#             omega_y[y] = (global_total * target_count[y]) / (M_target * global_counts[y])
#         else:
#             omega_y[y] = 1.0  # fallback

#     return omega_y

def compute_omega_y(target_client_id, label_counts, client_totals, global_counts, global_total, num_classes, sigma=0.0):
    target_count = label_counts[target_client_id].astype(float)

    if sigma > 0:
        target_count += np.random.normal(0, sigma, size=target_count.shape)
        target_count = np.clip(target_count, 1e-6, None)  

    M_target = client_totals[target_client_id]
    omega_y = np.zeros(num_classes)

    for y in range(num_classes):
        if global_counts[y] >= 1:
            omega_y[y] = (global_total * target_count[y]) / (M_target * global_counts[y])
        else:
            omega_y[y] = 1.0  # fallback

    omega_y = np.clip(omega_y, 1e-6, None)  
    return omega_y

def compute_score_weights(client_scores_dict, label_counts, client_totals, global_counts, global_total, args):
    """
    Compute per-sample weights w_ik for all calibration scores, and return total sum for normalization.
    """
    score_weights = {}
    denom_sum = 0.0

    for cid in client_scores_dict:
        labels = client_scores_dict[cid]['calib_labels']
        count_i = label_counts[cid]
        M_i = client_totals[cid]
        weights = []
        for yk in labels:
            if global_counts[yk] >= 1:
                w = (global_total * count_i[yk]) / (M_i * global_counts[yk])
            else:
                w = 1.0
            weights.append(w)
            denom_sum += w
        score_weights[cid] = np.array(weights)
    
    denom_sum /= args.num_class

    return score_weights, denom_sum

# def compute_score_weights(client_scores_dict, label_counts, client_totals, global_counts, global_total):
#     """
#     Compute per-sample weights w_ik for all calibration scores,
#     then normalize so that the total sum is 1 (global weight normalization).
#     """
#     score_weights = {}
#     total_weight = 0.0

#     # First pass: compute all weights and total sum
#     for cid in client_scores_dict:
#         labels = client_scores_dict[cid]['calib_labels']
#         count_i = label_counts[cid]
#         M_i = client_totals[cid]
#         weights = []

#         for yk in labels:
#             if global_counts[yk] >= 1:
#                 w = (global_total * count_i[yk]) / (M_i * global_counts[yk])
#             else:
#                 w = 1.0
#             weights.append(w)
#             total_weight += w

#         score_weights[cid] = np.array(weights)

#     # Second pass: normalize all weights so total sum is 1
#     for cid in score_weights:
#         score_weights[cid] /= total_weight

#     denom_sum = 1.0  # since all weights now sum to 1
#     return score_weights, denom_sum

def compute_p_star_y(omega_y, denom_sum):
    p_star_dict = {}
    for y in range(len(omega_y)):
        raw_p = omega_y[y] / (omega_y[y] + denom_sum)
        p_star_dict[y] = np.clip(raw_p, 1e-4, 0.99)  # 
    return p_star_dict

# def compute_p_star_y_from_omega(omega_y, alpha):
#     """
#     Normalize omega_y to obtain p*_y, and scale to ensure total sum is (1 - alpha).
#     """
#     omega_sum = np.sum(omega_y)
#     if omega_sum < 1e-8:
#         # fallback to uniform
#         return {y: (1 - alpha) / len(omega_y) for y in range(len(omega_y))}
#     else:
#         scale = (1 - alpha) / omega_sum
#         return {y: omega_y[y] * scale for y in range(len(omega_y))}

def compute_p_star_y_from_omega(omega_y, alpha):
    """
    Normalize omega_y to obtain p*_y, and scale to ensure total sum is (1 - alpha).
    """
    omega_sum = np.sum(omega_y)
    if omega_sum < 1e-8:
        return {y: (1 - alpha) / len(omega_y) for y in range(len(omega_y))}
    else:
        scale = (1 - alpha) / omega_sum
        return {y: omega_y[y] * scale for y in range(len(omega_y))}

def normalize_client_class_weights(all_omega_y_dict):
    """
    Normalize omega_y per client so that for each client:
        sum_y p_y^* = 1 - alpha

    Args:
        all_omega_y_dict: dict[cid] = array-like of shape [num_classes]
        alpha: miscoverage level (e.g., 0.1)

    Returns:
        all_p_star_dicts: dict[cid] = dict[y] = p_y^*
    """
    all_p_star_dicts = {}

    for cid, omega_y in all_omega_y_dict.items():
        omega_y = np.array(omega_y)  # ensure numpy array
        total = np.sum(omega_y)

        if total < 1e-8:
            # fallback: uniform weights
            num_classes = len(omega_y)
            p_star = {y: 1 / num_classes for y in range(num_classes)}
        else:
            scale = 1 / total
            p_star = {y: omega_y[y] * scale for y in range(len(omega_y))}

        all_p_star_dicts[cid] = p_star

    return all_p_star_dicts


def build_weighted_scores(client_scores_dict, score_weights):
    """
    Multiply calibration scores by their respective weights w_ik.
    """
    weighted_scores_dict = {}
    for cid in client_scores_dict:
        scores = client_scores_dict[cid]['calib']
        weighted_scores = score_weights[cid] * scores
        weighted_scores_dict[cid] = weighted_scores
    return weighted_scores_dict


def normalize_client_class_weights(omega_dict):
    return {cid: w / w.sum() for cid, w in omega_dict.items()}

def weighted_quantile(values, weights, q):
    idx = np.argsort(values)
    v_sorted, w_sorted = values[idx], weights[idx]
    cum_w = np.cumsum(w_sorted)
    cutoff = q * cum_w[-1]
    return v_sorted[np.searchsorted(cum_w, cutoff, side="right")]

def weighted_scores_2(client_scores_dict, args, method='empirical_cdf'):
    """
    Compute importance-weighted conformity scores.
    """
    key = 'calib'

    if method == 'FCP_full':
    
        global_sketch_scores = federated_sketching(client_scores_dict, args.sketch_method, args.sigma, args.bins, args.num_sketch)

        likelihood_ratios = estimate_empirical_likelihood_ratios_2(client_scores_dict, global_sketch_scores, min_cdf=1e-6)

        # normalized_weights = normalize_client_weights_2(likelihood_ratios)

        weighted_scores_dict = {}
        for cid in client_scores_dict:
            scores = client_scores_dict[cid][key]         # numpy array
            # weights = normalized_weights[cid]             # numpy array
            weights = likelihood_ratios[cid]             # numpy array
            weighted = weights * scores
            # weighted = 1.0 * scores
            weighted_scores_dict[cid] = weighted

            if np.all(weighted == 0):
                print(f"[Client {cid}] Warning: all weighted scores are zero.")

        return weighted_scores_dict, None

    elif method == 'FCP_LS':
        # Step 1: compute global statistics
        label_counts, client_totals = estimate_client_label_counts(client_scores_dict, args.num_class)
        global_counts, global_total = compute_global_label_counts(label_counts)

        # Step 3: for each client, compute their own omega_y, p*_y, and weighted scores
        all_weighted_scores = {}
        all_p_star_dicts = {}
        all_omega_y_dict = {}

        for cid in client_scores_dict:
            omega_y = compute_omega_y(
                target_client_id=cid,
                label_counts=label_counts,
                client_totals=client_totals,
                global_counts=global_counts,
                global_total=global_total,
                num_classes=args.num_class,
                sigma=args.dp_noise if hasattr(args, 'dp_noise') else 0.0
            )

            # print(f"[Client {cid}] omega_y mean = {np.mean(omega_y):.4f}, max = {np.max(omega_y):.4f}, min = {np.min(omega_y):.4f}")
            all_omega_y_dict[cid] = omega_y
            # print("denom_sum =", denom_sum)
        
        all_p_star_dicts = normalize_client_class_weights(all_omega_y_dict)

        per_class_scores  = {y: [] for y in range(args.num_class)}
        per_class_weights = {y: [] for y in range(args.num_class)}

        for cid, cdict in client_scores_dict.items():
            omega = all_omega_y_dict[cid]                     # shape (C,)
            
            calib_scores  = np.asarray(cdict["calib"], dtype=float)        # (n_calib,)
            calib_labels  = np.asarray(cdict["calib_labels"], dtype=int)   # (n_calib,)

            for y in range(args.num_class):
                idx = (calib_labels == y)
                if not np.any(idx):            
                    continue

                scores_y = calib_scores[idx]                 # (n_y,)
                w_y = np.full_like(scores_y, omega[y], dtype=float)
                per_class_scores[y].append(scores_y)
                per_class_weights[y].append(w_y)

        q_vec = np.full(args.num_class, np.inf, dtype=float)
        all_weighted_scores = {}     # class → (scores, weights)

        for y in range(args.num_class):

            if len(per_class_scores[y]) == 0:
                all_weighted_scores[y] = (np.array([]), np.array([]))
                continue 

            s = np.concatenate(per_class_scores[y])           # shape (Ny,)
            w = np.concatenate(per_class_weights[y]).astype(float)
            if w.sum() == 0:                                  
                all_weighted_scores[y] = (s, w)  
                continue
            w /= w.sum()                                      
            q_vec[y] = weighted_quantile(s, w, 1 - args.alpha)
            all_weighted_scores[y] = (s, w)

        return all_weighted_scores, q_vec        

    else:
        raise ValueError(f"Unsupported method: {method}")

def fit_client_gmm(client_scores_dict):
    """
    Returns
        gmm_dict[cid][y] = (π_y^i, μ_y^i, Σ_y^i)
    """
    gmm_dict = {}
    for cid, data in client_scores_dict.items():
        feats  = data['calib_features']   # (n_i, d)
        labels = data['calib_labels']
        total  = feats.shape[0]

        stats  = {}
        for y in np.unique(labels):
            idx      = (labels == y)
            feats_y  = feats[idx]
            pi_y      = feats_y.shape[0] / total
            mu_y      = feats_y.mean(0)
            if feats_y.shape[0] == 1:
                sigma__y = np.eye(feats_y.shape[1]) * 1e-5  # fallback to tiny identity
            else:
                sigma_y = np.cov(feats_y.T) + 1e-5 * np.eye(feats_y.shape[1])
            stats[int(y)] = (pi_y, mu_y, sigma_y)
        gmm_dict[cid] = stats
    return gmm_dict

def fit_client_gmm_diagonal(client_scores_dict, epsilon=1e-5):
    """
    Returns:
        gmm_dict[cid][y] = (π_y^i, μ_y^i, Σ_y^i)
        where Σ_y is diagonal covariance matrix (as full matrix)
    """
    gmm_dict = {}
    for cid, data in client_scores_dict.items():
        feats  = data['calib_features']   # shape: (n_i, d)
        labels = data['calib_labels']
        total  = feats.shape[0]
        d      = feats.shape[1]

        stats = {}
        for y in np.unique(labels):
            idx = (labels == y)
            feats_y = feats[idx]
            pi_y = feats_y.shape[0] / total
            mu_y = feats_y.mean(0)

            if feats_y.shape[0] == 1:
                sigma_diag = np.ones(d) * epsilon
            else:
                sigma_diag = feats_y.var(axis=0) + epsilon  # (d,)

            sigma_y = np.diag(sigma_diag)  # shape: (d, d)
            stats[int(y)] = (pi_y, mu_y, sigma_y)

        gmm_dict[cid] = stats

    return gmm_dict

# def log_mixture_pdf(phi, class_stats):
#     """phi: (d,), class_stats[y] = (π_y, μ_y, Σ_y)"""
#     log_probs = []
#     for pi_y, mu_y, sigma_y in class_stats.values():
#         mvn = D.MultivariateNormal(
#                 torch.as_tensor(mu_y), torch.as_tensor(sigma_y))
#         log_probs.append(np.log(pi_y) + mvn.log_prob(torch.as_tensor(phi)).item())
#     return np.logaddexp.reduce(log_probs)

def log_mixture_pdf(phi_batch, class_stats, batch_size=1000):
    """
    Vectorized + memory-efficient computation of log marginal density under diagonal GMM.

    Args:
        phi_batch: (n, d) ndarray of features
        class_stats: dict[y] = (π_y, μ_y, Σ_y), where Σ_y is diagonal (as full matrix)
        batch_size: number of φ per chunk (default: 1000)

    Returns:
        log_p: (n,) ndarray of log p(x)
    """
    n, d = phi_batch.shape
    K = len(class_stats)

    # Prepare GMM components once
    pi_y_list, mu_y_list, sigma_y_list = [], [], []

    for pi_y, mu_y, sigma_y in class_stats.values():
        pi_y_list.append(pi_y)
        mu_y_list.append(mu_y)
        sigma_y_list.append(np.diag(sigma_y))  # extract diag from (d,d)

    pi_y = np.array(pi_y_list)                  # (K,)
    mu_y = np.stack(mu_y_list, axis=0)          # (K, d)
    var_y = np.stack(sigma_y_list, axis=0)      # (K, d)

    log_coeff = -0.5 * np.sum(np.log(2 * np.pi * var_y), axis=1)  # (K,)

    results = []

    for i in range(0, n, batch_size):
        phi_chunk = phi_batch[i:i + batch_size]                    # (b, d)
        diff = phi_chunk[:, None, :] - mu_y[None, :, :]           # (b, K, d)
        sq_maha = (diff**2 / var_y[None, :, :]).sum(axis=2)       # (b, K)

        log_probs = np.log(pi_y)[None, :] + log_coeff[None, :] - 0.5 * sq_maha  # (b, K)
        log_p_chunk = logsumexp(log_probs, axis=1)  # (b,)
        results.append(log_p_chunk)

    return np.concatenate(results, axis=0)

# def compute_lambda_for_client(cid, feat_mat, gmm_dict, pi_i):
#     """返回该 client 所有样本的 λ 向量 (n_i,)"""
#     log_p_star = np.array([log_mixture_pdf(f, gmm_dict[cid]) for f in feat_mat])
#     # 计算分母 ∑_i π_i P^i_X(x)
#     log_mix_list = []
#     for other_id, gmm in gmm_dict.items():
#         weight = np.log(pi_i[other_id])
#         log_mix_list.append(weight + 
#             np.array([log_mixture_pdf(f, gmm) for f in feat_mat]))
#     # 对不同 client 先做加法后 log-sum-exp
#     log_p_mix = np.logaddexp.reduce(log_mix_list, axis=0)
#     return np.exp(log_p_star - log_p_mix)   # shape (n_i,)

# def compute_lambda_for_client(cid, feat_mat, gmm_dict, pi_i):
#     """
#     Efficient λ(x) for all samples in client `cid` using vectorized GMM evaluation.

#     Args:
#         feat_mat: ndarray (n_i, d), client `cid` calibration features
#         gmm_dict: dict[cid][y] = (π_y, μ_y, Σ_y) for all clients
#         pi_i: dict[cid] = weight of client i

#     Returns:
#         λ(x) for all x in client cid, shape (n_i,)
#     """
#     log_p_star = log_mixture_pdf(feat_mat, gmm_dict[cid])  # log p^i_X(x)

#     log_mix_list = []
#     for other_id, gmm in gmm_dict.items():
#         weight = np.log(pi_i[other_id])
#         log_p = log_mixture_pdf(feat_mat, gmm)  # log p^j_X(x)
#         log_mix_list.append(weight + log_p)           # (n_i,)

#     log_p_mix = logsumexp(log_mix_list, axis=0)  # shape: (n_i,)
#     return np.exp(log_p_star - log_p_mix)        # shape: (n_i,)

def compute_lambda_for_client(cid, feat_mat, gmm_dict, pi_i):
    """
    Efficient and memory-friendly λ(x) calculation for all samples in client `cid`.

    Args:
        feat_mat: ndarray (n_i, d), features of client `cid`
        gmm_dict: dict[cid][y] = (π_y, μ_y, Σ_y)
        pi_i: dict[cid] = client weight

    Returns:
        λ(x): ndarray of shape (n_i,)
    """
    # Numerator: log p^i_X(x)
    log_p_star = log_mixture_pdf(feat_mat, gmm_dict[cid])  # uses batch-safe version

    # Denominator: log sum_j π_j * p^j_X(x) via incremental logaddexp
    log_p_mix = None

    for other_cid, class_stats in gmm_dict.items():
        log_weight = np.log(pi_i[other_cid])
        log_p_j = log_mixture_pdf(feat_mat, class_stats)

        if log_p_mix is None:
            log_p_mix = log_weight + log_p_j
        else:
            log_p_mix = logaddexp(log_p_mix, log_weight + log_p_j)

    return np.exp(log_p_star - log_p_mix)  # shape: (n_i,)

def build_weighted_calib_scores(client_scores_dict, gmm_dict, pi_i):
    scores, weights, labels = [], [], []
    for cid, data in client_scores_dict.items():
        lam   = compute_lambda_for_client(
                    cid, data['calib_features'], gmm_dict, pi_i)
        scores.extend(data['calib'])           # conformal 分数
        weights.extend(lam)
        labels.extend(data['calib_labels'])
    scores   = np.array(scores)
    weights  = np.array(weights)
    labels   = np.array(labels)
    weights /= weights.sum()                  # 归一化
    return scores, weights, labels

def weighted_quantile_cs(values, weights, q):
    """values, weights: 1D np arrays; q∈(0,1)"""
    sort_idx = np.argsort(values)
    vs, ws   = values[sort_idx], weights[sort_idx]
    cdf      = np.cumsum(ws)
    return vs[np.searchsorted(cdf, q)]

def weighted_scores_3(client_scores_dict, args):
    """
    Compute importance-weighted conformity scores.
    """
    label_counts, client_totals = estimate_client_label_counts(client_scores_dict, args.num_class)
    global_counts, global_total = compute_global_label_counts(label_counts)

    pi_i = {cid: client_totals[cid] / global_total for cid in client_totals}

    gmm_dict = fit_client_gmm_diagonal(client_scores_dict)

    scores, weights, labels = build_weighted_calib_scores(client_scores_dict, gmm_dict, pi_i)

    qhat = weighted_quantile_cs(scores, weights, 1 - args.alpha)

    return qhat       

def weighted_scores_4(client_scores_dict, global_sketch_scores, args):
    """
    Compute importance-weighted conformity scores.
    """
    key = 'calib'

    likelihood_ratios = estimate_empirical_likelihood_ratios_2(client_scores_dict, global_sketch_scores, args.t, min_cdf=1e-6)

        # normalized_weights = normalize_client_weights_2(likelihood_ratios)

    weighted_scores_dict = {}
    for cid in client_scores_dict:
        scores = client_scores_dict[cid][key]         # numpy array
            # weights = normalized_weights[cid]             # numpy array
        weights = likelihood_ratios[cid]             # numpy array
        weighted = weights * scores
            # weighted = 1.0 * scores
        weighted_scores_dict[cid] = weighted

        if np.all(weighted == 0):
            print(f"[Client {cid}] Warning: all weighted scores are zero.")

    return weighted_scores_dict

def compute_Coverages_Sizes(preds, yTest):
    # Split conformal coverage and size (same threshold for all)
    
    coveragesSplit = np.array([yTest[i] in preds[i] for i in range(len(yTest))], dtype=float)
    setSizesSplit = np.array([len(x) for x in preds])

    return (coveragesSplit, setSizesSplit)

def compute_federated_global_quantile(client_scores_dict, args, return_loss_curve=False):
    """
    Estimate the global quantile (e.g., 90th percentile) over all clients'
    calibration conformity scores, without aggregating raw data.
    
    Uses a pooled quantile loss objective over all calibration scores.
    """
    all_scores = []
    for cid, scores in client_scores_dict.items():
        v = torch.tensor(scores, dtype=torch.float32)
        all_scores.append(v)

    # Concatenate all local scores (as if logically central, still memory-efficient)
    v_all = torch.cat(all_scores, dim=0).detach()

    # Initialize scalar quantile estimate
    qhat = torch.tensor(0.9, requires_grad=True)

    optimizer = torch.optim.Adam([qhat], lr=args.lr)
    loss_curve = []

    for i in range(args.num_epochs):  
        optimizer.zero_grad()

        # diff = v_all - qhat
        # loss = torch.mean((args.alpha - (diff < 0).float()) * diff)
        loss = torch.mean(torch.where(v_all >= qhat, (1 - args.alpha) * (v_all - qhat), args.alpha * (qhat - v_all)))

        loss.backward()
        optimizer.step()

        loss_curve.append(loss.item())

        if i % 50 == 0:
            print(f"Epoch {i:03d} | Quantile: {qhat.item():.6f}| Quantile Loss: {loss.item():.6f}")


    # q_test = get_conformal_quantile(v_all, args.alpha)

    # print(q_test)

    return qhat.item(), loss_curve


def compute_federated_global_quantile_2(client_scores_dict, weight_dict, args, return_loss_curve=False):
    """
    Estimate the global quantile (e.g., 90th percentile) over all clients'
    calibration conformity scores, without aggregating raw data.
    
    Uses a pooled quantile loss objective over all calibration scores.
    """
    all_scores = []
    all_weights = []
    for cid in client_scores_dict:
        v = torch.tensor(client_scores_dict[cid], dtype=torch.float32)
        w = torch.tensor(weight_dict[cid], dtype=torch.float32)
        all_scores.append(v)
        all_weights.append(w)

    # Concatenate all local scores (as if logically central, still memory-efficient)
    V = torch.cat(all_scores, dim=0)
    W = torch.cat(all_weights, dim=0)

    # Initialize scalar quantile estimate
    qhat = torch.tensor(0.9, requires_grad=True)

    optimizer = torch.optim.Adam([qhat], lr=args.lr)
    loss_curve = []

    for i in range(args.num_epochs):  
        optimizer.zero_grad()

        # diff = v_all - qhat
        # loss = torch.mean((args.alpha - (diff < 0).float()) * diff)
        loss = loss = torch.mean(torch.where(V >= qhat, (1 - args.alpha) * (V - qhat), args.alpha * (qhat - V)) * W)

        loss.backward()
        optimizer.step()

        loss_curve.append(loss.item())

        if i % 50 == 0:
            print(f"Epoch {i:03d} | Quantile: {qhat.item():.6f}| Quantile Loss: {loss.item():.6f}")

    return qhat.item(), loss_curve

# def global_conformal(val_scores_all, val_labels, qhat):
#     '''
#     Use cal_scores_all and cal_labels to compute 1-alpha conformal quantiles for standard conformal.
#     If exact_coverage is True, apply randomized to achieve exact 1-alpha coverage. Otherwise, use
#     unrandomized conservative sets. 
#     Create predictions and compute evaluation metrics on val_scores_all and val_labels.
#     '''
#     # num = len(cal_scores_all)/num_classes
#     # mc_alpha = alpha - (marignal_gap/np.sqrt(num))

#     # print(f'Marginal Quantile:{standard_qhat}')
#     standard_preds = create_prediction_sets(val_scores_all, qhat)

#     (coverages, setSizes) = compute_Coverages_Sizes(standard_preds, val_labels)

#     marginal_cov = compute_coverage(val_labels, standard_preds)
#     curr_set_sizes = [len(x) for x in standard_preds]

#     print('Marginal Coverage:', marginal_cov)

#     print('Average Set Size:', np.mean(curr_set_sizes))

#     return (coverages, setSizes)


def global_conformal(val_scores_all, val_labels, qhat, client_scores=None, save_path=None):
    '''
    Compute marginal and per-client conformal metrics, optionally save to CSV.
    '''
    standard_preds = create_prediction_sets(val_scores_all, qhat)

    (coverages, setSizes) = compute_Coverages_Sizes(standard_preds, val_labels)

    marginal_cov = np.mean(coverages)
    avg_set_size = np.mean(setSizes)

    print('Marginal Coverage:', marginal_cov)
    print('Average Set Size:', avg_set_size)

    # Store results in a DataFrame
    result_rows = []

    result_rows.append({
        'client_id': 'global',
        'coverage': marginal_cov,
        'avg_set_size': avg_set_size
    })

    if client_scores is not None:
        # Compute per-client stats
        start = 0
        for cid in sorted(client_scores.keys()):
            num_samples = len(client_scores[cid]['test_labels'])
            end = start + num_samples

            client_coverages = coverages[start:end]
            client_sizes = setSizes[start:end]

            result_rows.append({
                'client_id': cid,
                'coverage': np.mean(client_coverages),
                'avg_set_size': np.mean(client_sizes)
            })

            start = end

    if save_path:
        df = pd.DataFrame(result_rows)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Saved results to {save_path}")

    return (coverages, setSizes)
    
# def compress_scores(scores, sketch_method='histogram', bins=50):
#     """
#     Compress calibration scores using histogram or GK sketch (TDigest).
#     """
#     if sketch_method == 'histogram':
#         hist, bin_edges = np.histogram(scores, bins=bins, range=(0, 1), density=False)
#         return {'type': 'histogram', 'hist': hist, 'bin_edges': bin_edges}
    
#     elif sketch_method == 'tdigest':  # GK-like using t-digest
#         td = TDigest()
#         for s in scores:
#             td.update(s)
#         return {'type': 'tdigest', 'sketch': td}

#     elif sketch_method == 'ddsketch':
#         dd = DDSketch()
#         for s in scores:
#             dd.add(s)
#         return {'type': 'ddsketch', 'sketch': dd}
    
#     else:
#         raise NotImplementedError(f"Sketch method {sketch_method} not supported yet.")

# def add_dp_noise(sketch, sigma):
#     """
#     Add Gaussian noise to sketch bins or t-digest centroids.
#     """
#     if sketch['type'] == 'histogram':
#         noisy_hist = sketch['hist'] + np.random.normal(0, sigma, size=sketch['hist'].shape)
#         noisy_hist = np.maximum(noisy_hist, 0)  # Clip negatives
#         return {'type': 'histogram', 'hist': noisy_hist, 'bin_edges': sketch['bin_edges']}
    
#     elif sketch['type'] == 'tdigest':

#         noisy_td = TDigest()
#         centroids = sketch['sketch'].centroids_to_list()

#         # print("[Debug] TDigest.centroids_to_list() item example:", centroids[:3])
#         # print("[Debug] centroids type:", type(centroids[0]))
#         # print("[Debug] centroids example:", centroids[:3])

#         for item in centroids:
#             try:
#                 if isinstance(item, dict):
#                     m = float(item.get('mean', item.get('m', 0)))  
#                     w = float(item.get('count', item.get('w', 0)))
#                 elif isinstance(item, tuple):
#                     m, w = item
#                 elif isinstance(item, str):
#                     parts = item.split(',')
#                     m = float(parts[0].split('=')[1].strip())
#                     w = float(parts[1].split('=')[1].strip())
#                 else:
#                     raise TypeError(f"Unrecognized centroid format: {type(item)} | {item}")
#             except Exception as e:
#                 raise ValueError(f"Failed to parse TDigest centroid item: {item}") from e

#         noisy_weight = max(w + np.random.normal(0, sigma), 0)
#         noisy_td.update(m, noisy_weight)

#         return {'type': 'tdigest', 'sketch': noisy_td}

#     elif sketch['type'] == 'ddsketch':
#         noisy_dd = deepcopy(sketch['sketch'])
#         # Access underlying bin counts (positive, negative, zero)
#         noisy_dd = DDSketch()
#         for q in np.linspace(0, 1, 200):
#             try:
#                 v = original.get_quantile_value(q)
#                 weight = max(1 + np.random.normal(0, sigma), 0.1)
#                 for _ in range(int(weight)):
#                     noisy_dd.add(v)
#             except Exception:
#                 continue
#         return {'type': 'ddsketch', 'sketch': noisy_dd}

#     else:
#         raise ValueError("Unknown sketch type")


# def merge_tdigests(centroid_dicts):
#     """
#     Merge a list of TDigest sketches by re-inserting all centroids.
#     Assumes centroids are in format {'m': ..., 'c': ...}
#     """
#     merged = TDigest()
#     for td in centroid_dicts:
#         for item in td.centroids_to_list():
#             m = float(item['m'])
#             c = float(item['c'])
#             merged.update(m, c)
#     return merged

# def aggregate_sketches(sketches, sketch_method='histogram'):
#     """
#     Aggregate client sketches at server.
#     """
#     if sketch_method == 'histogram':
#         total_hist = None
#         bin_edges = sketches[0]['bin_edges']
#         for s in sketches:
#             if total_hist is None:
#                 total_hist = np.array(s['hist'])
#             else:
#                 total_hist += np.array(s['hist'])
#         return {'type': 'histogram', 'hist': total_hist, 'bin_edges': bin_edges}
    
#     elif sketch_method == 'tdigest':
#         td_global = TDigest()
#         for s in sketches:
#             td_global = merge_tdigests([s['sketch'] for s in sketches])
#         return {'type': 'tdigest', 'sketch': td_global}
    
#     elif sketch_method == 'ddsketch':
#         dd_agg = DDSketch()
#         for s in sketches:
#             dd_agg.merge(s['sketch'])
#         return {'type': 'ddsketch', 'sketch': dd_agg}

#     else:
#         raise ValueError("Unsupported sketch type for aggregation")

# def sample_from_aggregated(sketch, num_samples=1000):
#     """
#     Recover global calibration scores (approximate) from aggregated sketch.
#     """
#     if sketch['type'] == 'histogram':
#         hist = sketch['hist']
#         bin_edges = sketch['bin_edges']
#         p = hist / np.sum(hist)
#         samples = np.random.choice((bin_edges[:-1] + bin_edges[1:]) / 2, size=num_samples, p=p)
#         return samples
    
#     elif sketch['type'] == 'tdigest':
#         return np.array([sketch['sketch'].percentile(q) / 100 for q in np.linspace(0, 100, num_samples)])
    
#     elif sketch['type'] == 'ddsketch':
#         # Percentile from 0 to 100
#         return np.array([sketch['sketch'].get_quantile_value(q/100) for q in np.linspace(0, 100, num_samples)])

#     else:
#         raise ValueError("Unknown sketch type")


# def federated_sketching(client_scores, sketch_method='histogram', sigma=0.1, bins=50, global_samples=1000):
#     """
#     Perform sketching + noise + aggregation over client_scores[cid]['calib'] only.
#     Return global_calib_scores ∈ ℝ^num_samples.
#     """
#     client_sketches = []

#     for cid in client_scores:
#         local_scores = np.array(client_scores[cid]['calib'])  # only use 'calib' field
#         local_sketch = compress_scores(local_scores, sketch_method=sketch_method, bins=bins)
#         noisy_sketch = add_dp_noise(local_sketch, sigma=sigma)
#         client_sketches.append(noisy_sketch)

#     aggregated_sketch = aggregate_sketches(client_sketches, sketch_method=sketch_method)
#     global_calib_scores = sample_from_aggregated(aggregated_sketch, num_samples=global_samples)

#     return global_calib_scores

def compress_scores(scores):
    td = TDigest()
    for s in scores:
        td.update(s)
    return {'type': 'tdigest', 'sketch': td}

def merge_tdigests(td_list):
    merged = TDigest()
    for td in td_list:
        for item in td.centroids_to_list():
            m = float(item['m'])
            c = float(item['c'])
            merged.update(m, c)
    return merged

def aggregate_sketches(sketches):
    td_list = [s['sketch'] for s in sketches]
    return {'type': 'tdigest', 'sketch': merge_tdigests(td_list)}

def construct_global_cdf_from_tdigest(sketch):
    centroids = sketch['sketch'].centroids_to_list()
    values = [float(c['m']) for c in centroids]
    weights = [float(c['c']) for c in centroids]
    values = np.array(values)
    weights = np.array(weights)
    idx = np.argsort(values)
    values = values[idx]
    weights = weights[idx]
    cum_weights = np.cumsum(weights)
    cum_weights /= cum_weights[-1]
    return interp1d(values, cum_weights, kind='linear', bounds_error=False, fill_value=(0.0, 1.0))

def federated_sketching(client_scores_dict):
    sketches = []
    for cid in client_scores_dict:
        local_scores = np.array(client_scores_dict[cid]['calib'])
        sketch = compress_scores(local_scores)
        sketches.append(sketch)
    return aggregate_sketches(sketches)

def estimate_empirical_likelihood_ratios_3(client_scores_dict, global_cdf_fn, t, min_cdf=1e-6):
    client_weights = {}
    for cid, data in client_scores_dict.items():
        local_scores = np.array(data['calib'])
        sorted_local = np.sort(local_scores)
        N_local = len(local_scores)
        F_local = np.searchsorted(sorted_local, local_scores, side='right') / N_local
        F_global = global_cdf_fn(local_scores)

        F_local_clipped = np.clip(F_local, min_cdf, None)
        omega = np.exp(t * np.log(F_global / F_local_clipped))
        # omega = F_global / F_local_clipped
        client_weights[cid] = omega
    return client_weights

def weighted_scores_accurate(client_scores_dict, args):
    key = 'calib'
    aggregated_sketch = federated_sketching(client_scores_dict)
    global_cdf_fn = construct_global_cdf_from_tdigest(aggregated_sketch)
    likelihood_ratios = estimate_empirical_likelihood_ratios_3(client_scores_dict, global_cdf_fn, args.t)

    all_weights = np.concatenate([likelihood_ratios[cid] for cid in client_scores_dict])
    total_weight = np.sum(all_weights)

    weighted_scores_dict = {}
    weights_dict = {}

    for cid in client_scores_dict:
        scores = np.array(client_scores_dict[cid][key])
        weights = likelihood_ratios[cid]
        # weights /= total_weight
        weighted = weights * scores
        weighted_scores_dict[cid] = weighted
        weights_dict[cid] = weights

        if np.all(weighted == 0):
            print(f"[Client {cid}] Warning: all weighted scores are zero.")

    return weighted_scores_dict, weights_dict 

def compute_federated_per_class_quantiles(
    weighted_scores_dict,
    p_star_dicts,
    client_scores,
    num_classes,
    alpha
):
    """
    Compute per-class quantiles for federated conformal prediction.

    Args:
        weighted_scores_dict: dict[client_id] -> array of weighted conformity scores
        p_star_dicts: dict[client_id] -> dict[class_id] -> p*_y
        num_classes: total number of classes
        alpha: miscoverage level (1 - target coverage)

    Returns:
        qhat_dict: dict[class_id] -> q̂_y
    """
    from collections import defaultdict

    # For each class, collect all weighted scores and weights
    class_scores = defaultdict(list)
    class_weights = defaultdict(list)

    for cid in weighted_scores_dict:
        scores = weighted_scores_dict[cid]
        labels = client_scores[cid]['calib_labels']
        p_star_dict = p_star_dicts[cid]

        for i in range(len(scores)):
            y = labels[i]
            s = scores[i]
            p_star_y = p_star_dict[y]
            w = 1 - p_star_y  # conformal weighting
            class_scores[y].append(s)
            class_weights[y].append(w)

        # Add delta-mass at value=1 for each class
        for y in range(num_classes):
            if y in p_star_dict:
                class_scores[y].append(1.0)
                class_weights[y].append(p_star_dict[y])

    # Compute per-class weighted quantile
    qhat_dict = {}
    for y in range(num_classes):
        scores_y = np.array(class_scores[y])
        weights_y = np.array(class_weights[y])
        if len(scores_y) == 0:
            qhat_dict[y] = 1.0  # fallback: always predict
        else:
            qhat_dict[y] = compute_weighted_quantile(scores_y, weights_y, quantile=1 - alpha)

    return qhat_dict


def compute_weighted_quantile(values, weights, quantile):
    """
    Compute the weighted quantile of values with given weights.
    """
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    cumulative_weights = np.cumsum(weights)
    cutoff = quantile * np.sum(weights)
    return values[np.searchsorted(cumulative_weights, cutoff)]


# def create_prediction_sets_per_class(scores_all, qhat_dict):
#     """
#     Create prediction sets using per-class quantiles.

#     Args:
#         scores_all: [n, C] numpy array of test scores (non-conformity)
#         qhat_dict: dict[class_id] -> q̂_y

#     Returns:
#         list of prediction sets (list of np.arrays)
#     """
#     n, C = scores_all.shape
#     pred_sets = []
#     for i in range(n):
#         pred = []
#         for y in range(C):
#             if scores_all[i, y] <= qhat_dict[y]:
#                 pred.append(y)
#         pred_sets.append(np.array(pred))
#     return pred_sets

def create_prediction_sets_per_class(scores_all, q_hat):
    scores_all = np.array(scores_all)
    set_preds = []
    num_samples = len(scores_all)
    for i in range(num_samples):
        set_preds.append(np.where(scores_all[i,:] <= q_hat)[0])
        
    return set_preds

def global_conformal_per_class(val_scores_all, val_labels, qhat, client_scores=None, save_path=None):
    '''
    Compute marginal and per-client conformal metrics, optionally save to CSV.
    '''
    # standard_preds = create_prediction_sets(val_scores_all, qhat)
    standard_preds = create_prediction_sets_per_class(val_scores_all, qhat)

    (coverages, setSizes) = compute_Coverages_Sizes(standard_preds, val_labels)

    marginal_cov = np.mean(coverages)
    avg_set_size = np.mean(setSizes)

    print('Marginal Coverage:', marginal_cov)
    print('Average Set Size:', avg_set_size)

    # Store results in a DataFrame
    result_rows = []

    result_rows.append({
        'client_id': 'global',
        'coverage': marginal_cov,
        'avg_set_size': avg_set_size
    })

    if client_scores is not None:
        # Compute per-client stats
        start = 0
        for cid in sorted(client_scores.keys()):
            num_samples = len(client_scores[cid]['test_labels'])
            end = start + num_samples

            client_coverages = coverages[start:end]
            client_sizes = setSizes[start:end]

            result_rows.append({
                'client_id': cid,
                'coverage': np.mean(client_coverages),
                'avg_set_size': np.mean(client_sizes)
            })

            start = end

    if save_path:
        df = pd.DataFrame(result_rows)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Saved results to {save_path}")

    return (coverages, setSizes)

def distributed_quantile_from_scores(
    client_scores,
    args,
    sketch_method='tdigest'
):
    """
    Use distributed sketching over raw calibration scores to estimate q_hat.
    Implements the (N + K)(1 - alpha) / N quantile rule.
    """

    if sketch_method == 'tdigest':
        global_sketch = TDigest()
    elif sketch_method == 'ddsketch':
        global_sketch = DDSketch()
    else:
        raise ValueError("Unsupported sketch method")

    N = 0
    K = args.num_clients

    for cid in client_scores:
        local_scores = np.asarray(client_scores[cid]['calib'])  
        N += len(local_scores)

        if sketch_method == 'tdigest':
            local_sketch = TDigest()
            local_sketch.batch_update(local_scores)  
            global_sketch = global_sketch + local_sketch

        elif sketch_method == 'ddsketch':
            local_sketch = DDSketch()
            for s in local_scores:
                local_sketch.add(s)
            global_sketch.merge(local_sketch)

    # Compute quantile level
    t = np.ceil((N + K) * (1 - args.alpha)) / N

    if sketch_method == 'tdigest':
        q_hat = global_sketch.percentile(t * 100)
    elif sketch_method == 'ddsketch':
        q_hat = global_sketch.get_quantile_value(t)

    return q_hat

def visualize_client_joint_distribution(metadata, client_partition):
    domain_list = sorted(metadata['domain'].unique())
    label_list = sorted(metadata['label'].unique())

    for cid, indices in client_partition.items():
        subset = metadata.iloc[indices]
        pivot = pd.crosstab(subset['domain'], subset['label'], normalize='index')
        pivot = pivot.reindex(domain_list).fillna(0)

        plt.figure(figsize=(6,4))
        sns.heatmap(pivot, annot=True, cmap='viridis', cbar=True)
        plt.title(f"Client {cid} Joint (domain vs label)")
        plt.xlabel('Label')
        plt.ylabel('Domain')
        plt.tight_layout()
        plt.show()


