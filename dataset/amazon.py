import torch
import torch.nn as nn
import torchvision 
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
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
from transformers import BertTokenizer, BertModel
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

from random import sample
import random

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader

from conditionalconformal.synthetic_data import generate_cqr_data
from conditionalconformal.condconf import setup_cvx_problem_calib
from temperatureScaling import torch_ts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results_path = "./results"
os.makedirs(results_path, exist_ok=True)

### === Step 1: Load WILDS Amazon Reviews Test Set === ###
dataset = get_dataset("amazon", download=False)
test_set = dataset.get_subset("test")
metadata_fields = dataset.metadata_fields

texts = [test_set[i][0] for i in range(len(test_set))]
labels = np.array([test_set[i][1] for i in range(len(test_set))])
metadata = np.vstack([test_set[i][2].numpy() for i in range(len(test_set))])
indices = np.arange(len(test_set))

print("Loaded Amazon test set")

### === Step 2: Load DistilBERT tokenizer and featurizer === ###
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Load model directly
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)
model.to(device)

# Load WILDS-trained checkpoint
ckpt = torch.load("data/amazon_seed_0_epoch_best_model.pth", map_location=device)

# If checkpoint has 'algorithm' key, strip it
state = ckpt['algorithm'] if 'algorithm' in ckpt else ckpt
model.load_state_dict(state, strict=False)  # strict=False allows for missing irrelevant keys
model.eval()

print("Loaded pretrained model.")

### === Step 5: Extract softmax outputs === ###
from torch.nn.functional import softmax

batch_size = 64
softmax_scores = []

with torch.no_grad():
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
        output = model(**encoded)
        probs = softmax(output.logits, dim=1).cpu().numpy()
        softmax_scores.append(probs)

softmax_scores = np.vstack(softmax_scores)

### === Step 6: Save results === ###
def save_dataset(softmax_scores, labels, indices, dataset, metadata_array=None, metadata_fields=None, save_dir='.'):
    os.makedirs(save_dir, exist_ok=True)
    npz_path = os.path.join(save_dir, f'{dataset}.npz')
    np.savez(npz_path, softmax=softmax_scores, labels=labels, indices=indices)
    print(f"Saved softmax + labels + indices to {npz_path}")

    if metadata_array is not None and metadata_fields is not None:
        df = pd.DataFrame(metadata_array, columns=metadata_fields)
        csv_path = os.path.join(save_dir, f'{dataset}_metadata.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved metadata to {csv_path}")

save_dataset(
    softmax_scores=softmax_scores,
    labels=labels,
    indices=indices,
    dataset="amazon",
    metadata_array=metadata,
    metadata_fields=metadata_fields,
    save_dir=results_path
)


