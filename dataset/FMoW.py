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

from random import sample
import random

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.models as models

from conditionalconformal.synthetic_data import generate_cqr_data
from conditionalconformal.condconf import setup_cvx_problem_calib
from temperatureScaling import torch_ts

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

def initializeFMoWTransform():
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()

def classConfScore(probs,y):
    return sum(probs[probs > probs[int(y)]])

def computeSoftmaxScores(XTrain,XTest,yTrain,yTest):
    T = torch_ts(XTrain,yTrain)
    scaleFactor = T.detach().numpy()
    
    normXtrain = np.apply_along_axis(softmax,1,XTrain/scaleFactor)
    normXtest = np.apply_along_axis(softmax,1,XTest/scaleFactor)

    return normXtrain, normXtest

def save_dataset(softmax_scores, labels, indices, dataset, metadata_array=None, metadata_fields=None, save_dir='.'):
    os.makedirs(save_dir, exist_ok=True)
    npz_path = os.path.join(save_dir, f'{dataset}.npz')
    np.savez(npz_path, softmax=softmax_scores, labels=labels, indices=indices)
    print(f"Saved softmax + labels{' + indices' if indices is not None else ''} to {npz_path}")

    if metadata_array is not None and metadata_fields is not None:
        assert metadata_array.shape[1] == len(metadata_fields), "metadata shape mismatch"
        df = pd.DataFrame(metadata_array, columns=metadata_fields)
        csv_path = os.path.join(save_dir, f'{dataset}_metadata.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved metadata to {csv_path}")

### Load in pretrained neural network
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

results_path = './results'
os.makedirs(results_path, exist_ok=True)

dimOut = 62  # FMoW has 62 classes
constructor = models.densenet121
model = constructor(pretrained=True)
dimFeatures = model.classifier.in_features
lastLayer = nn.Linear(dimFeatures, dimOut)
model.classifier = lastLayer
model.d_out = dimOut


fmowModel = model.to(device)
myLoad(fmowModel,
            'data/fmow_seed_0_epoch_best_model.pth',
            device=device)
featurizer = fmowModel
classifier = fmowModel.classifier
featurizer.classifier = nn.Identity()

fmowModel = nn.Sequential(*(featurizer,classifier))

fmowModel.eval()
featurizer.eval()
classifier.eval()

print('Finishing loading models')

### Normalize images for neural net evaluation. Code taken from WILDS repository

myTransform = initializeFMoWTransform()

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

### Load in data from WILDS repository
fmowData = get_dataset(dataset="fmow", download=False)
# fmowTestImages = fmowData.get_subset(
#     "test",
#     transform = myTransform
# )
# metaData = pd.read_csv('data/rxrx1_v1.0/metadata.csv')
# metaData = metaData[metaData['dataset'] == 'test']
fmowTest = fmowData.get_subset("test", transform=myTransform)
metadata_fields = fmowData.metadata_fields

print('Finishing loading dataset')

n = len(fmowTest)
indices = np.arange(n)
y = np.zeros(n)
metaDataFinal = np.zeros((n, len(fmowTest[0][2])))
featureMat = np.zeros((n, dimFeatures))
rawProbMat = np.zeros((n, dimOut))

print('Extracting features and predictions...')
for i in range(n):
    image, label, metadata = fmowTest[i]
    y[i] = label.item()
    metaDataFinal[i, :] = metadata.numpy()
    with torch.no_grad():
        feature = featurizer(image.unsqueeze(0).to(device)).cpu().numpy()[0]
        featureMat[i, :] = feature
        logits = classifier(torch.tensor(feature).unsqueeze(0).to(torch.float32).to(device)).cpu().numpy()[0]
        rawProbMat[i, :] = logits

# Normalize using softmax
normX = np.apply_along_axis(softmax, 1, rawProbMat)

# Save
save_dataset(
    softmax_scores=normX,
    labels=y,
    indices=indices, 
    dataset='fmow',
    metadata_array=metaDataFinal,
    metadata_fields=metadata_fields
)
