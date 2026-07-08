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

def initializeRxrx1Transform(): 
    def standardize(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2))
        std = x.std(dim=(1, 2))
        std[std == 0.] = 1.
        return TF.normalize(x, mean, std)
    t_standardize = transforms.Lambda(lambda x: standardize(x))

    transforms_ls = [
        transforms.ToTensor(),
        t_standardize,
    ]
    return transforms.Compose(transforms_ls)

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
dimOut = 1139
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# print(device)

results_path = './results'
os.makedirs(results_path, exist_ok=True)

constructor = torchvision.models.resnet50
model = constructor()
dimFeatures = model.fc.in_features
lastLayer = nn.Linear(dimFeatures,dimOut)
model.d_out = dimFeatures
model.fc = lastLayer


rx1Model = model.to(device)
myLoad(rx1Model,
            'data/rxrx1_seed_0_epoch_best_model.pth',
            device=device)
featurizer = rx1Model
classifier = rx1Model.fc
featurizer.fc = nn.Identity()

rx1Model = nn.Sequential(*(featurizer,classifier))

rx1Model.eval()
featurizer.eval()
classifier.eval()

print('Finishing loading models')

### Normalize images for neural net evaluation. Code taken from WILDS repository

myTransform = initializeRxrx1Transform()

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

### Load in data from WILDS repository
rx1Data = get_dataset(dataset="rxrx1", download=False)
rx1TestImages = rx1Data.get_subset(
    "test",
    transform = myTransform
)
metaData = pd.read_csv('data/rxrx1_v1.0/metadata.csv')
metaData = metaData[metaData['dataset'] == 'test']

print('Finishing loading dataset')

n = len(rx1TestImages) 
indices = np.arange(n)

### Get feature representation from pretrained neural network
featureMat = np.zeros((n,2048))
for i in range(n):
    featureMat[i,:] = featurizer(rx1TestImages[i][0].reshape((1,3,256,256)).to(device)).cpu().detach().numpy()[0,:]

### reshape meta data
metaDataFinal = np.zeros((n,len(rx1TestImages[0][2])))

### extract y values
y = np.zeros(n)
for i in range(n):
    metaDataFinal[i,:] = rx1TestImages[i][2].numpy()
    y[i] = rx1TestImages[i][1].numpy()
    
### get probabilities output by pretrained neural network
rawProbMat = np.zeros((n,1139))
for i in range(n):
    rawProbMat[i,:] = classifier(torch.from_numpy(featureMat[i,:].reshape(1,2048)).to(torch.float32).to(device)).cpu().detach().numpy()[0,:]


T = torch_ts(rawProbMat,y)
scaleFactor = T.detach().numpy()
normX = np.apply_along_axis(softmax,1,rawProbMat/scaleFactor)

meta_fields = rx1Data.metadata_fields

save_dataset(
    softmax_scores=normX,
    labels=y,
    indices=indices, 
    dataset='RXRX1',
    metadata_array=metaDataFinal,     # shape (n, D)
    metadata_fields=meta_fields       # list of strings
)
