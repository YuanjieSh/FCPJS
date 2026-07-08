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

def initializeiWildTransform():
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
dimOut = 182
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


iwildModel = model.to(device)
myLoad(iwildModel,
            'data/best_model.pth',
            device=device)
featurizer = iwildModel
classifier = iwildModel.fc
featurizer.fc = nn.Identity()

iwildModel = nn.Sequential(*(featurizer,classifier))

iwildModel.eval()
featurizer.eval()
classifier.eval()

print('Finishing loading models')

### Normalize images for neural net evaluation. Code taken from WILDS repository

myTransform = initializeiWildTransform()

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

### Load in data from WILDS repository
iwildData = get_dataset(dataset="iwildcam", download=False)
iwildTestImages = iwildData.get_subset(
    "test",
    transform = myTransform
)
metaData = pd.read_csv('data/iwildcam_v2.0/metadata.csv')
metaData = metaData[metaData['split'] == 'test']

print('Finishing loading dataset')

n = len(iwildTestImages) 
indices = np.arange(n)

### Get feature representation from pretrained neural network
featureMat = np.zeros((n,2048))
for i in range(n):
    featureMat[i,:] = featurizer(iwildTestImages[i][0].unsqueeze(0).to(device)).cpu().detach().numpy()[0,:]

### reshape meta data
metaDataFinal = np.zeros((n,len(iwildTestImages[0][2])))

### extract y values
y = np.zeros(n)
for i in range(n):
    metaDataFinal[i,:] = iwildTestImages[i][2].numpy()
    y[i] = iwildTestImages[i][1].numpy()
    
### get probabilities output by pretrained neural network
rawProbMat = np.zeros((n, dimOut))
for i in range(n):
    rawProbMat[i,:] = classifier(torch.from_numpy(featureMat[i,:].reshape(1,2048)).to(torch.float32).to(device)).cpu().detach().numpy()[0,:]


T = torch_ts(rawProbMat,y)
scaleFactor = T.detach().numpy()
normX = np.apply_along_axis(softmax,1,rawProbMat/scaleFactor)

meta_fields = iwildData.metadata_fields

save_dataset(
    softmax_scores=normX,
    labels=y,
    indices=indices, 
    dataset='iwild',
    metadata_array=metaDataFinal,     # shape (n, D)
    metadata_fields=meta_fields       # list of strings
)
