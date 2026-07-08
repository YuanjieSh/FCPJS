## Valid and Efficient Uncertainty Quantification for Federated Joint Shift (FCPJS)

This repository contains a implementation of **FCPJS**
corresponding to the follow paper:

Yuanjie Shi, Peihong Li, Xuanyu Cao, Yan Yan.
*[Valid and Efficient Uncertainty Quantification for Federated Joint Shift](
https://openreview.net/forum?id=PjKpx6VCov)*.
UAI, 2026 .

## Overview

Reliable uncertainty quantification (UQ) is critical for safety-sensitive federated learning (FL) applications, such as cross-hospital diagnosis and global sensor networks.
In FL, privacy constraints prevent centralized data pooling, while client data exhibit joint distribution shifts across covariates, labels, and conditionals.
These shifts violate the exchangeability assumption required by conformal prediction (CP), which otherwise guarantees distribution-free coverage under i.i.d. data.
To address this gap, we propose Federated Conformal Prediction for Joint Shift (\newCP), enabling valid CP under heterogeneous clients without sharing raw data.
Specifically, the method operates in two stages:
(i) the server constructs a privacy-preserving global sketch of nonconformity score distributions;
(ii) each client performs importance-weighted calibration by contrasting its local scores with the global sketch.
This distribution-ratio weighting corrects joint shifts in a unified manner.
We prove that FCPJS attains valid marginal coverage across clients, up to an $O(\sqrt{\varepsilon_m}+1/\sqrt{n})$ error from the finite number of calibration samples $n$ and sketch size $m$.
Experiments on four heterogeneous benchmarks show that \newCP\ preserves coverage and improves predictive efficiency by $12.64\%$ over the strongest baseline.
To our knowledge, \newCP\ is the first method providing provably valid and efficient conformal UQ for FL under joint distribution shift.


## Running instructions

Please run the commands mentioned below to produce results:

## Training commands
1. Download the model rxrx1_seed_0_epoch_best_model.pth for RxRx1, fmow_seed_0_epoch_best_model.pth for FMoW, best_model.pth for iWildCam from Wilds.

2. Run following commands to get softmax scores, labels and indices.

**RxRx1**
```
Python dataset/rxrx1_2.py 
```
**FMoW**
```
Python dataset/FMoW.py 
```
**iWildCam**
```
Python dataset/iWild.py 
```
  
## Calibration commands
**FCPJS**
# For IID
```
python main_5.py --dataset rxrx1 --use_iid yes --num_clients 50 --splits 10 --method FCP_full --score HPS
python main_5.py --dataset rxrx1 --use_iid yes --num_clients 50 --splits 10 --method FCP_full --score APS
python main_5.py --dataset rxrx1 --use_iid yes --num_clients 50 --splits 10 --method FCP_full --score RAPS
```
# For non-IID
```
python main_5.py --dataset rxrx1 --use_iid no --num_clients 50 --splits 10 --method FCP_full --score HPS
python main_5.py --dataset rxrx1 --use_iid no --num_clients 50 --splits 10 --method FCP_full --score APS
python main_5.py --dataset rxrx1 --use_iid no --num_clients 50 --splits 10 --method FCP_full --score RAPS
```
**FCP**
# For IID
```
python main_5.py --dataset rxrx1 --use_iid yes --num_clients 50 --splits 10 --method FCP --score HPS
```
# For non-IID
```
python main_5.py --dataset rxrx1 --use_iid no --num_clients 50 --splits 10 --method FCP --rho 1.0 --score HPS
```
**FCPLS**
# For IID
```
python main_5.py --dataset rxrx1 --use_iid yes --num_clients 50 --splits 10 --method FCP_LS --sigma 0.1 --score HPS
```
# For non-IID
```
python main_5.py --dataset rxrx1 --use_iid no --num_clients 50 --splits 10 --method FCP_LS --sigma 0.1 --score HPS
```
**FCPCS**
# For IID
```
python main_4.py --dataset rxrx1 --use_iid yes --num_clients 50 --splits 3
```