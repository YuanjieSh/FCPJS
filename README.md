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