import torch
import argparse

import collections
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torchvision
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# from temperature import tune_temp
# from utils import (Net, Net_eNTK, average_models, client_update, compute_eNTK,
#                    evaluate_many_models, evaluate_model, get_datasets,
#                    make_model, partition_dataset, replace_last_layer,
#                    scaffold_update)
import torchvision 
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

import pickle
# import pandas as pd 
# import numpy as np
# import cvxpy as cp
import math
# import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import random
from random import sample

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from transformers import BertTokenizer

from conformal_utils2 import *
from utils import *
from FedQR import *
import json, time, csv, os
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--num_clients", default=5, type=int)
parser.add_argument("--num_class", default=1139, type=int)
parser.add_argument("--seed", default=123, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument('--num_epochs', default = 300, type = int, help = 'total number of epochs')
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument('--lr_schedule', nargs='+', type = int, help='in what epochs we want to decay the learning rates')
parser.add_argument('--optimizer', default = 'Adam', choices = ['SGD', 'Adam'], help = 'which optimizer, SGD or Adam')
parser.add_argument('--gamma', default = 0.1, type = float, help = 'initial learning rate')
parser.add_argument('--weight_decay', default = 1e-4, type = float, help = 'initial learning rate')
parser.add_argument("--dataset", default="rxrx1", type=str)
parser.add_argument("--score", default="APS", type=str)
parser.add_argument("--method", default="FCP_full", type=str)
# parser.add_argument("--num_workers", default=16, type=int)
# parser.add_argument("--num_cal_samples", default=10000, type=int)
parser.add_argument('--use_iid', type=str, default='no',
                    help='Whether to compute all results')
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--alpha", default=0.1, type=float)
parser.add_argument("--rho", default=1.0, type=float)
parser.add_argument('--lmbda_val',  default=0.01, type=float)
parser.add_argument('--k_reg', default=0, type=int)
parser.add_argument('--splits', default=10, type=int, help='Number of experiments to estimate mean set size and coverage')

parser.add_argument("--sketch_method", default="ddsketch", type=str)
parser.add_argument('--sigma',  default=0.1, type=float, help='STD for gaussian noise')
parser.add_argument('--t',  default=0.001, type=float, help='temperature for weight')
parser.add_argument('--num_sketch', default=512, type=int, help='Number of global samples after sketch')
# ---------------- DP (Gaussian) + DP-hist reweighting params ----------------
parser.add_argument('--dp_epsilon', default=1.0, type=float,help='Gaussian DP epsilon (set to enable DP). e.g., 1.0, 2.0, 5.0')
parser.add_argument('--dp_delta', default=1e-5, type=float, help='Gaussian DP delta (set to enable DP). e.g., 1e-5')
parser.add_argument('--dp_seed', default=33, type=int, help='Random seed for DP noise (for reproducibility). If None, uses nondeterministic RNG.')
# ---------------- DP histogram / PDF construction params ----------------
parser.add_argument('--pdf_bins', default=400, type=int,help='Number of bins for DP histogram over squashed scores in [0,1].')
parser.add_argument('--pdf_eps_floor', default=1e-12, type=float,help='Numerical floor for pdf evaluation to avoid log(0).')

# ---------------- Score squash-to-[0,1] params (avoid manual L,U) ----------------
parser.add_argument('--squash_kind', default='atan', type=str, choices=['atan', 'sigmoid'], help='Monotone squash mapping from raw scores to (0,1).')
parser.add_argument('--squash_s', default=1.0, type=float, help='Scale for squash mapping. Smaller -> more saturated near 0/1; larger -> smoother.')

args = parser.parse_args()

sys.path.insert(0, './')

def main():

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    t_wall_start = time.perf_counter()
    wall_start_ts = datetime.datetime.now().isoformat(timespec='seconds')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_end   = torch.cuda.Event(enable_timing=True)
        gpu_start.record()  

    # model, featurizer, classifier = load_model(args.dataset, device)

    # train_dataset, test_dataset = load_dataset(args.dataset)

    softmax_scores, labels, indices = load_soft_dataset(args.dataset)
    
    metadata = pd.read_csv(f'./dataset/{args.dataset}_metadata.csv')
    
    # feature = load_feature(args.dataset, indices, featurizer, device)
    
    # print("labels type:", type(labels))
    # print("labels dtype:", labels.dtype)
    # print("labels sample:", labels[:5])

    labels = labels.astype(int)

    metadata['label'] = labels

    if args.use_iid == 'yes':
        partition_type='iid'
    else:
        partition_type='non-iid'

    # client_loaders = load_federated_data(train_dataset, test_dataset, args.num_clients, iid, args.num_bins, args.batch_size)
    if args.method == 'FCP_full':
        base_path = "dataset={}/method={}/iid={}/clients={}/rho={}/alpha={}/".format(args.dataset, args.method, args.use_iid, args.num_clients, args.rho, args.alpha)
    elif args.method == 'FCP_LS': 
        base_path = "dataset={}/method={}/iid={}/clients={}/rho={}/alpha={}/sigma={}/".format(args.dataset, args.method, args.use_iid, args.num_clients, args.rho, args.alpha, args.sigma)
    elif args.method == 'FCP_full_dp': 
        base_path = "dataset={}/method={}/iid={}/clients={}/rho={}/alpha={}/sigma={}/".format(args.dataset, args.method, args.use_iid, args.num_clients, args.rho, args.alpha, args.dp_epsilon)
    elif args.method == 'FCP': 
        base_path = "dataset={}/method={}/iid={}/clients={}/rho={}/alpha={}/sketch_method={}/".format(args.dataset, args.method, args.use_iid, args.num_clients, args.rho, args.alpha, args.sketch_method)    
    else: 
        raise ValueError("Unsupported method")    
    
    dataset_path = "client_data/dataset={}/iid={}/clients={}/rho={}/".format(args.dataset, args.use_iid, args.num_clients, args.rho)
    
    patha = 'Results/'+ base_path + 'client_data/'
    
    if not os.path.exists(patha):
        os.makedirs(patha)

    result_folder = os.path.join(patha, f'alpha={args.alpha}/score={args.score}/')
    os.makedirs(result_folder, exist_ok=True)

    client_data, client_indices = prepare_or_load_client_data(softmax_scores, labels, metadata, args.dataset, dataset_path, args.num_clients, args, partition_type, seed=1)
    
    # client_feature = split_features_by_client(features, client_indices)
    # client_softmax_data = load_softmax_scores(model, client_loaders, output_dir=patha, device=device)

    all_result_rows = []

    for seed in tqdm(range(args.splits)):

        # client_scores = compute_conformal_scores_all_clients_2(client_softmax_data, seed, args.score, args)
        result_csv = os.path.join(result_folder, f'seed={seed}_conformal_results.csv')

 
        client_scores = compute_conformal_scores_all_clients_2(client_data, seed, args.score, args)

        # global_sketch_scores = federated_sketching(client_scores, args.sketch_method, args.sigma, args.bins, args.num_sketch)

        global_calib_scores = np.concatenate([client_scores[cid]['calib'] for cid in client_scores])
        global_calib_labels = np.concatenate([client_scores[cid]['calib_labels'] for cid in client_scores])

        global_test_scores = np.concatenate([client_scores[cid]['test'] for cid in client_scores])
        global_test_labels = np.concatenate([client_scores[cid]['test_labels'] for cid in client_scores])

        # args.num_sketch = global_calib_scores
        
        # print(len(global_calib_scores))

        # print(len(global_test_scores))

        # weighted_scores_dict = weighted_scores_2(client_scores, global_calib_scores, args.method)

        if args.method == 'FCP_full':

            if os.path.exists(result_csv):
                os.remove(result_csv)

            all_coverages = []
            all_sizes = []

            for target_cid in client_scores:
                scores_dict, weights_dict = weighted_scores_accurate_2(client_scores, args, target_cid)

                qhat, _ = compute_federated_global_quantile_3(scores_dict, weights_dict, args, key="calib", use_qr=False, w_x=1)

                test_scores = np.concatenate([client_scores[target_cid]['test']])
                test_labels = np.concatenate([client_scores[target_cid]['test_labels']])


                (coverages, sizes) = global_conformal_2(test_scores, test_labels, qhat, client_id=target_cid, save_path=result_csv)

                all_coverages.append(coverages)
                all_sizes.append(sizes)

            
            # weighted_scores_dict, weights_dict = weighted_scores_accurate_2(client_scores, args)

            # # weighted_scores_ref = weighted_scores_4(client_scores, global_calib_scores, args)

            # # qhat, _ = compute_federated_global_quantile_2(weighted_scores_dict, weights_dict, args, return_loss_curve=False)

            # qhat, _ = compute_federated_global_quantile_3(weighted_scores_dict, weights_dict, args,key="calib", use_qr=False, w_x=1)


            # # qhat_ref, _ = compute_federated_global_quantile(weighted_scores_ref, args, return_loss_curve=False)

            # print(qhat)

            # print(qhat_ref)

            # (coverages, sizes) = global_conformal(global_test_scores, global_test_labels, qhat, client_scores, result_csv)
        
        elif args.method == 'FCP_LS':
            weighted_scores_dict, q_by_target = weighted_scores_2(client_scores, args, args.method)

            if os.path.exists(result_csv):
                os.remove(result_csv)

            all_coverages = []
            all_sizes = []

            for target_cid in client_scores:

                test_scores = np.concatenate([client_scores[target_cid]['test']])
                test_labels = np.concatenate([client_scores[target_cid]['test_labels']])

                q = q_by_target[target_cid]

                (coverages, sizes) = global_conformal_per_class_2(test_scores, test_labels, q, client_id=target_cid, save_path=result_csv)

                all_coverages.append(coverages)
                all_sizes.append(sizes)

            # (coverages, sizes) = global_conformal_per_class(global_test_scores, global_test_labels, q, client_scores, result_csv)

        elif args.method == 'FCP':

            q_hat = distributed_quantile_from_scores(client_scores, args, args.sketch_method)

            if os.path.exists(result_csv):
                os.remove(result_csv)

            all_coverages = []
            all_sizes = []

            for target_cid in client_scores:

                test_scores = np.concatenate([client_scores[target_cid]['test']])
                test_labels = np.concatenate([client_scores[target_cid]['test_labels']])

                (coverages, sizes) = global_conformal_2(test_scores, test_labels, q_hat, client_id=target_cid, save_path=result_csv)

                all_coverages.append(coverages)
                all_sizes.append(sizes)
            
            # print(q_hat)

            # (coverages, sizes) = global_conformal(global_test_scores, global_test_labels, q_hat, client_scores, result_csv)

        elif args.method == 'FCP_full_dp':

            if os.path.exists(result_csv):
                os.remove(result_csv)

            all_coverages = []
            all_sizes = []

            for target_cid in client_scores:
                scores_dict, weights_dict = weighted_scores_accurate_2(client_scores, args, target_cid)

                qhat, _ = compute_federated_global_quantile_3(scores_dict, weights_dict, args, key="calib", use_qr=False, w_x=1)

                test_scores = np.concatenate([client_scores[target_cid]['test']])
                test_labels = np.concatenate([client_scores[target_cid]['test_labels']])


                (coverages, sizes) = global_conformal_2(test_scores, test_labels, qhat, client_id=target_cid, save_path=result_csv)

                all_coverages.append(coverages)
                all_sizes.append(sizes)

        else: 

            raise ValueError(f"Unsupported method: {args.method}")


        coverages = np.concatenate(all_coverages)
        sizes = np.concatenate(all_sizes)

        global_row = pd.DataFrame([{
            "client_id": "global",
            "coverage": float(np.mean(coverages)),
            "avg_set_size": float(np.mean(sizes)),
        }])
        if os.path.exists(result_csv):
            global_row.to_csv(result_csv, mode="a", header=False, index=False)
        else:
            global_row.to_csv(result_csv, index=False)


        save_coverage = os.path.join(result_folder, f'seed={seed}_coverages.pkl')

        save_size = os.path.join(result_folder, f'seed={seed}_size.pkl')

        with open(save_coverage, "wb") as f:
            pickle.dump(coverages, f)

        with open(save_size, "wb") as f:
            pickle.dump(sizes, f)
        
        df = pd.read_csv(result_csv)
        df['seed'] = seed
        all_result_rows.append(df)

    if all_result_rows:
        all_results_df = pd.concat(all_result_rows, ignore_index=True)

        # Group by client_id, compute mean and std
        grouped = all_results_df.groupby('client_id').agg({
            'coverage': ['mean', 'std'],
            'avg_set_size': ['mean', 'std']
        })

        # Flatten multi-index columns
        grouped.columns = ['coverage_mean', 'coverage_std', 'avg_set_size_mean', 'avg_set_size_std']
        grouped = grouped.reset_index()

        # Save to final result file
        result_final_csv = os.path.join(result_folder, f'final_conformal_results.csv')
        grouped.to_csv(result_final_csv, index=False)

        print(f"Saved aggregated results to {result_final_csv}")

        global_row = grouped[grouped['client_id'] == 'global']
        if not global_row.empty:
            row = global_row.iloc[0]
            print("=== Aggregated Global Conformal Results ===")
            print(f"Coverage Mean       : {row['coverage_mean']:.4f}")
            print(f"Coverage Std        : {row['coverage_std']:.4f}")
            print(f"Avg Set Size Mean   : {row['avg_set_size_mean']:.4f}")
            print(f"Avg Set Size Std    : {row['avg_set_size_std']:.4f}")
        else:
            print("[Warning] 'global' row not found in aggregated results.")

    else:
        print("No result CSVs found. Skipping aggregation.")

    gpu_elapsed_sec = 0.0
    if torch.cuda.is_available():
        gpu_end.record()
        torch.cuda.synchronize()
        gpu_elapsed_sec = gpu_start.elapsed_time(gpu_end) / 1000.0  

    # ---------------- Wall clock end ----------------
    wall_elapsed_sec = time.perf_counter() - t_wall_start
    wall_end_ts = datetime.datetime.now().isoformat(timespec='seconds')
    cpu_elapsed_sec = wall_elapsed_sec - gpu_elapsed_sec

    runtime_stats = {
        "start_time": wall_start_ts,
        "end_time"  : wall_end_ts,
        "wall_sec"  : round(wall_elapsed_sec, 3),
        "gpu_sec"   : round(gpu_elapsed_sec, 3),
        "cpu_sec"   : round(cpu_elapsed_sec, 3)
    }

    with open(os.path.join(result_folder, "runtime.json"), "w") as f:
        json.dump(runtime_stats, f, indent=2)

    print("\n=== RUNTIME SUMMARY ===")
    print(f"Wall‑clock : {wall_elapsed_sec/60:.2f} min "
          f"({wall_elapsed_sec:.1f} s)")
    print(f"GPU kernels: {gpu_elapsed_sec/60:.2f} min "
          f"({gpu_elapsed_sec:.1f} s)")
    print(f"CPU approx : {cpu_elapsed_sec/60:.2f} min "
          f"({cpu_elapsed_sec:.1f} s)")


if __name__ == '__main__':

    main()
