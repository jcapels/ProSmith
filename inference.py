
import gc
import os
from os.path import join
import shutil
import random
import argparse
import time
import logging
import numpy as np
from time import gmtime, strftime
import pandas as pd
import pickle
from tqdm import tqdm
import xgboost as xgb
from hyperopt import fmin, tpe, hp, Trials, rand
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef, r2_score, mean_squared_error, accuracy_score
from lifelines.utils import concordance_index


import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from code.preprocessing.protein_embeddings import calculate_protein_embeddings
from code.preprocessing.smiles_embeddings import calculate_smiles_embeddings
from code.training.utils.modules import (
    MM_TN,
    MM_TNConfig)

from code.training.utils.datautils import SMILESProteinDataset
from code.training.utils.train_utils import *

import os
import numpy as np
from os.path import join

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
fhandler = logging.FileHandler(filename= "inference" +'.txt', mode='a')
logger.addHandler(fhandler)

def extract_repr(model, dataloader, device):

    print("device: %s" % device)
    # evaluate the model on validation set
    model.eval()
    logging.info(f"Extracting repr")

    if is_cuda(device):
        model = model.to(device)
    
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            # move batch to device
            batch = [r.to(device) for r in batch]
            smiles_emb, smiles_attn, protein_emb, protein_attn, labels, indices = batch
            _, cls_repr = model(smiles_emb=smiles_emb, 
                                                    smiles_attn=smiles_attn, 
                                                    protein_emb=protein_emb,
                                                    protein_attn=protein_attn,
                                                    device=device,
                                                    gpu=0,
                                                    get_repr=True)

            protein_attn = int(sum(protein_attn.cpu().detach().numpy()[0]))
            smiles_attn = int(sum(smiles_attn.cpu().detach().numpy()[0]))

            smiles = smiles_emb[0][:smiles_attn].mean(0).cpu().detach().numpy()
            esm1b = protein_emb[0][:protein_attn].mean(0).cpu().detach().numpy()
            cls_rep = cls_repr[0].cpu().detach().numpy()

            if step ==0:
                cls_repr_all = cls_rep.reshape(1,-1)
                esm1b_repr_all = esm1b.reshape(1,-1)
                smiles_repr_all = smiles.reshape(1,-1)
                labels_all = labels[0]
                logging.info(indices.cpu().detach().numpy())
                orginal_indices = list(indices.cpu().detach().numpy())
            else:
                cls_repr_all = np.concatenate((cls_repr_all, cls_rep.reshape(1,-1)), axis=0)
                smiles_repr_all = np.concatenate((smiles_repr_all, smiles.reshape(1,-1)), axis=0)
                esm1b_repr_all = np.concatenate((esm1b_repr_all, esm1b.reshape(1,-1)), axis=0)
                labels_all = torch.cat((labels_all, labels[0]), dim=0)
                orginal_indices = orginal_indices + list(indices.cpu().detach().numpy())
    return cls_repr_all, esm1b_repr_all, smiles_repr_all, labels_all.cpu().detach().numpy(), orginal_indices


def set_param_values_V2(param, dtrain):
    depth_array = [6,7,8,9,10,11,12,13,14]
    num_round = int(param["num_rounds"])
    param["max_depth"] = int(depth_array[param["max_depth"]])
    param["tree_method"] = "gpu_hist"
    param["sampling_method"] = "gradient_based"
    param['objective'] = 'binary:logistic'
    weights = np.array([param["weight"] if y == 0 else 1.0 for y in dtrain.get_label()])
    dtrain.set_weight(weights)
    del param["num_rounds"]
    del param["weight"]
    return(param, num_round, dtrain)

def inference(dataset_dir, df):

    dataset_df = pd.read_csv(df)

    all_sequences = list(set(dataset_df["Protein sequence"]))
    all_smiles = list(set(dataset_df["SMILES"]))

    print("Calculating protein embeddings:")
    calculate_protein_embeddings(all_sequences, dataset_dir, 2000, device="cuda:0")

    print("Calculating SMILES embeddings:")
    calculate_smiles_embeddings(all_smiles, dataset_dir, 2000)

    with open(join("data/my_data_curated_stereo/train_val_compounds_04/saved_predictions", "best.pkl"), "rb") as f:
        best = pickle.load(f)
        
    config = MM_TNConfig.from_dict({"s_hidden_size":600,
        "p_hidden_size":1280,
        "hidden_size": 768,
        "max_seq_len":1276,
        "num_hidden_layers" : 6,
        "binary_task" : True})
    
    logging.info(f"Loading model")
    model = MM_TN(config)
    
    model = model.to("cuda:0")
    # model = DDP(model, device_ids=[gpu])

    try:
        state_dict = torch.load("data/my_data_curated_stereo/train_val_compounds_04/saved_model/40_compounds_4gpus_bs96_1e-05_layers6.txt.pkl", weights_only=True)
        new_model_state_dict = model.state_dict()
        for key in new_model_state_dict.keys():
            if key in state_dict.keys():
                try:
                    new_model_state_dict[key].copy_(state_dict[key])
                    #logging.info("Updatete key: %s" % key)
                except:
                    None
        model.load_state_dict(new_model_state_dict)
        logging.info("Successfully loaded pretrained model")
    except:
        new_state_dict = {}
        for key, value in state_dict.items():
            new_state_dict[key.replace("module.", "")] = value
        model.load_state_dict(new_state_dict)
        logging.info("Successfully loaded pretrained model (V2)")


    # Check if the file exists
    dataset = SMILESProteinDataset(
    data_path=df,
    embed_dir = os.path.join(dataset_dir),
    train=True,
    device="cuda:0", 
    gpu="cuda:0",
    random_state = 123,
    binary_task = True,
    extraction_mode = True) 

    # # Create samplers and dataloaders
    # trainsampler = DistributedSampler(dataset, shuffle=False, num_replicas=args.world_size, rank=gpu, drop_last=True)

    logging.info(f"Loading dataloader")
    trainloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    # Extract representations
    model = model.to("cuda:0")
    train_cls, train_esm1b, train_smiles, train_labels, _ = extract_repr(model, trainloader, "cuda:0")

    model = model.to("cpu")


    # Save all variables to a .npz file
    np.savez(
        join(dataset_dir, 'data.npz'),
        train_cls=train_cls, train_esm1b=train_esm1b, train_smiles=train_smiles, train_labels=train_labels,
    )
    
    logging.info(f"Extraction complete")

    ############# ESM1b+ChemBERTa +cls
    train_X_all_cls = np.concatenate([np.concatenate([train_esm1b, train_smiles], axis = 1), train_cls], axis=1)
    torch.cuda.empty_cache()
    del train_cls, train_esm1b, train_smiles, model

    # Force garbage collection
    gc.collect()

    # Clear PyTorch's CUDA cache
    torch.cuda.empty_cache()

    permutation = np.random.permutation(len(train_X_all_cls))

    # Shuffle both arrays using the same permutation
    train_X_all_cls = train_X_all_cls[permutation]

    slice_ = train_X_all_cls[:100]

    dtrain_val_all_cls = xgb.DMatrix(train_X_all_cls)
    slice_matrix = xgb.DMatrix(slice_, label=train_labels[:100])
    
    best, num_round, dtrain = set_param_values_V2(best, dtrain_val_all_cls)
    bst = xgb.train(best,  slice_matrix, num_round)

    predictions_proba = bst.predict(dtrain_val_all_cls)

if __name__ == "__main__":

    import datetime
    import os
    import time
    import tracemalloc
    import numpy as np
    import pandas as pd
    from hurry.filesize import size
    datasets = [
        # "dataset_100_100.csv", "dataset_300_1000.csv", 
        "dataset_700_5000.csv", 
                "dataset_7000.csv", 
                "curated_dataset.csv"]
    
    if os.path.exists("benchmark_results_inference.csv"):
        results = pd.read_csv("benchmark_results_inference.csv")
    else:
        results = pd.DataFrame()

    os.makedirs("efficiency_evaluation", exist_ok=True)

    for dataset in datasets:
        tracemalloc.start()
        start = time.time()

        inference("efficiency_evaluation", dataset)

        end = time.time()
        print("Time spent: ", end - start)
        print("Memory needed: ", tracemalloc.get_traced_memory())
        dataset_df = pd.read_csv(dataset)
        unique_substrates_dataset = np.unique(dataset_df["Substrate ID"])
        num_unique_substrates = len(unique_substrates_dataset)
        unique_enzymes_dataset = np.unique(dataset_df["Enzyme ID"])
        num_unique_enzymes = len(unique_enzymes_dataset)
        num_rows = dataset_df.shape[0]

        results = pd.concat((results, 
                                pd.DataFrame({
                                                "unique_enzymes": [num_unique_enzymes],
                                                "unique_substrates": [num_unique_substrates],
                                                "num_pairs": [num_rows],
                                                "time": [str(datetime.timedelta(seconds=end - start))], 
                                                "memory": [size(int(tracemalloc.get_traced_memory()[1]))]})), 
                                                ignore_index=True, axis=0)
        tracemalloc.stop()

        results.to_csv("benchmark_results_inference.csv", index=False)

