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

from utils.modules import (
    MM_TN,
    MM_TNConfig)

from utils.datautils import SMILESProteinDataset
from utils.train_utils import *




n_gpus = len(list(range(torch.cuda.device_count())))


###########################################################################

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="The input train dataset",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        required=True,
        help="The input val dataset",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="The input val dataset",
    )
    parser.add_argument(
        "--embed_path",
        type=str,
        required=True,
        help="Path that contains subfolders SMILES and Protein with embedding dictionaries",
    )
    parser.add_argument(
        "--save_pred_path",
        type=str,
        default="",
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--binary_task",
        default=False,
        type=bool,
        help="Specifies wether the target variable is binary or continous.",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        required=True,
        help="Path of trained Transformer Network.",
    )
    parser.add_argument(
        "--num_hidden_layers",
        default=6,
        type=int,
        help="The num_hidden_layers size of MM_TN",
    )
    parser.add_argument(
        '--port',
        default=12557,
        type=int,
        help='Port for tcp connection for multiprocessing'
    )
    parser.add_argument(
        '--log_name',
        default="",
        type=str,
        help='Will be added to the file name of the log file'
    )
    parser.add_argument(
        "--class_name",
        type=str,
        required=True,
        help="The class name to evaluate",
    )
    return parser.parse_args()

args = get_arguments()
setting = args.log_name + '_gpus' + str(n_gpus) +'_layers' + str(args.num_hidden_layers) +'_xgboost_training'
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
fhandler = logging.FileHandler(filename= setting +'.txt', mode='a')
logger.addHandler(fhandler)

def extract_repr(args, model, dataloader, device):
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



depth_array = [6,7,8,9,10,11,12,13,14]
space_gradient_boosting = {"learning_rate": hp.uniform("learning_rate", 0.01, 0.5),
    "max_depth": hp.choice("max_depth", depth_array),
    "reg_lambda": hp.uniform("reg_lambda", 0, 5),
    "reg_alpha": hp.uniform("reg_alpha", 0, 5),
    "max_delta_step": hp.uniform("max_delta_step", 0, 5),
    "min_child_weight": hp.uniform("min_child_weight", 0.1, 15),
    "num_rounds":  hp.uniform("num_rounds", 30, 1000),
    "weight" : hp.uniform("weight", 0.01,0.99)}


def evaluate(gpu, args, device):
    for seed in range(5):

        os.makedirs(args.save_pred_path, exist_ok=True)

        with open(join(args.save_pred_path, "best.pkl"), "rb") as f:
            best = pickle.load(f)
            
        logging.info(args)

        # if is_cuda(device):
        #     setup(gpu, args.world_size, str(args.port))
        #     torch.manual_seed(0)
        #     torch.cuda.set_device(gpu)
        

        config = MM_TNConfig.from_dict({"s_hidden_size":600,
            "p_hidden_size":1280,
            "hidden_size": 768,
            "max_seq_len":1276,
            "num_hidden_layers" : args.num_hidden_layers,
            "binary_task" : args.binary_task})
        
        def get_predictions(param, dM_train, dM_val):
            param, num_round, dM_train = set_param_values_V2(param = param, dtrain = dM_train)
            bst = xgb.train(param,  dM_train, num_round)
            y_val_pred = bst.predict(dM_val)
            return(y_val_pred)
            
        def get_performance_metrics(pred, true):
            if args.binary_task:
                acc = np.mean(np.round(pred) == np.array(true))
                roc_auc = roc_auc_score(np.array(true), pred)
                mcc = matthews_corrcoef(np.array(true),np.round(pred))
                logging.info("accuracy: %s,ROC AUC: %s, MCC: %s" % (acc, roc_auc, mcc))
            else:
                mse = mean_squared_error(true, pred)
                CI = concordance_index(true, pred)
                rm2 = get_rm2(ys_orig = true, ys_line = pred)
                R2 = r2_score(true, pred)
                logging.info("MSE: %s,R2: %s, rm2: %s, CI: %s" % (mse, R2, rm2, CI))
        
        def set_param_values(param, dtrain):
            num_round = int(param["num_rounds"])
            param["tree_method"] = "gpu_hist"
            param["sampling_method"] = "gradient_based"
            if not args.binary_task:
                param['objective'] = 'reg:squarederror'
                weights = None
            else:
                param['objective'] = 'binary:logistic'
                weights = np.array([param["weight"] if y == 0 else 1.0 for y in dtrain.get_label()])
                dtrain.set_weight(weights)

            del param["num_rounds"]
            del param["weight"]
            return(param, num_round)

        def set_param_values_V2(param, dtrain):
            num_round = int(param["num_rounds"])
            param["max_depth"] = int(depth_array[param["max_depth"]])
            param["tree_method"] = "gpu_hist"
            param["sampling_method"] = "gradient_based"
            if not args.binary_task:
                param['objective'] = 'reg:squarederror'
                weights = None
            else:
                param['objective'] = 'binary:logistic'
                weights = np.array([param["weight"] if y == 0 else 1.0 for y in dtrain.get_label()])
                dtrain.set_weight(weights)
            del param["num_rounds"]
            del param["weight"]
            return(param, num_round, dtrain)


            
        def get_performance(pred, true):
            if args.binary_task:
                MCC = matthews_corrcoef(true, np.round(pred))
                return(-MCC)
            else:
                MSE = mean_squared_error(true, pred)
                return(MSE)


        logging.info(f"Loading dataset to {device}:{gpu}")
        logging.info(f"Loading model")
        model = MM_TN(config)
        
        if is_cuda(device):
            model = model.to(gpu)
            # model = DDP(model, device_ids=[gpu])

        if os.path.exists(args.pretrained_model) and args.pretrained_model != "":
            logging.info(f"Loading model")
            try:
                state_dict = torch.load(args.pretrained_model, weights_only=True)
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

        else:
            logging.info("Model path is invalid, cannot load pretrained Interbert model")
        

        # Check if the file exists
        if os.path.exists(join(args.save_pred_path, f"data_{args.class_name}.npz")):
            # Load all variables from the .npz file
            data = np.load(join(args.save_pred_path, f"data_{args.class_name}.npz"))

            # Assign loaded variables
            train_cls, train_esm1b, train_smiles, train_labels = (
                data['train_cls'], data['train_esm1b'], data['train_smiles'], data['train_labels']
            )
            val_cls, val_esm1b, val_smiles, val_labels = (
                data['val_cls'], data['val_esm1b'], data['val_smiles'], data['val_labels']
            )
            test_cls, test_esm1b, test_smiles, test_labels = (
                data['test_cls'], data['test_esm1b'], data['test_smiles'], data['test_labels']
            )
        else:
            train_dataset = SMILESProteinDataset(
            data_path=args.train_dir,
            embed_dir = args.embed_path,
            train=True,
            device=device, 
            gpu=gpu,
            random_state = seed,
            binary_task = args.binary_task,
            extraction_mode = True) 

            val_dataset = SMILESProteinDataset(
                data_path=args.val_dir,
                embed_dir = args.embed_path,
                train=False, 
                device=device, 
                gpu=gpu,
                random_state = seed,
                binary_task = args.binary_task,
                extraction_mode = True)
                
            test_dataset = SMILESProteinDataset(
                data_path=args.test_dir,
                embed_dir = args.embed_path,
                train=False, 
                device=device, 
                gpu=gpu,
                random_state = seed,
                binary_task = args.binary_task,
                extraction_mode = True)

            # Create samplers and dataloaders
            trainsampler = DistributedSampler(train_dataset, shuffle=False, num_replicas=args.world_size, rank=gpu, drop_last=True)
            valsampler = DistributedSampler(val_dataset, shuffle=False, num_replicas=args.world_size, rank=gpu, drop_last=True)
            testsampler = DistributedSampler(test_dataset, shuffle=False, num_replicas=args.world_size, rank=gpu, drop_last=True)

            logging.info(f"Loading dataloader")
            trainloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1, sampler=trainsampler)
            valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, sampler=valsampler)
            testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, sampler=testsampler)

            # Extract representations
            val_cls, val_esm1b, val_smiles, val_labels, _ = extract_repr(args, model, valloader, device)
            test_cls, test_esm1b, test_smiles, test_labels, test_indices = extract_repr(args, model, testloader, device)
            train_cls, train_esm1b, train_smiles, train_labels, _ = extract_repr(args, model, trainloader, device)

            # Save all variables to a .npz file
            np.savez(
                join(args.save_pred_path, f"data_{args.class_name}.npz"),
                train_cls=train_cls, train_esm1b=train_esm1b, train_smiles=train_smiles, train_labels=train_labels,
                val_cls=val_cls, val_esm1b=val_esm1b, val_smiles=val_smiles, val_labels=val_labels,
                test_cls=test_cls, test_esm1b=test_esm1b, test_smiles=test_smiles, test_labels=test_labels
            )


        logging.info(str(len(test_labels)))
        
        logging.info(f"Extraction complete")
        
        

        ############# ESM1b+ChemBERTa2
        
        train_X_all = np.concatenate([train_esm1b, train_smiles], axis = 1)
        val_X_all = np.concatenate([val_esm1b, val_smiles], axis = 1)

        dtrain = xgb.DMatrix(np.concatenate([np.array(train_X_all), np.array(val_X_all)], axis = 0),
                                    label = np.concatenate([np.array(train_labels).astype(float),np.array(val_labels).astype(float)], axis = 0))
        
        ############# ESM1b+ChemBERTa +cls
        train_X_all_cls = np.concatenate([np.concatenate([train_esm1b, train_smiles], axis = 1), train_cls], axis=1)
        test_X_all_cls = np.concatenate([np.concatenate([test_esm1b, test_smiles], axis = 1), test_cls], axis=1)
        val_X_all_cls = np.concatenate([np.concatenate([val_esm1b, val_smiles], axis = 1), val_cls], axis=1)

        train_X_all_cls = np.concatenate([np.array(train_X_all_cls), np.array(val_X_all_cls)], axis = 0)
        train_y_all = np.concatenate([np.array(train_labels).astype(float),np.array(val_labels).astype(float)], axis = 0)

        np.random.seed(seed)

        permutation = np.random.permutation(len(train_X_all_cls))

        # Shuffle both arrays using the same permutation
        train_X_all_cls = train_X_all_cls[permutation]
        train_y_all = train_y_all[permutation]

        dtest_all_cls = xgb.DMatrix(np.array(test_X_all_cls), label = np.array(test_labels).astype(float))
        dtrain_val_all_cls = xgb.DMatrix(train_X_all_cls,
                                    label = train_y_all)
        
        best["seed"] = seed
        
        best, num_round, dtrain = set_param_values_V2(best, dtrain)
        bst = xgb.train(best,  dtrain_val_all_cls, num_round)

        predictions_proba = bst.predict(dtest_all_cls)
        predictions = np.round(predictions_proba)

        accuracy = accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions)
        recall = recall_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions)
        roc_auc = roc_auc_score(test_labels, predictions_proba)
        mcc = matthews_corrcoef(test_labels, predictions)

        results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'recall': recall,
            'precision': precision,
            'roc_auc': roc_auc,
            'mcc': mcc,
            'seed':seed,
            'class_name': args.class_name
        }
        df = pd.DataFrame([results])
        if os.path.exists(join(args.save_pred_path, "class_results.csv")):
            file_exists=True
        else:
            file_exists=False

        df.to_csv(join(args.save_pred_path, "class_results.csv"), mode='a', index=False, header=not file_exists)

if __name__ == '__main__':
    # Set up the device
    
    # Check if multiple GPUs are available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_ids = list(range(torch.cuda.device_count()))
        gpus = 1
        args.world_size = gpus
        
    else:
        device = torch.device('cpu')
        args.world_size = -1

        
    evaluate(0, args, device)