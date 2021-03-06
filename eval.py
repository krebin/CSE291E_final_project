from torchvision import utils
from data_loader import *
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from PIL import Image
from train_utils import *
import pickle as pkl
import time
import itertools
import matplotlib.pyplot as plt
import random
import os
import csv
import torch.nn as nn
import json
np.random.seed(42)


import argparse
parser = argparse.ArgumentParser(description='Choose a config file')

# experiment
parser.add_argument(
    '--experiment',
    default='dummy',
    help='Choose a config file (default: \'base\')'
)

# ensemble
parser.add_argument(
    '--num_models',
    default=1,
    help='Choose the number of models to ensemble (default: 1)'
)

args = parser.parse_args()

# grab values from arguments
experiment = args.experiment
num_models = int(args.num_models)

# base architecture
if "base" in experiment:
    import base1_config as cfg
    from base_model import BaseModel as Model
elif experiment == "prot_vec":
    import prot_vec_config as cfg
    from base_model import BaseModel as Model
elif experiment == "pssm_only":
    import pssm_only_config as cfg
    from base_model import BaseModel as Model
elif experiment == "one_hot_only":
    import one_hot_only_config as cfg
    from base_model import BaseModel as Model
elif experiment == "prot_vec_baseline":
    import prot_vec_baseline_config as cfg
    from base_model import BaseModel as Model
elif experiment == "lstm":
    import base1_config as cfg
    from lstm_model import LSTMModel as Model
elif experiment == "rezero":
    import base1_config as cfg
    from rezero_model import ReZeroModel as Model
elif experiment == "residual":
    import residual_config as cfg
    from residual_model import ResidualModel as Model
elif experiment == "residual2":
    import residual2_config as cfg
    from residual_model import ResidualModel as Model
elif experiment == "residual3":
    import residual2_config as cfg
    from residual_model import ResidualModel as Model
elif experiment == "residual_bnorm":
    import residual_config as cfg
    from residual_bnorm_model import ResidualModel as Model
elif experiment == "wavezero":
    import wavezero_config as cfg
    from wavezero_model import ResidualModel as Model
else:
    import dummy1_config as cfg
    from base_model import BaseModel as Model
    experiment = "dummy1"

# Get argument from config
cfg = cfg.cfg
batch_size = cfg["test_batch_size"]
valid_batch_size = cfg["valid_batch_size"]
num_workers = cfg["num_workers"]
epochs = cfg["epochs"]
lr = cfg["lr"]
num_features = cfg["num_features"] # base=51, prot_vec=100
one_hot_embed = cfg["one_hot_embed"]
model_type = experiment

models = []

# if model_type == "prot_vec":
#     prot_vec = True
# else:
#     prot_vec = False

print(model_type)

if __name__ == "__main__":
    
    if experiment == "prot_vec":
        cb513_data = json.load(open("CB513_prot_vec_only.json", "r"))
    elif experiment == "pssm_only":
        cb513_data = json.load(open("CB513_pssm_only.json", "r"))
    elif experiment == "one_hot_only":
        cb513_data = json.load(open("CB513_one_hot_only.json", "r"))
    elif experiment == "prot_vec_baseline":
        cb513_data = json.load(open("CB513_prot_vec_baseline.json", "r"))
    else:
        cb513_data = json.load(open("CB513.json", "r"))
    
    ids = np.arange(len(cb513_data))

    if experiment == "dummy1":
        test_loader, len_test = get_loader(protein_data=cb513_data,
                                           ids=[0, 1, 2],
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=num_workers,
                                           num_features=num_features)
    else:
        test_loader, len_test = get_loader(protein_data=cb513_data,
                                           ids=ids,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=num_workers,
                                           num_features=num_features)
    
    if not os.path.exists(model_type):
        os.mkdir(model_type)

    models = []

    for model_num in range(1,num_models+1):
        
        best_model_path = os.path.join("models", model_type+str(model_num), "best_model.pt")
        stats_path = os.path.join("stats", model_type+str(model_num), "stats.pkl")

        if os.path.exists(best_model_path):
            print("Model exists. Loading from {0}".format(best_model_path))
            model = torch.load(best_model_path)
        else:
            print("NO BEST MODEL")
            print("Exiting...")        
            exit()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print("Model is using GPU: {0}".format(next(model.parameters()).is_cuda))
        
        models.append(model)
    
    
    # Use esemble test if more than one model
    if num_models > 1:
        acc = ensemble_test(models, test_loader, device, num_features, one_hot_embed)
    else:
        acc = test(model, test_loader,device,num_features,one_hot_embed)

    print(acc)
    
    with open(stats_path, "rb") as f:
        stats_dict = pkl.load(f)
        
    stats_dict["test"]["acc"] = acc

    with open(stats_path, "wb") as f:
        pkl.dump(stats_dict, f)
