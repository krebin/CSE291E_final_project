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
np.random.seed(42)


import argparse
parser = argparse.ArgumentParser(description='Choose a config file')

# experiment
parser.add_argument(
    '--experiment',
    default='dummy',
    help='Choose a config file (default: \'base\')'
)

args = parser.parse_args()

# grab values from arguments
experiment = args.experiment

# base architecture
if experiment == "base1":
    import base1_config as cfg
    from base_model import BaseModel as Model
else:
    import dummy1_config as cfg
    from base_model import BaseModel as Model

    experiment = "dummy1"

# Get argument from config
cfg = cfg.cfg
batch_size = cfg["batch_size"]
valid_batch_size = cfg["valid_batch_size"]
num_workers = cfg["num_workers"]
epochs = cfg["epochs"]
lr = cfg["lr"]
model_type = experiment


if __name__ == "__main__":
    cb513_data = pkl.load(open("CB513.pkl", "rb"))
    ids = len(cb513_data)

    if experiment == "dummy1":
        test_loader, len_test = get_loader(protein_data=cb513_data,
                                           ids=[0, 1, 2],
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=num_workers)
    else:
        test_loader, len_test = get_loader(protein_data=cb513_data,
                                           ids=ids,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=num_workers)
    
    if not os.path.exists(model_type):
        os.mkdir(model_type)

    best_model_path = os.path.join("models", model_type, "best_model.pt")
    stats_path = os.path.join("stats", model_type, "stats.pkl")

    if os.path.exists(best_model_path):
        print("Model exists. Loading from {0}".format(best_model_path))
        model = torch.load(best_model_path)

    else:
        print("NO BEST MODEL")
        print("Exiting...")        
        exit()

    model.cuda()
    print("Model is using GPU: {0}".format(next(model.parameters()).is_cuda))

    acc = test(model, test_loader)
    
    with open(stats_path, "rb") as f:
        stats_dict = pkl.load(f)
        
    stats_dict["test"]["acc"] = acc

    with open(stats_path, "wb") as f:
        pkl.dump(stats_dict, f)
