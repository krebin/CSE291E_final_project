from torchvision import utils
from data_loader import *
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from train_utils import *
import pickle as pkl
import time
import itertools
import random
import os
import json
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
    if "dummy" in experiment:
        tr5534_data = json.load(open("CB513.json", "r"))
    else:
        tr5534_data = json.load(open("TR5534.json", "r"))

    cb513_data = json.load(open("CB513.json", "r"))

    len_train = len(tr5534_data)
    percent_train = .8

    train_start = 0
    train_end = int(len_train * percent_train)

    val_start = train_end
    val_end = len_train

    print(train_start, train_end)
    print(val_start, val_end)

    ids = np.random.choice(len_train, len_train, replace=False)

    if experiment == "dummy1":
        train_loader, len_train = get_loader(protein_data=tr5534_data,
                                             ids=[0, 1, 2],
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers)

        val_loader, len_val = get_loader(protein_data=tr5534_data,
                                         ids=[0, 1, 2],
                                         batch_size=valid_batch_size,
                                         shuffle=False,
                                         num_workers=num_workers)


    else:
        train_loader, len_train = get_loader(protein_data=tr5534_data,
                                             ids=ids[train_start:train_end],
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)

        val_loader, len_val = get_loader(protein_data=tr5534_data,
                                         ids=ids[val_start:val_end],
                                         batch_size=valid_batch_size,
                                         shuffle=True,
                                         num_workers=num_workers)

    print(len_train, len_val)

    if not os.path.exists(os.path.join("models", model_type)):
        os.makedirs(os.path.join("stats", model_type), exist_ok=True)
        os.makedirs(os.path.join("models", model_type), exist_ok=True)

    latest_model_path = os.path.join("models", model_type, "latest_model.pt")
    best_model_path = os.path.join("models", model_type, "best_model.pt")
    optim_path = os.path.join("models", model_type, "optim.pt")
    stats_path = os.path.join("stats", model_type, "stats.pkl")


    def init_weights(m):
        try:
            torch.nn.init.kaiming_uniform_(m.weight.data)
            m.bias.data.zero_()
        except:
            pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=8)
    model.apply(init_weights)
    print(type(model))

    if os.path.exists(latest_model_path):
        print("Model exists. Loading from {0}".format(latest_model_path))
        model = torch.load(latest_model_path)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    if os.path.exists(optim_path):
        print("Optimizer state dict exists. Loading from {0}".format(optim_path))
        optim = torch.load(optim_path)
        optimizer.load_state_dict(optim['optimizer'])

    model.to(device)
    print("Model is using GPU: {0}".format(next(model.parameters()).is_cuda))

    stats_dict, model = train(epochs, model, stats_path, train_loader, val_loader, optimizer, criterion,
                              len_train, len_val, latest_model_path, best_model_path, optim_path,device)
