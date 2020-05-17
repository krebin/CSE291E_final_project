import os
import numpy as np
import pickle as pkl
import time
import torch
from collections import defaultdict
from utils import *
import json


def rec_dd():
    return defaultdict(rec_dd)


def train(epochs, model, stats_path,
          train_loader, val_loader,
          optimizer, criterion,
          len_train, len_val,
          latest_model_path,
          best_model_path, optim_path):

    fmt_string = "Epoch[{0}/{1}], Batch[{3}/{4}], Train Loss: {2}"

    # Load stats if path exists
    if os.path.exists(stats_path):
        with open(stats_path, "rb") as f:
            stats_dict = pkl.load(f)
        print(stats_dict["best_epoch"])
        start_epoch = stats_dict["next_epoch"]
        min_val_loss = stats_dict["valid"][stats_dict["best_epoch"]]["loss"]
        print("Stats exist. Loading from {0}. Starting from Epoch {1}".format(stats_path, start_epoch))
    else:
        min_val_loss = np.inf
        stats_dict = rec_dd()
        start_epoch = 0

        # See loss before training
        accs, val_loss = val(-1, model, val_loader, len_val, criterion, epochs)

        # Update statistics dict
        stats_dict["valid"][-1]["accs"] = accs
        stats_dict["valid"][-1]["loss"] = val_loss

    model.train()
    for epoch in range(start_epoch, epochs):
        train_loss = 0.
        all_labels = []
        all_predictions = []

        ts = time.time()
        for iter, (X, Y, seq_lens) in enumerate(train_loader):
            optimizer.zero_grad()

            X = X.view([-1, 51, 700]).cuda()
            Y = Y.view([-1, 700, 9])

            outputs = model(X)

            T = Y.argmax(dim=1).long().cuda()
            loss = criterion(outputs, T)
            train_loss += (loss.item() * len(X))

            labels = Y.argmax(dim=2).cpu().numpy()
            predictions = outputs.argmax(axis=2).cpu().numpy()

            for label, prediction, length in zip(labels, predictions, seq_lens):
                all_labels += list(label[:length])
                all_predictions += list(prediction[:length])

            if iter % 10 == 0:
                print(fmt_string.format(epoch, epochs, loss.item(), iter, len(train_loader)))

            loss.backward()
            optimizer.step()

        print("\nFinished Epoch {}, Time elapsed: {}, Loss: {}".format(epoch, time.time() - ts,
                                                                       train_loss / len_train))

        # Avg train loss. Batch losses were un-averaged before when added to train_loss
        labels = np.hstack(all_labels)
        predictions = np.hstack(all_predictions)

        stats_dict["train"][epoch]["loss"] = train_loss / len_train
        stats_dict["train"][epoch]["acc"] = np.mean(labels == predictions)

        # The validation stats after additional epoch
        accs, val_loss = val(epoch, model, val_loader, len_val, criterion, epochs)

        # Update statistics dict
        stats_dict["valid"][epoch]["accs"] = accs
        stats_dict["valid"][epoch]["loss"] = val_loss
        stats_dict["next_epoch"] = epoch + 1

        # Save latest model
        torch.save(model, latest_model_path)

        # Save optimizer state dict
        optim_state = {'optimizer': optimizer.state_dict()}
        torch.save(optim_state, optim_path)

        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            # Save best model
            torch.save(model, best_model_path)
            stats_dict["best_epoch"] = epoch

        # Save stats
        with open(stats_path, "wb") as f:
            pkl.dump(stats_dict, f)

        # Set back to train mode
        model.train()

    return stats_dict, model


def val(epoch, model, val_loader, len_val, criterion, epochs):
    # Complete this function - Calculate loss, accuracy and IoU for every epoch
    # Make sure to include a softmax after the output from your model

    fmt_string = "Epoch[{0}/{1}], Batch[{3}/{4}], Batch Validation Loss: {2}"
    all_labels = []
    all_predictions = []
    model.eval()
    num_val_batches = len(val_loader)
    loss = 0.
    with torch.no_grad():
        for iter, (X, Y, seq_lens) in enumerate(val_loader):

            X = X.view([-1, 51, 700]).cuda()
            Y = Y.view([-1, 700, 9])

            outputs = model(X)

            T = Y.argmax(dim=1).long().cuda()
            batch_loss = criterion(outputs, T).item()

            # Unaverage to do total average later b/c last batch may have unequal number of samples
            loss += (batch_loss * len(X))

            labels = Y.argmax(dim=2).cpu().numpy()
            predictions = outputs.argmax(axis=2).cpu().numpy()

            for label, prediction, length in zip(labels, predictions, seq_lens):
                all_labels += list(label[:length])
                all_predictions += list(prediction[:length])

            if iter % 10 == 0:
                print(fmt_string.format(epoch, epochs, batch_loss, iter, num_val_batches))

    # Avg loss
    loss /= len_val
    print("Total Validation Loss: {0}".format(loss))

    labels = np.array(all_labels)
    predictions = np.array(all_predictions)

    accs = np.mean(labels == predictions)
    return accs, loss


def test(model, test_loader):
    all_labels = []
    all_predictions = []
    model.eval()

    fmt_string = "Batch[{0}/{1}]"
    with torch.no_grad():
        for iter, (X, Y, seq_lens) in enumerate(test_loader):
            X = X.view([-1, 51, 700]).cuda()
            Y = Y.view([-1, 700, 9])

            outputs = model(X)

            if iter % 10 == 0:
                print(fmt_string.format(iter, len(test_loader)))

            # cross entropy does softmax so we can take index of max of outputs as prediction
            labels = Y.argmax(dim=2).cpu().numpy()
            predictions = outputs.argmax(axis=2).cpu().numpy()

            for label, prediction, length in zip(labels, predictions, seq_lens):
                print(label[:length])
                print("pred:")
                print(prediction[:length])
                all_labels += list(label[:length])
                all_predictions += list(prediction[:length])

    labels = np.array(all_labels)
    predictions = np.array(all_predictions)
    accs = np.mean(labels == predictions)
    return accs
