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
          best_model_path, optim_path, device, num_features, one_hot_embed, early_stop=10):

    fmt_string = "Epoch[{0}/{1}], Batch[{3}/{4}], Train Loss: {2}"

    # # Load stats if path exists
    # if os.path.exists(stats_path):
    #     with open(stats_path, "rb") as f:
    #         stats_dict = pkl.load(f)
    #     print(stats_dict["best_epoch"])
    #     start_epoch = stats_dict["next_epoch"]
    #     min_val_loss = stats_dict["valid"][stats_dict["best_epoch"]]["loss"]
    #     print("Stats exist. Loading from {0}. Starting from Epoch {1}".format(stats_path, start_epoch))
    # else:
    min_val_loss = np.inf
    stats_dict = rec_dd()
    start_epoch = 0

    # See loss before training
    accs, val_loss = val(-1, model, val_loader, len_val, criterion, epochs, device, num_features, one_hot_embed)

    # Update statistics dict
    stats_dict["valid"][-1]["acc"] = accs
    stats_dict["valid"][-1]["loss"] = val_loss
    torch.cuda.empty_cache()

    model.train()
    for epoch in range(start_epoch, epochs):
        train_loss = 0.
        all_labels = []
        all_predictions = []

        ts = time.time()
        for iter, (X, Y, seq_lens) in enumerate(train_loader):
            optimizer.zero_grad()

#             if (prot_vec):
#                 X = X.reshape([-1, 700, 100]).to(device)
#             else:
#                 X = X.reshape([-1, 700, 51]).to(device)
            X = X.reshape([-1, 700, num_features]).to(device)

            X = X.permute(0, 2, 1)
            Y = Y.view([-1, 700, 9])
            T = Y.argmax(dim=2).long().to(device)

            # print("DEBUG: TRAIN--")
            outputs = model(X,device,one_hot_embed)
            # print("DEBUG OUTPUT SIZE: ", outputs.size())
            # print("DEBUG TARGET SIZE: ", T.size())
            loss = criterion(outputs.permute(0, 2, 1), T)
            train_loss += (loss.item() * len(X))

            labels = Y.argmax(dim=2).cpu().numpy()
            predictions = outputs.argmax(axis=2).cpu().detach().numpy()

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
        accs, val_loss = val(epoch, model, val_loader, len_val, criterion, epochs, device, num_features, one_hot_embed)

        # Update statistics dict
        stats_dict["valid"][epoch]["acc"] = accs
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
        else:
            early_stop -= 1

        # Save stats
        with open(stats_path, "wb") as f:
            pkl.dump(stats_dict, f)
        
        if early_stop == 0:
            print('='*10,'Early stopping.','='*10)
            break

        # Set back to train mode
        model.train()

    return stats_dict, model


def val(epoch, model, val_loader, len_val, criterion, epochs, device, num_features, one_hot_embed):
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

#             if prot_vec:
#                 X = X.reshape([-1, 700, 100]).to(device)
#             else:
#                 X = X.reshape([-1, 700, 51]).to(device)

            X = X.reshape([-1, 700, num_features]).to(device)


            X = X.permute(0, 2, 1)
            Y = Y.view([-1, 700, 9])
            T = Y.argmax(dim=2).long().to(device)

            # print("DEBUG: VAL--", '-' * 30)
            outputs = model(X, device, one_hot_embed)
            batch_loss = criterion(outputs.permute(0, 2, 1), T).item()           

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
    torch.cuda.empty_cache()
    return accs, loss


def test(model, test_loader, device, num_features, one_hot_embed, experiment):
    all_labels = []
    all_predictions = []
    model.eval()

    fmt_string = "Batch[{0}/{1}]"
    with torch.no_grad():
        for iter, (X, Y, seq_lens) in enumerate(test_loader):
            
#             if (prot_vec):
#                 X = X.reshape([-1, 700, 100]).to(device)                
#             else:
#                 X = X.reshape([-1, 700, 51]).to(device)

            X = X.reshape([-1, 700, num_features]).to(device)

                
            X = X.permute(0, 2, 1)

            Y = Y.view([-1, 700, 9])

            outputs = model(X, device, one_hot_embed)

            if iter % 10 == 0:
                print(fmt_string.format(iter, len(test_loader)))

            # cross entropy does softmax so we can take index of max of outputs as prediction
            labels = Y.argmax(dim=2).cpu().numpy()
            predictions = outputs.argmax(axis=2).cpu().numpy()

            for label, prediction, length in zip(labels, predictions, seq_lens):
#                 print(label[:length])
#                 print("pred:")
#                 print(prediction[:length])
                all_labels += list(label[:length])
                all_predictions += list(prediction[:length])

    labels = np.array(all_labels)
    predictions = np.array(all_predictions)
    
     # Calc test precision, recall, and F1
    precision_recall_f1(predictions, labels)
    accs = np.mean(labels == predictions)

    # Evaluate loss, acc, conf. matrix, and class. report on devset
    class_report_conf_matrix(predictions, labels, experiment)

    return accs
