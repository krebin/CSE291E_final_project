import numpy as np
import pickle as pkl
import json
import matplotlib.pyplot as plt
import os
from PIL import Image
import torch.nn.functional as F
from data_loader import *
import pandas as pd

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from seqeval.metrics import classification_report

colors = ["midnightblue", "maroon", "darkgreen", "indigo", "black", "darkslateblue", "purple", "cyan"]
temps = [0.1, 0.2, 0.7, 1.0, 1.5, 2.0]

def json_fix_keys(x):
    if isinstance(x, dict):
        key = list(x.keys())[0]
        if key.isnumeric():
            return {int(k): v for k,v in x.items()}
        else:
            return {k: v for k,v in x.items()}
        
    return x

def format_stats(stats_dict, length=75):
    print(length)
    train_info = {"loss": []}
    valid_info = {"loss": [stats_dict["valid"]['-1']["loss"]],
                  "bleu4_stoch": [stats_dict["valid"]['-1']["stoch"]["bleu4"]],
                  "bleu1_stoch": [stats_dict["valid"]['-1']["stoch"]["bleu1"]],
                  "bleu4_det": [stats_dict["valid"]['-1']["det"]["bleu4"]],
                  "bleu1_det": [stats_dict["valid"]['-1']["det"]["bleu1"]]}
    
    best_epoch = stats_dict["best_epoch"] - 1

    
    for epoch in range(min(len(stats_dict["train"]), length)):
        train_info["loss"].append(stats_dict["train"][epoch]["loss"])
        valid_info["loss"].append(stats_dict["valid"][str(epoch)]["loss"])
        valid_info["bleu1_stoch"].append(stats_dict["valid"][str(epoch)]["stoch"]["bleu1"])
        valid_info["bleu4_stoch"].append(stats_dict["valid"][str(epoch)]["stoch"]["bleu4"])
        valid_info["bleu1_det"].append(stats_dict["valid"][str(epoch)]["det"]["bleu1"])
        valid_info["bleu4_det"].append(stats_dict["valid"][str(epoch)]["det"]["bleu4"])
        
        
    best_valid_info = {"bleu1_stoch": valid_info["bleu1_stoch"][best_epoch + 2], 
                       "bleu4_stoch": valid_info["bleu4_stoch"][best_epoch + 2],
                       "bleu1_det": valid_info["bleu1_det"][best_epoch + 2], 
                       "bleu4_det": valid_info["bleu4_det"][best_epoch + 2],
                       "loss": valid_info["loss"][best_epoch + 2]}
    
    test_info = {} 
    
    
    if "best" in stats_dict["valid"]:
        for val in stats_dict["valid"]["best"]:
            best_valid_info[val] = stats_dict["valid"]["best"][val]
            
    if "test" in stats_dict:
        if "best" in stats_dict["test"]:
            for val in stats_dict["test"]["best"]:
                test_info[val] = stats_dict["test"]["best"][val]
        if "perplexity" in stats_dict["test"]:
            test_info["perplexity"] = perplexity
        
    
    return {"train": train_info, "valid": valid_info, "best_valid": best_valid_info,
            "test": test_info}


def plot_bleu_scores(bleu1_stoch, bleu4_stoch, bleu1_det, bleu4_det, valid=True, split="valid", arch="base"):
    fig, axs = plt.subplots(1, 1, figsize=(12,6))

    X = np.arange(-1, len(bleu1_stoch) - 1)


    axs.set_title("BLEU1/4 over Epochs ({0}, {1})".format(split, arch))
    axs.set_xlabel("Epoch")
    axs.set_ylabel("BLEU")

    for i, bleu in enumerate([bleu1_stoch, bleu4_stoch, bleu1_det, bleu4_det]):
        axs.plot(X, [bleu1_stoch, bleu4_stoch, bleu1_det, bleu4_det][i], label=["bleu1_stoch", "bleu4_stoch", "bleu1_det",
                                                                                "bleu4_det"][i], color=colors[i])
        axs.legend(loc='best', fontsize='x-large')
        
    path_name = os.path.join(arch, "ious")
#     plt.savefig(path_name)

    plt.show()
    
def plot_loss(train_losses, valid_losses, arch="base"):
    fig, axs = plt.subplots(1, 1, figsize=(12,6))
    
    # Valid dataset has -1 epoch b/c we wanted to see stats before training
    X_valid = np.arange(-1, len(valid_losses) - 1)
    X_train = np.arange(len(train_losses))
    train_losses = np.array(train_losses)
    valid_losses = np.array(valid_losses)
        
    axs.set_title("Loss over Epochs ({0})".format(arch))
    axs.set_xlabel("Epoch")
    axs.set_ylabel("Loss")
    
    axs.plot(X_train, train_losses, label="Train", color=colors[0])
    axs.plot(X_valid, valid_losses, label="Valid", color=colors[1])
    
    axs.legend(loc='best', fontsize='x-large')
    axs.set_ylim([0, 1.1*max(np.max(train_losses), np.max(valid_losses))])
    
    path_name = os.path.join(arch, "loss")
    plt.savefig(path_name)
    plt.show()

    
def print_stats(arch):
    with open("{0}/stats.pkl".format(arch), "rb") as f:
        stats_dict = pkl.load(f)
    stats_dict = json.loads(json.dumps(stats_dict), object_hook=json_fix_keys)
    formatted_stats_dict = format_stats(stats_dict)
    val_info = formatted_stats_dict["valid"]
    plot_bleu_scores(val_info["bleu1_stoch"], val_info["bleu4_stoch"], val_info["bleu1_det"], val_info["bleu4_det"], arch=arch)
    plot_loss(formatted_stats_dict["train"]["loss"], formatted_stats_dict["valid"]["loss"], arch=arch)
    print(formatted_stats_dict["best_valid"])
    print(stats_dict["best_epoch"])
    
def format_caption(row, vocab, punctuation_set, token_set):
    
    caption_out = []
    
    for idx, word_num in enumerate(row):
        if (word_num not in token_set):

            word = vocab(word_num)
            
            word = " " + word
            
            # if prev word was start token, remove the space and capitalize the first letter
            if (idx > 0 and row[idx-1]==vocab("<start>")):
                word = word[1:]
                word = word.title()
            
            # if token is some punctuation, remove the space
            if (word_num in punctuation_set):
                word = word[1:]
            
            caption_out.append(word)
            
    caption_out = ''.join(caption_out)
        
    return caption_out
    
def caption_image(filename, punctuation_set, token_set, model, vocab, determinisitc=True, temperature=0.5):
    # get image
    im_show = Image.open(filename).convert("RGB")
    
    # reduce largest dimension to 512
    im_show.thumbnail((512,512), Image.NEAREST)
    
    # normalize image for resnet
    trans = [transforms.ToTensor(), 
             transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])]
    trans = transforms.Compose(trans)      
    im = trans(im_show)

    # add fourth dimension (expects batches)
    im = im.unsqueeze(0)

    # get outputs from model
    outputs = model(im.cuda(), deterministic=determinisitc, temperature=temperature)
    outputs = outputs.cpu().detach().numpy().argmax(axis=1)

    for idx, row in enumerate(outputs):
            caption = format_caption(row, vocab, punctuation_set, token_set)
            
    return im_show, caption


def plot_bar_stats_test(stats, title="stats for {0}-{1}-{2}", arch="base", temp="1.0"):
    title = title.format("test", arch, temp)
    perplexity = stats["perplexity"]
    stats = stats[temp]
    bar_names = [item for item in stats]
    print(bar_names)

    X = range(len(bar_names) + 1)
    Y = [perplexity] + [stats[item] for item in bar_names]
    bar_names = ["perplexity"] + bar_names
    labels = bar_names

    plt.figure(figsize=(10,10))
    barlist = plt.bar(X, Y, align="center")
    [barlist[i].set_color(colors[i]) for i in range(len(barlist))]
    plt.title(title, fontsize=20)
    plt.xlabel("Data", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.xticks(X, labels, rotation=45, fontsize=12)

    for i in range(len(barlist)):
        bar = barlist[i]
        yval = bar.get_height()
        plt.text(X[i], yval, "%.4f" % yval, fontsize=15, color='black', ha='center', va='bottom')

#     path_name = os.path.join(arch, "best_stats")
#     plt.savefig(path_name)
    plt.show()

def plot_bar_stats(stats, title="BLEU scores w/ various temperatures for {0}-{1}", arch="base", dataset="valid"):
    fig, ax = plt.subplots(figsize=(12, 8))
    labels = ["det"] + [str(temp) for temp in temps]
    ind = np.arange(len(labels))
    width = 0.35
    
    bleu4s = [stats["det"]["bleu4"]] + [stats[item]["bleu4"] for item in labels[1:]]
    bleu1s = [stats["det"]["bleu1"]] + [stats[item]["bleu1"] for item in labels[1:]]
    
    p1 = ax.bar(ind, bleu1s, width, bottom=0)
    p2 = ax.bar(ind + width, bleu4s, width, bottom=0)
    
    for i, (bar1, bar2) in enumerate(zip(p1, p2)):
        y1 = bar1.get_height()
        y2 = bar2.get_height()
        
        plt.text(ind[i], y1, "%.4f" % y1, fontsize=12, color='black', ha='center', va='bottom')
        plt.text(ind[i] + width, y2, "%.2f" % y2, fontsize=12, color='black', ha='center', va='bottom')
    
    plt.xlabel("Temperature")
    plt.ylabel("BLEU")
    
    ax.set_title(title.format(arch, dataset))
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(labels, rotation=45)

    ax.legend((p1[0], p2[0]), ("BLEU1", "BLEU4"))
    ax.autoscale_view()
    
    path_name = os.path.join(arch, "bar_stats_{0}".format(dataset))
    plt.savefig(path_name)

    plt.show()
    
def precision_recall_f1(y_pred, y_true):
    scores = {}
    
    # Micro
    scores['precision_micro'] = precision_score(y_true, y_pred, average='micro')
    scores['recall_micro']    = recall_score(y_true, y_pred, average='micro')
    scores['f1_micro']        = f1_score(y_true, y_pred, average='micro')
    print("test.precision_micro: ", scores['precision_micro']) 
    print("test.recall_micro: ", scores['recall_micro']) 
    print("test.f1_micro", scores['f1_micro']) 
    
    # Macro
    scores['precision_macro']    = precision_score(y_true, y_pred, average='macro')
    scores['recall_macro']       = recall_score(y_true, y_pred, average='macro')
    scores['f1_macro']           = f1_score(y_true, y_pred, average='macro')
    print("test.precision_macro: ", scores['precision_macro']) 
    print("test.recall_macro: ", scores['recall_macro']) 
    print("test.f1_macro: ", scores['f1_macro']) 
    
    # Weighted
    scores['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
    scores['recall_weighted']    = recall_score(y_true, y_pred, average='weighted')
    scores['f1_weighted']        = f1_score(y_true, y_pred, average='weighted')
    print("test.precision_weighted: ", scores['precision_weighted']) 
    print("test.recall_weighted: ", scores['recall_weighted']) 
    print("test.f1_weighted: ", scores['f1_weighted'])  
    
def class_report_conf_matrix(predictions, labels, experiment):
    id_to_label = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T', 'pad']
    labels = [id_to_label[i] for i in labels.tolist()]
    predictions = [id_to_label[i] for i in predictions.tolist()]
    cl_report = classification_report(labels, predictions)
    conf_mat = annot_confusion_matrix(labels, predictions)
    print(f"Classification Report:\n {cl_report}")
    print(f"Confusion Matrix:\n {conf_mat}")
    
    # Save test output
    df = pd.DataFrame()
    df['labels']      = labels
    df['predictions'] = predictions
    df.to_csv('stats/' + experiment + '/test_output.tsv', sep='\t', encoding='utf-8')
    print("Sample output saved in: stats/" + experiment + "/test_output.tsv")
    

def annot_confusion_matrix(valid_tags, pred_tags):

    """
    Create an annotated confusion matrix by adding label
    annotations and formatting to sklearn's `confusion_matrix`.
    """

    # Create header from unique tags
    header = sorted(list(set(valid_tags + pred_tags)))

    # Calculate the actual confusion matrix
    matrix = confusion_matrix(valid_tags, pred_tags, labels=header)

    # Final formatting touches for the string output
    mat_formatted = [header[i] + "\t" + str(row) for i, row in enumerate(matrix)]
    content = "\t" + " ".join(header) + "\n" + "\n".join(mat_formatted)

    return content


    