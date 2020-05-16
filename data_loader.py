import torch
import torchvision.transforms as transforms
import torch.functional as F
import torch.utils.data as data
import os
import pickle as pkl
import numpy as np
from PIL import Image


class ProteinDataset(data.Dataset):
    def __init__(self, protein_data, ids):

        protein_lengths = 0
        for i, id in enumerate(ids):
            d = protein_data[id]
            protein_lengths += d["protein_length"]

        all_encodings = np.zeros([protein_lengths, 50])
        all_labels = np.zeros([protein_lengths, 8])

        total_length = 0
        for i, id in enumerate(ids):
            if i % 250 == 0:
                print("Stacking {0}/{1} proteins".format(i, len(ids)))

            d = protein_data[id]
            protein_length = d["protein_length"]

            all_encodings[total_length:total_length + protein_length] = d["protein_encoding"]
            all_labels[total_length:total_length + protein_length] = d["secondary_structure_onehot"]

            total_length += protein_length
        
        self.all_encodings = all_encodings.astype(np.float32)
        self.all_labels = all_labels.astype(np.int32)

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        encoding = self.all_encodings[index]
        label = self.all_labels[index]
        
        return encoding, label

    def __len__(self):
        return len(self.all_encodings)


def get_loader(protein_data, ids, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader"""
    protein = ProteinDataset(protein_data, ids)

    # def collate_fn(data):
    #     return data

    data_loader = torch.utils.data.DataLoader(dataset=protein, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,)
                                              #collate_fn=collate_fn)
    return data_loader, len(protein)