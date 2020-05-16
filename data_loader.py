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
        
        all_encodings = np.array([], dtype=np.float32).reshape(0, 50)
        all_labels = np.array([], dtype=np.int32).reshape(0, 8)
        
        for id in ids:
            d = protein_data[id]
            all_encodings = np.vstack((all_encodings, d["protein_encoding"]))
            all_labels =  np.vstack((all_labels, d["secondary_structure_onehot"]))
        
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