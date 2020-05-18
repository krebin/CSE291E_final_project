import torch
import torchvision.transforms as transforms
import torch.functional as F
import torch.utils.data as data
import os
import pickle as pkl
import numpy as np
from PIL import Image


class ProteinDataset(data.Dataset):
    def __init__(self, protein_data, ids, size=35700):

        data_len = len(ids)

        all_encodings = np.zeros([data_len, size])
        all_labels = np.zeros([data_len, 6300])
        all_lengths = []

        for i, id in enumerate(ids):
            id = str(id)
            if i % 250 == 0:
                print("Loading {0}/{1} proteins".format(i, len(ids)))

            d = protein_data[id]
            protein_length = d["protein_length"]
            all_lengths.append(protein_length)

            all_encodings[i, :] = d["protein_encoding"]
            all_labels[i, :] = d["secondary_structure_onehot"]
        
        self.all_encodings = all_encodings.astype(np.float32)
        self.all_labels = all_labels.astype(np.int32)
        self.all_lengths = np.array(all_lengths).astype(np.int32)

        print(len(all_encodings), len(all_labels), len(all_lengths))

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        encoding = self.all_encodings[index]
        label = self.all_labels[index]
        length = self.all_lengths[index]
        
        return encoding, label, length

    def __len__(self):
        return len(self.all_encodings)


def get_loader(protein_data, ids, batch_size, shuffle, num_workers, prot_vec):
    """Returns torch.utils.data.DataLoader"""
    
    if (prot_vec):
        protein = ProteinDataset(protein_data, ids, size=70000)
    else:
        protein = ProteinDataset(protein_data, ids)

    # def collate_fn(data):
    #     return data

    data_loader = torch.utils.data.DataLoader(dataset=protein, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,)
                                              #collate_fn=collate_fn)
    return data_loader, len(protein)