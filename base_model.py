import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F
import numpy as np

class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.cnn_1d_3 = nn.Conv1d(in_channels=51, out_channels=100, stride=1, kernel_size=3, padding=1, bias=True)
        self.cnn_1d_5 = nn.Conv1d(in_channels=51, out_channels=100, stride=1, kernel_size=5, padding=2, bias=True)
        self.cnn_1d_1_1 = nn.Conv1d(in_channels=751, out_channels=500, stride=1, kernel_size=1, bias=True)
        self.cnn_1d_1_2 = nn.Conv1d(in_channels=1000, out_channels=500, stride=1, kernel_size=1, bias=True)

        self.gru_1 = nn.GRU(input_size=251, hidden_size=250, num_layers=1, batch_first=True, bidirectional=True)
        self.gru_2 = nn.GRU(input_size=500, hidden_size=500, num_layers=1, batch_first=True, bidirectional=True)
        self.gru_3 = nn.GRU(input_size=500, hidden_size=500, num_layers=1, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(500, 128)
        self.fc2 = nn.Linear(128, 9)

        self.dropout = nn.Dropout(p=0.5)
        self.embedding = nn.Embedding(22, 22)
        
        self.bnorm1 = nn.BatchNorm1d(251)
        self.bnorm2 = nn.BatchNorm1d(500)
        self.bnorm3 = nn.BatchNorm1d(500)


    def forward(self, x, device):
        # embed one hot
        one_hot = x[:, 0:22, :].argmax(axis=1)
        embedded = self.embedding(one_hot.long()).permute(0, 2, 1)
        x[:, 0:22, :] = embedded

        # Local Block
        local_block_3 = self.cnn_1d_3(x)
        local_block_5 = self.cnn_1d_5(x)
        x = nn.functional.relu(torch.cat((x, local_block_3, local_block_5), dim=1))
        x = self.dropout(x)
        x = x.permute(0, 2, 1)

        # BGRU
        h_t = torch.zeros(2, x.shape[0], 250).to(device)
        O1, _ = self.gru_1(x, h_t)

        # BGRU Block 1
        x = torch.cat((x, O1), dim=2)
        x = nn.functional.relu(self.cnn_1d_1_1(x.permute(0, 2, 1)))
        x = x.permute(0, 2, 1)

        h_t = torch.zeros(2, x.shape[0], 500).to(device)

        O2, _ = self.gru_2(x, h_t)
        O2 = O2[:, :, :500] + O2[:, :, 500:]

        # BGRU Block 2
        x = torch.cat((O1, O2), dim=2).permute(0, 2, 1)
        x = nn.functional.relu(self.cnn_1d_1_2(x))
        x = self.dropout(x)
        
        h_t = torch.zeros(2, x.shape[0], 500).to(device)
        x = x.permute([0, 2, 1])
        x, _ = self.gru_3(x, h_t)
        x = x[:, :, :500] + x[:, :, 500:]
    
        x = self.dropout(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)

        return x

