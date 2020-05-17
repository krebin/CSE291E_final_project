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

        self.gru_f_1 = nn.GRU(input_size=251, hidden_size=250, num_layers=1, batch_first=True)
        self.gru_b_1 = nn.GRU(input_size=251, hidden_size=250, num_layers=1, batch_first=True)

        self.gru_f_2 = nn.GRU(input_size=500, hidden_size=500, num_layers=1, batch_first=True)
        self.gru_b_2 = nn.GRU(input_size=500, hidden_size=500, num_layers=1, batch_first=True)

        self.gru_f_3 = nn.GRU(input_size=500, hidden_size=500, num_layers=1, batch_first=True)
        self.gru_b_3 = nn.GRU(input_size=500, hidden_size=500, num_layers=1, batch_first=True)

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
        # x = self.bnorm1(x)
        x = x.permute(0, 2, 1)

        # BGRU
        T = x.shape[1]
        h_t_f = torch.zeros(1, x.shape[0], 250).to(device)
        h_t_b = torch.zeros(1, x.shape[0], 250).to(device)

        h_f = []
        h_b = []

        for t in range(T):
            input_t_f = x[:, t, :].unsqueeze(1)
            input_t_b = x[:, T - (t + 1), :].unsqueeze(1)

            _, h_t_f = self.gru_f_1(input_t_f, h_t_f)
            _, h_t_b = self.gru_b_1(input_t_b, h_t_b)

            h_f.append(h_t_f)
            h_b.append(h_t_b)


        F = torch.stack(h_f, dim=2).squeeze(0)
        B = torch.stack(h_b, dim=2).squeeze(0)      
        O1 = torch.cat((F, B), dim=2)


        # BGRU Block 1
        x = torch.cat((x, O1), dim=2)
        x = nn.functional.relu(self.cnn_1d_1_1(x.permute(0,2,1)))
        x = self.dropout(x)
        # x = self.bnorm2(x)

        h_t_f = torch.zeros(1, x.shape[0], 500).to(device)
        h_t_b = torch.zeros(1, x.shape[0], 500).to(device)

        h_f = []
        h_b = []

        x = x.permute([0, 2, 1])
        T = x.shape[1]
        for t in range(T):
            input_t_f = x[:, t, :].unsqueeze(1)
            input_t_b = x[:, T - (t + 1), :].unsqueeze(1)


            _, h_t_f = self.gru_f_2(input_t_f, h_t_f)
            _, h_t_b = self.gru_b_2(input_t_b, h_t_b)

            h_f.append(h_t_f)
            h_b.append(h_t_b)

        F = torch.stack(h_f, dim=2).squeeze(0)
        B = torch.stack(h_b, dim=2).squeeze(0)   
        O2 = (F + B)

        # BGRU Block 2
        x = torch.cat((O1, O2), dim=2).permute(0, 2, 1)
        x = nn.functional.relu(self.cnn_1d_1_2(x))
        x = self.dropout(x)
        # x = self.bnorm3(x)

        h_t_f = torch.zeros(1, x.shape[0], 500).to(device)
        h_t_b = torch.zeros(1, x.shape[0], 500).to(device)

        h_f = []
        h_b = []

        x = x.permute([0, 2, 1])
        T = x.shape[1]
        for t in range(T):
            input_t_f = x[:, t, :].unsqueeze(1)
            input_t_b = x[:, T - (t + 1), :].unsqueeze(1)

            _, h_t_f = self.gru_f_3(input_t_f, h_t_f)
            _, h_t_b = self.gru_b_3(input_t_b, h_t_b)

            h_f.append(h_t_f)
            h_b.append(h_t_b)


        F = torch.stack(h_f, dim=2).squeeze(0)
        B = torch.stack(h_b, dim=2).squeeze(0)   
        x = (F + B)
    
        x = self.dropout(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)

        return x

