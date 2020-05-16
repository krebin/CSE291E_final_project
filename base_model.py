import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F
import numpy as np

class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.cnn_1d_3 = nn.Conv1d(in_channels=50, out_channels=100, stride=1, kernel_size=3, padding=1, bias=True)
        self.cnn_1d_5 = nn.Conv1d(in_channels=50, out_channels=100, stride=1, kernel_size=5, padding=2, bias=True)
        self.cnn_1d_1_1 = nn.Conv1d(in_channels=750, out_channels=500, stride=1, kernel_size=1, bias=True)
        self.cnn_1d_1_2 = nn.Conv1d(in_channels=1000, out_channels=500, stride=1, kernel_size=1, bias=True)

        self.gru_f_1 = nn.GRU(input_size=1, hidden_size=1, num_layers=1, batch_first=True)
        self.gru_b_1 = nn.GRU(input_size=1, hidden_size=1, num_layers=1, batch_first=True)

        self.gru_f_2 = nn.GRU(input_size=1, hidden_size=1, num_layers=1, batch_first=True)
        self.gru_b_2 = nn.GRU(input_size=1, hidden_size=1, num_layers=1, batch_first=True)

        self.gru_f_3 = nn.GRU(input_size=1, hidden_size=1, num_layers=1, batch_first=True)
        self.gru_b_3 = nn.GRU(input_size=1, hidden_size=1, num_layers=1, batch_first=True)

        self.fc1 = nn.Linear(500, 1024)
        self.fc2 = nn.Linear(1024, 8)

        self.dropout = nn.Dropout(p=0.5)
        self.embedding = nn.Embedding(21, 21)


    def forward(self, x):
        # embed one hot
        one_hot = x[:, 0:21, 0].argmax(axis=1)
        embedded = self.embedding(one_hot.long()).unsqueeze(2)
        x[:, 0:21, :] = embedded

        # Local Block
        local_block_3 = self.cnn_1d_3(x)
        local_block_5 = self.cnn_1d_5(x)
        x = nn.functional.relu(torch.cat((x, local_block_3, local_block_5), dim=1))

        # BGRU
        T = x.shape[1]
        h_t_f = torch.zeros(1, x.shape[0], 1).cuda()
        h_t_b = torch.zeros(1, x.shape[0], 1).cuda()

        h_f = []
        h_b = []


        for t in range(T):
            input_t_f = x[:, t, :].unsqueeze(2)
            input_t_b = x[:, t, :].unsqueeze(2)


            _, h_t_f = self.gru_f_1(input_t_f, h_t_f)
            _, h_t_b = self.gru_b_1(input_t_b, h_t_b)

            h_f.append(h_t_f)
            h_b.append(h_t_b)


        F = torch.cat(h_f, dim=2)
        B = torch.cat(h_b, dim=2)
        O1 = torch.cat((F, B), dim=2).view([x.shape[0], -1, 1])


        # BGRU Block 1
        x = torch.cat((x, O1), dim=1)
        x = nn.functional.relu(self.cnn_1d_1_1(x))
        x = self.dropout(x)

        h_t_f = torch.zeros(1, x.shape[0], 1).cuda()
        h_t_b = torch.zeros(1, x.shape[0], 1).cuda()

        h_f = []
        h_b = []

        T = x.shape[1]
        for t in range(T):
            input_t_f = x[:, t, :].unsqueeze(2)
            input_t_b = x[:, t, :].unsqueeze(2)


            _, h_t_f = self.gru_f_2(input_t_f, h_t_f)
            _, h_t_b = self.gru_b_2(input_t_b, h_t_b)

            h_f.append(h_t_f)
            h_b.append(h_t_b)

        F = torch.cat(h_f, dim=2)
        B = torch.cat(h_b, dim=2)
        O2 = (F + B).view([x.shape[0], -1, 1])

        # BGRU Block 2
        x = torch.cat((x, O2), dim=1)
        x = nn.functional.relu(self.cnn_1d_1_2(x))
        x = self.dropout(x)

        h_t_f = torch.zeros(1, x.shape[0], 1).cuda()
        h_t_b = torch.zeros(1, x.shape[0], 1).cuda()

        h_f = []
        h_b = []

        T = x.shape[1]
        for t in range(T):
            input_t_f = x[:, t, :].unsqueeze(2)
            input_t_b = x[:, t, :].unsqueeze(2)

            _, h_t_f = self.gru_f_2(input_t_f, h_t_f)
            _, h_t_b = self.gru_b_2(input_t_b, h_t_b)

            h_f.append(h_t_f)
            h_b.append(h_t_b)

        F = torch.cat(h_f, dim=2)
        B = torch.cat(h_b, dim=2)
        x = (F + B).view([x.shape[0], -1])
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)

        return x

