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
        
        self.gru_bi_1 = nn.GRU(input_size=251, hidden_size=250, num_layers=1, batch_first=True,bidirectional=True)
        self.gru_bi_2 = nn.GRU(input_size=500, hidden_size=500, num_layers=1, batch_first=True,bidirectional=True)
        self.gru_bi_3 = nn.GRU(input_size=500, hidden_size=500, num_layers=1, batch_first=True,bidirectional=True)

        self.fc1 = nn.Linear(500, 1024)
        self.fc2 = nn.Linear(1024, 9)

        self.dropout = nn.Dropout(p=0.5)
        self.embedding = nn.Embedding(22, 22)


    def forward(self, x, device):
        # embed one hot
        one_hot = x[:, 0:22, :].argmax(axis=1)
        embedded = self.embedding(one_hot.long()).permute(0, 2, 1)
        x[:, 0:22, :] = embedded # ! x: bs x 51 x 700

        # Local Block
        local_block_3 = self.cnn_1d_3(x) # ! local_block_3: bs x 100 x 700
        local_block_5 = self.cnn_1d_5(x) # ! local_block_5: bs x 100 x 700
        x = nn.functional.relu(torch.cat((x, local_block_3, local_block_5), dim=1)) 
        x = x.permute(0, 2, 1) # *** x  here is concat'd with O1 for BGRU block #1 
        # ! x: bs x 700 x 251

        # BGRU
        # !!! TRY BI_DIR GRU
        O1, _ = self.gru_bi_1(x) # *** O1 here is concat'd with x for BGRU block #1, and O2 for BGRU block #2
        # T = x.shape[1] # ! T size = 700
        # h_t_f = torch.zeros(1, x.shape[0], 250).to(device) # ! 1 x bs x 250
        # h_t_b = torch.zeros(1, x.shape[0], 250).to(device) # ! 1 x bs x 250

        # h_f = []
        # h_b = []

        # for t in range(T): # ! T size = 700
        #     input_t_f = x[:, t, :].unsqueeze(1) # ! input_t_f: bs x 1 x 251
        #     input_t_b = x[:, T - (t + 1), :].unsqueeze(1) # ! input_t_b: bs x 1 x 251

        #     _, h_t_f = self.gru_f_1(input_t_f, h_t_f)  # ! 1 x bs x 250
        #     _, h_t_b = self.gru_b_1(input_t_b, h_t_b)  # ! 1 x bs x 250

        #     h_f.append(h_t_f)
        #     h_b.append(h_t_b)


        # F = torch.cat(h_f, dim=2) # ! hf len = 700, F = 1 x bs x 175000    --> this is 700 * 250
        # B = torch.cat(h_b, dim=2) # ! hb len = 700, B = 1 x bs x 175000    --> this is 700 * 250
        # O1 = torch.cat((F, B), dim=2) #! 01 = 1 x bs x 350000     --> this is 175000 + 175000
        # O1 = O1.view([x.shape[0], 700, -1]) # *** O1 here is concat'd with x for BGRU block #1, and O2 for BGRU block #2
        # ! O1 size: 3 x 700 x 500

        # BGRU Block 1
        x = torch.cat((x, O1), dim=2) #! x: 3, 700, 751
        # x = nn.functional.relu(self.cnn_1d_1_1(x.view([-1, 751, 700]))) # ? ReLU here, too?
        x = nn.functional.relu(self.cnn_1d_1_1(x.permute(0,2,1))) # ? ReLU here, too?
        x = self.dropout(x)
        # ! x size: 3 x 500 x 700
        
        O2, _ = self.gru_bi_2(x.permute(0,2,1))

        # h_t_f = torch.zeros(1, x.shape[0], 500).to(device)
        # h_t_b = torch.zeros(1, x.shape[0], 500).to(device)

        # h_f = []
        # h_b = []

        # x = x.permute([0, 2, 1])
        # T = x.shape[1]
        # for t in range(T):
        #     input_t_f = x[:, t, :].unsqueeze(1)
        #     input_t_b = x[:, T - (t + 1), :].unsqueeze(1)


        #     _, h_t_f = self.gru_f_2(input_t_f, h_t_f)
        #     _, h_t_b = self.gru_b_2(input_t_b, h_t_b)

        #     h_f.append(h_t_f)
        #     h_b.append(h_t_b)

        # F = torch.cat(h_f, dim=2)
        # B = torch.cat(h_b, dim=2)
        # O2 = (F + B).view([x.shape[0], 700, -1])
        # ! O2 size: 3 x 700 x 500
        
        # BGRU Block 2
        x = torch.cat((O1, O2), dim=2).permute(0, 2, 1)
        x = nn.functional.relu(self.cnn_1d_1_2(x))
        x = self.dropout(x)

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

        F = torch.cat(h_f, dim=2)
        B = torch.cat(h_b, dim=2)
        x = (F + B).view([x.shape[0], 700, 500])
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)

        return x

