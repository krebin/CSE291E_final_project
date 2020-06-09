import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F
import numpy as np

class ResidualModel(nn.Module):

    def __init__(self, num_features=51, num_residual=5):
        
        self.num_features = num_features
        
        super().__init__()
        self.cnn_1d_3 = nn.Conv1d(in_channels=self.num_features, out_channels=100, stride=1, kernel_size=3, padding=1, bias=True)
        self.cnn_1d_5 = nn.Conv1d(in_channels=self.num_features, out_channels=100, stride=1, kernel_size=5, padding=2, bias=True)
        self.cnn_1d_1_1 = nn.Conv1d(in_channels=(700 + self.num_features), out_channels=500, stride=1, kernel_size=1, bias=True)
        self.cnn_1d_1_2 = nn.Conv1d(in_channels=1000, out_channels=500, stride=1, kernel_size=1, bias=True)
        self.num_residual = num_residual

        self.cnn_1d_3_filters = nn.ModuleList(
            [nn.Conv1d(in_channels=100, out_channels=100, stride=1, kernel_size=3, padding=1, bias=True).cuda() for _ in
             range(num_residual)])
        self.cnn_1d_3_gates = nn.ModuleList(
            [nn.Conv1d(in_channels=100, out_channels=100, stride=1, kernel_size=3, padding=1, bias=True).cuda() for _ in
             range(num_residual)])

        self.cnn_1d_3_bnorms = nn.ModuleList([nn.BatchNorm1d(100) for _ in range(num_residual)])


        self.cnn_1d_5_filters = nn.ModuleList(
            [nn.Conv1d(in_channels=100, out_channels=100, stride=1, kernel_size=5, padding=2, bias=True).cuda() for _ in
             range(num_residual)])
        self.cnn_1d_5_gates = nn.ModuleList(
            [nn.Conv1d(in_channels=100, out_channels=100, stride=1, kernel_size=5, padding=2, bias=True).cuda() for _ in
             range(num_residual)])

        self.cnn_1d_5_bnorms = nn.ModuleList([nn.BatchNorm1d(100) for _ in range(num_residual)])


        self.gru_1 = nn.GRU(input_size=(200 + self.num_features), hidden_size=250, num_layers=1, batch_first=True, bidirectional=True)
        self.gru_2 = nn.GRU(input_size=500, hidden_size=500, num_layers=1, batch_first=True, bidirectional=True)
        self.gru_3 = nn.GRU(input_size=500, hidden_size=500, num_layers=1, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(500, 128)
        self.fc2 = nn.Linear(128, 9)

        self.dropout = nn.Dropout(p=0.5)
        self.embedding = nn.Embedding(22, 22)
        
        self.bnorm1 = nn.BatchNorm1d((200+self.num_features))
        self.bnorm2 = nn.BatchNorm1d(500)
        self.bnorm3 = nn.BatchNorm1d(500)
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        def _init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)

        self.cnn_1d_3_filters.apply(_init_weights)
        self.cnn_1d_3_gates.apply(_init_weights)

        self.cnn_1d_5_filters.apply(_init_weights)
        self.cnn_1d_5_gates.apply(_init_weights)


    def forward(self, x, device, one_hot_embed):
        
        if one_hot_embed:
            # embed one hot
            one_hot = x[:, 0:22, :].argmax(axis=1)
            embedded = self.embedding(one_hot.long()).permute(0, 2, 1)
            x[:, 0:22, :] = embedded
            

        # Local Block
        local_block_3 = self.cnn_1d_3(x)
        local_block_5 = self.cnn_1d_5(x)
        
        local_block_3_sums = torch.zeros_like(local_block_3)
        local_block_3_sums += local_block_3

        for gate, filter, bnorm in zip(self.cnn_1d_3_gates, self.cnn_1d_3_filters, self.cnn_1d_3_bnorms):

            filtered = self.tanh(filter(local_block_3))
            gated = self.sigmoid(gate(local_block_3))

            out = bnorm(filtered * gated)
            local_block_3_sums = local_block_3_sums + out
            local_block_3 = local_block_3 + out

        local_block_5_sums = torch.zeros_like(local_block_5)
        local_block_5_sums += local_block_5

        for gate, filter, bnorm in zip(self.cnn_1d_5_gates, self.cnn_1d_5_filters, self.cnn_1d_5_bnorms):
            filtered = self.tanh(filter(local_block_5))
            gated = self.sigmoid(gate(local_block_5))

            out = bnorm(filtered * gated)
            local_block_5_sums = local_block_5_sums + out
            local_block_5 = local_block_5 + out

        local_block_3 = local_block_3_sums
        local_block_5 = local_block_5_sums
    
        x = torch.cat((x, local_block_3, local_block_5), dim=1)
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

