import torch
import torch.nn as nn

import torch.nn.functional as F

class MalConv(nn.Module):
    def __init__(self, input_length=2000000, window_size=500, stride = 500, embed = 8):
        super(MalConv, self).__init__()

        self.embed = nn.Embedding(257, embed, padding_idx=0)
        self.conv_1 = nn.Conv1d(4, 128, window_size, stride = stride, bias = True)
        self.conv_2 = nn.Conv1d(4, 128, window_size, stride = stride, bias = True)
        self.pooling = nn.MaxPool1d(int(input_length / window_size))
        self.fc_1 = nn.Linear(128, 128)
        self.fc_2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.long()
        x = self.embed(x)
        x = torch.transpose(x, -1, -2)
        cnn_value = self.conv_1(x.narrow(-2, 0, 4))
        gating_weight = self.sigmoid(self.conv_2(x.narrow(-2, 4, 4)))
        x = cnn_value * gating_weight
        x = self.pooling(x)
        x = x.view(-1, 128)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return x

class MalLSTM(nn.Module):
    def __init__(self, input_length=2000000, window_size=500, stride = 500, embed = 8):
        super(MalLSTM, self).__init__()

        self.embed = nn.Embedding(257, embed, padding_idx=0)
        self.conv_1 = nn.Conv1d(4, 128, window_size, stride = stride, bias = True)
        self.conv_2 = nn.Conv1d(4, 128, window_size, stride = stride, bias = True)
        self.pooling = nn.MaxPool1d(int(input_length / window_size))
        self.lstm = nn.LSTM(input_size = 128, hidden_size = 128, num_layers = 1)
        self.fc_1 = nn.Linear(128, 128)
        self.fc_2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.long()
        x = self.embed(x)
        x = torch.transpose(x, -1, -2)
        cnn_value = self.conv_1(x.narrow(-2, 0, 4))
        gating_weight = self.sigmoid(self.conv_2(x.narrow(-2, 4, 4)))
        x = cnn_value * gating_weight
        x = self.pooling(x)
        x = x.view(-1, 1, 128)
        x, hidden = self.lstm(x)
        x = x.view(-1, 128)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return x


#WIP
class MalTF(nn.Module):
    def __init__(self, input_length=2000000, window_size=256, stride = 256, embed = 4, headers = 4, layers = 2, ff_dim = 256):
        super(MalTF, self).__init__()

        self.embed = nn.Embedding(257, embed, padding_idx=0)
        self.conv_1 = nn.Conv1d(4, 128, window_size, stride=stride, bias=True)
        self.conv_2 = nn.Conv1d(4, 128, window_size, stride=stride, bias=True)
        self.pooling = nn.MaxPool1d(int(input_length / window_size))

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model = 128, nhead = headers, dim_feedforward = ff_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers = layers)

        self.fc_1 = nn.Linear(128, 128)
        self.fc_2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.long()
        x = self.embed(x)
        x = torch.transpose(x, -1, -2)

        cnn_value = self.conv_1(x.narrow(-2, 0, 4))
        gating_weight = self.sigmoid(self.conv_2(x.narrow(-2, 4, 4)))
        x = cnn_value * gating_weight
        x = self.pooling(x)

        x = x.transpose(1,2)
        x = self.transformer_encoder(x)


        x = x.view(-1, 128)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return x

