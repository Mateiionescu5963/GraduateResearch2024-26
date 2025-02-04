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




















# @Author : bamtercelboo
# @Datetime : 2018/07/19 22:35
# @File : model.py
# @Last Modify Time : 12/04/24 by mliones
# @Contact : bamtercelboo@{gmail.com, 163.com}
"""
    Neural Network: CNN_LSTM
    Detail: the input crosss cnn model and LSTM model independly, then the result of both concat
"""
class CNN_LSTM(nn.Module):
    def __init__(self, embed_dim, kernel_sizes = [1], lstm_hidden_dim = 128, lstm_num_layers = 1, embed_num = 257, class_num=1, kernel_num=128, paddingId = 0, word_Embedding = False, pretrained_weight = None, dropout = 0.04, device = torch.device('cpu')):
        super(CNN_LSTM, self).__init__()
        self.hidden_dim = lstm_hidden_dim
        self.num_layers = lstm_num_layers
        V = embed_num
        D = embed_dim
        C = class_num
        Ci = 1
        Co = kernel_num
        Ks = kernel_sizes
        self.C = C
        self.embed = nn.Embedding(V, D, padding_idx=paddingId)
        # pretrained  embedding
        if word_Embedding:
            self.embed.weight.data.copy_(pretrained_weight)

        # CNN
        self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.dropout = nn.Dropout(dropout)
        # for cnn cuda
        for conv in self.convs1:
            conv.to(device)

        # LSTM
        self.lstm = nn.LSTM(D, self.hidden_dim, dropout=dropout, num_layers=self.num_layers)

        # linear
        L = len(Ks) * Co + self.hidden_dim
        self.hidden2label1 = nn.Linear(L, L // 2)
        self.hidden2label2 = nn.Linear(L // 2, C)

    def forward(self, x):
        device = next(self.parameters()).device
        x = x.to(device)

        embed = self.embed(x)

        # CNN
        cnn_x = embed
        cnn_x = torch.transpose(cnn_x, 0, 1)
        cnn_x = cnn_x.unsqueeze(1)
        cnn_x = [F.relu(conv(cnn_x)).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
        cnn_x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in cnn_x]  # [(N,Co), ...]*len(Ks)
        cnn_x = torch.cat(cnn_x, 1)
        cnn_x = self.dropout(cnn_x)

        # LSTM
        lstm_x = embed.view(len(x), embed.size(1), -1)
        lstm_out, _ = self.lstm(lstm_x)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)

        # CNN and LSTM cat
        cnn_x = torch.transpose(cnn_x, 0, 1)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        cnn_lstm_out = torch.cat((cnn_x, lstm_out), 0)
        cnn_lstm_out = torch.transpose(cnn_lstm_out, 0, 1)

        # linear
        cnn_lstm_out = self.hidden2label1(F.tanh(cnn_lstm_out))
        cnn_lstm_out = self.hidden2label2(F.tanh(cnn_lstm_out))

        # output
        logit = cnn_lstm_out
        return logit

