import torch
import torch.nn as nn
from cbp_conv import CBPConv
from cbp_linear import CBPLinear

class MalConv(nn.Module):
    def __init__(self, input_length=2000000, window_size=500, replacement_rate=0, maturity_threshold=100, init='default'):
        super(MalConv, self).__init__()

        self.embed = nn.Embedding(257, 8, padding_idx=0)
        
        # 使用CBP卷积层替换标准卷积层
        self.conv_1 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)
        self.conv_2 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)
        
        # 初始化CBP卷积层
        self.cbp_conv1 = CBPConv(in_layer=self.conv_1, out_layer=self.conv_2, replacement_rate=replacement_rate, maturity_threshold=maturity_threshold, init=init)
        
        self.pooling = nn.MaxPool1d(int(input_length / window_size))
        
        # 使用CBP全连接层替换标准全连接层
        self.fc_1 = nn.Linear(128, 128)
        self.fc_2 = nn.Linear(128, 1)
        
        # 初始化CBP全连接层
        self.cbp_fc1 = CBPLinear(in_layer=self.fc_1, out_layer=self.fc_2, replacement_rate=replacement_rate, maturity_threshold=maturity_threshold, init=init)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.long()
        x = self.embed(x)
        x = torch.transpose(x, -1, -2)
        
        # 通过CBP卷积层进行前向传播
        cnn_value = self.cbp_conv1(self.conv_1(x.narrow(-2, 0, 4)))
        gating_weight = self.sigmoid(self.conv_2(x.narrow(-2, 4, 4)))
        x = cnn_value * gating_weight
        
        x = self.pooling(x)
        x = x.view(-1, 128)
        
        # 通过CBP全连接层进行前向传播
        x = self.cbp_fc1(self.fc_1(x))
        x = self.fc_2(x)
        
        return x
