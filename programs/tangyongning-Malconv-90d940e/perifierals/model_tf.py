import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PETransformer(nn.Module):
    def __init__(self, input_dim, nhead, num_encoder_layers, dim_feedforward, num_classes, chunk_size):
        super(PETransformer, self).__init__()
        self.embedding = nn.Embedding(257, input_dim)  # Assuming byte values range from 1 to 256
        self.pos_encoder = PositionalEncoding(input_dim)
        encoder_layers = TransformerEncoderLayer(input_dim, nhead, dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        self.fc = nn.Linear(input_dim, num_classes)
        self.chunk_size = chunk_size

    def forward(self, src):
        batch_size, seq_len = src.size()
        if seq_len > self.chunk_size:
            src = src[:, :self.chunk_size]

        src = src.long()  # Ensure the input is of type LongTensor
        src = self.embedding(src) * torch.sqrt(torch.tensor(src.size(1), dtype=torch.float32, device=src.device))
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)
        output = self.fc(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
