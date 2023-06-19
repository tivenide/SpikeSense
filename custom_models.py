import math
import torch
from torch import Tensor
import torch.nn as nn
class DenseModel(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = True,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dtype=None) -> None:
        super(DenseModel, self).__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.bias = bias
        self.device = device

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, out_features)
        self.selu = nn.SELU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_tensor: Tensor) -> Tensor:
        # flatten_tensor = self.flatten(input_tensor)
        fc1_out = self.selu(self.fc1(input_tensor))
        fc2_out = self.selu(self.fc2(fc1_out))
        fc3_out = self.selu(self.fc3(fc2_out))
        return self.softmax(fc3_out)

    def get_model_metadata(self):
        return {
            'model_type': 'DenseModel',
            'in_features': self.in_features,
            'hidden_features': self.hidden_features,
            'out_features': self.out_features
        }


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 200): # max_len equals maximum window size
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_classes: int, num_layers: int, num_heads: int,
                 dropout: float, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(TransformerModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device

        self.embedding = nn.Linear(input_dim, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        encoder_layer = nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward=hidden_size*4, dropout=dropout, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.selu = nn.SELU()

    def forward(self, src: Tensor) -> Tensor:
        batch_size, seq_len = src.shape[0], src.shape[1]
        src = src.view(batch_size, seq_len, self.input_dim)
        src = src.transpose(0, 1).contiguous()
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=0)
        #output = self.dense(output)
        output = self.fc(output)
        output = self.softmax(output)
        #output = self.sigmoid(output)
        #pred = (output >= 0.9).float()
        return output

    def get_model_metadata(self):
        return {
            'model_type': 'TransformerModel',
            'input_dim': self.input_dim,
            'hidden_size': self.hidden_size,
            'num_classes': self.num_classes,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'dropout': self.dropout
        }
