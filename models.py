"""
Battery RUL Prediction DNN Models
"""
import math
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, sequence_length, hidden_size, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        self.activation = nn.ReLU()
        self.loss_function = nn.MSELoss()

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        x = self.activation(self.fc2(x))
        x = self.fc3(x)

        x = x.view(-1, self.sequence_length)
        x = torch.mean(x, axis=1, keepdim=True)
        return x

    def loss(self, x, y):
        y_hat = self.forward(x)
        loss = self.loss_function(y, y_hat)
        return loss
    

class RNN_GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(RNN_GRU, self).__init__()
        self.rnn = torch.nn.GRU(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.ReLU = nn.ReLU()
        self.fc1 = torch.nn.Linear(hidden_dim, 20, bias=True)
        self.fc2 = torch.nn.Linear(20, 30, bias=True)
        self.fc3 = torch.nn.Linear(30, output_dim, bias=True)
        self.loss_function = torch.nn.MSELoss()

    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc1(x[:,-1])
        x = self.ReLU(x)
        x = self.fc2(x)
        x = self.ReLU(x)
        x = self.fc3(x)
        return x

    def loss(self, x, y):
        y_hat = self.forward(x)
        return self.loss_function(y, y_hat)


class RNN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(RNN_LSTM, self).__init__()
        self.rnn = torch.nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.ReLU = nn.ReLU()
        self.fc1 = torch.nn.Linear(hidden_dim, 20, bias=True)
        self.fc2 = torch.nn.Linear(20, 30, bias=True)
        self.fc3 = torch.nn.Linear(30, output_dim, bias=True)
        self.loss_function = torch.nn.MSELoss()

    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc1(x[:,-1])
        x = self.ReLU(x)
        x = self.fc2(x)
        x = self.ReLU(x)
        x = self.fc3(x)
        return x

    def loss(self, x, y):
        y_hat = self.forward(x)
        return self.loss_function(y, y_hat)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, nhead, dropout=0.5):
        super(TransformerModel, self).__init__()

        self.input_dim = input_dim
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout,
                                                        batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.init_weights()
        self.loss_function = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.bias.data.zero_()
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output[:, -1, :]

    def loss(self, x, y):
        y_hat = self.forward(x)
        mse_loss = self.loss_function(y, y_hat)

        return mse_loss