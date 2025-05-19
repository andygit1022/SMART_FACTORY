import torch
import torch.nn as nn

class ConvLSTM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cks, filters = cfg.cnn_kernel, cfg.cnn_filters
        cnn_layers = []
        in_ch = 1
        for f in filters:
            cnn_layers += [nn.Conv1d(in_ch, f, cks, padding=cks//2),
                           nn.BatchNorm1d(f),
                           nn.ReLU()]
            in_ch = f
        self.cnn = nn.Sequential(*cnn_layers)
        self.lstm = nn.LSTM(input_size=filters[-1],
                            hidden_size=cfg.lstm_hidden,
                            num_layers=cfg.lstm_layers,
                            batch_first=True, dropout=cfg.dropout)
    def forward(self, x):      # x: (B, T, 1)
        z = self.cnn(x.transpose(1,2)).transpose(1,2)
        out, _ = self.lstm(z)
        return out[:,-1]       # (B, hidden)
