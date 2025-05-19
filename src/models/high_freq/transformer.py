import torch
import torch.nn as nn
import math

class PosEnc(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        t = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2)*(-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(t*div)
        pe[:, 1::2] = torch.cos(t*div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]

class HighTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = cfg.d_model
        self.pre = nn.Linear(1, d_model)
        self.pos = PosEnc(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=cfg.heads,
            dim_feedforward=d_model*4,
            dropout=cfg.dropout,
            batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=cfg.layers)
        self.out = nn.Linear(d_model, d_model)  # 요약 출력

    def forward(self, x):
        # x: (B, T, 1)
        z = self.pre(x)
        z = self.pos(z)
        z = self.enc(z)
        return self.out(z).mean(dim=1)   # (B, d_model)
