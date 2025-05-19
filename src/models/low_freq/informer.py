import torch
import torch.nn as nn

class InformerDecoder(nn.Module):
    """매우 축약한 Dummy Informer (실제 논문 구현과 다름)"""
    def __init__(self, cfg, input_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.pred_len))
    def forward(self, h):     # h: (B, input_dim)
        return self.fc(h)     # (B, 672)
