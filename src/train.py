import torch, yaml, os
from omegaconf import OmegaConf
from utils.metrics import mae, rmse, smape
from utils.plot import plot_history
from datamodule import build_dataloader
from models.high_freq import build as build_high
from models.mid_freq  import build as build_mid
from models.low_freq  import build as build_low

def load_cfg():
    cfg = OmegaConf.merge(
        OmegaConf.load("config/run.yaml"),
        OmegaConf.load("config/data.yaml"),
        OmegaConf.load("config/model_high.yaml"),
        OmegaConf.load("config/model_mid.yaml"),
        OmegaConf.load("config/model_low.yaml"),
        OmegaConf.load("config/log.yaml"),
    )
    return cfg

def build_model(cfg):
    high = build_high(cfg.run.high_model, cfg.model_high.get(cfg.run.high_model))
    mid  = build_mid (cfg.run.mid_model , cfg.model_mid.get(cfg.run.mid_model))
    low  = build_low (cfg.run.low_model , cfg.model_low.get(cfg.run.low_model))
    # 간단히 직렬
    class Full(torch.nn.Module):
        def __init__(self, h, m, l):
            super().__init__()
            self.h=h
            self.m=m; self.l=l
        def forward(self, x):      # x (B, enc_len, 1)
            z = self.h(x[:,-cfg.data.win_nmin:, :])
            z = self.m(z.unsqueeze(-1))      # dummy 처리
            y = self.l(z)
            return y
    return Full(high, mid, low)

def main():
    cfg = load_cfg()
    torch.manual_seed(cfg.run.seed)
    loader = build_dataloader(cfg)

    model = build_model(cfg)
    device = cfg.run.device if torch.cuda.is_available() else "cpu"
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.run.lr)
    history = {k: [] for k in ["train_MAE","val_MAE"]}

    for epoch in range(cfg.run.epochs):
        model.train(); loss_sum=0
        for x, y in loader:
            x, y = x.unsqueeze(-1).to(device), y.to(device)
            opt.zero_grad()
            y_hat = model(x)
            loss = torch.mean(torch.abs(y_hat - y))
            loss.backward(); opt.step()
            loss_sum += loss.item()
        history["train_MAE"].append(loss_sum/len(loader))
        history["val_MAE"].append(history["train_MAE"][-1])  # demo

        print(f"Epoch {epoch:02d} MAE {history['train_MAE'][-1]:.3f}")
    os.makedirs(cfg.log.fig_dir, exist_ok=True)
    plot_history(history, ["MAE"], cfg.log.fig_dir)

if __name__ == "__main__":
    main()
