# src/datamodule.py
import polars as pl
import numpy as np
import torch, math
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig
from datetime import datetime, timedelta


# ── 1. 시퀀스 Dataset ─────────────────────────────────────────
class HourlyDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):  return len(self.x)
    def __getitem__(self, idx):  return self.x[idx], self.y[idx]

# ── 2. 5 s → n min → 1 h  리샘플 ─────────────────────────────
def resample_to_hour(df: pl.DataFrame, n_min: int):
    # ①  5 s → n min (mean)
    df_n = df.groupby_dynamic("ts", every=f"{n_min}m").agg(pl.mean("activePower"))
    # ②  n min → 1 h (sum  = kWh)
    df_h = df_n.groupby_dynamic("ts", every="1h").agg(pl.sum("activePower").alias("kWh"))
    return df_h

# ── 3. DataLoader 빌더 (demo: 인공 데이터) ───────────────────
def build_dataloader(cfg: DictConfig):
    enc, pred = cfg.data.enc_len_hours, cfg.data.pred_len_hours
    # =====  DEMO  =================================================
    hours = 24*50                                                   # 50 일
    ts = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(hours)]
    kWh = 50 + 10*np.sin(np.arange(hours)/24*2*math.pi)
    df_h = pl.DataFrame({"ts": ts, "kWh": kWh})
    # ==============================================================

    xs, ys = [], []
    for i in range(0, hours - enc - pred):
        xs.append(kWh[i:i+enc])
        ys.append(kWh[i+enc:i+enc+pred])
    xs, ys = np.stack(xs), np.stack(ys)
    return DataLoader(HourlyDataset(xs, ys),
                      batch_size=cfg.run.batch_size, shuffle=True)
