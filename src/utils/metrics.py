import torch

def mae(y, y_hat):
    return torch.mean(torch.abs(y - y_hat)).item()

def rmse(y, y_hat):
    return torch.sqrt(torch.mean((y - y_hat)**2)).item()

def smape(y, y_hat, eps=1e-6):
    return torch.mean(2*torch.abs(y - y_hat)/(torch.abs(y)+torch.abs(y_hat)+eps)).item()
