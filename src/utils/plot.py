import matplotlib.pyplot as plt

def plot_history(hist, metrics, path):
    for k in metrics:
        plt.figure()
        plt.plot(hist["train_"+k], label="train")
        plt.plot(hist["val_"+k], label="val")
        plt.xlabel("epoch"); plt.ylabel(k); plt.legend()
        plt.tight_layout(); plt.savefig(f"{path}/{k}.png", dpi=200); plt.close()
