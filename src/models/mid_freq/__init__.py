# src/models/mid_freq/__init__.py
from .convlstm import ConvLSTM
#from .gru import GRUStack
def build(name, cfg):
    return {"convlstm": ConvLSTM
            #"gru": GRUStack
            }[name](cfg)
