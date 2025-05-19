# src/models/low_freq/__init__.py
from .informer import InformerDecoder
#from .seq2seq import Seq2Seq
def build(name, cfg):
    return {"informer": InformerDecoder
            #"seq2seq": Seq2Seq
            }[name](cfg)
