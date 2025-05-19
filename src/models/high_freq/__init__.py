from .transformer import HighTransformer
#from .temporal_cnn import TemporalCNN

def build(name, cfg):
    table = {
        "transformer": HighTransformer
        #"temp_cnn": TemporalCNN,
    }
    return table[name](cfg)