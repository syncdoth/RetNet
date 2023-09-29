import torch

from retnet.modeling_retnet import RetNetModel
from retnet.configuration_retnet import RetNetConfig
from torchscale.retnet import RetNetModel as TSRetNetModel
from torchscale.config import RetNetConfig as TSRetNetConfig

my_config = RetNetConfig(
    decoder_layers=2,
    decoder_embed_dim=128,
    decoder_ffn_embed_dim=256,
    decoder_retention_heads=2,
)
my_retnet = RetNetModel(my_config)

ts_config = TSRetNetConfig(
    decoder_layers=2,
    decoder_embed_dim=128,
    decoder_ffn_embed_dim=256,
    decoder_retention_heads=2,
)
ts_retnet = TSRetNetModel(ts_config, embed_tokens=torch.nn.Embedding(50257, 128))

# compare configs
for key in ts_config.__dict__.keys():
    if key not in my_config.__dict__.keys():
        print("missing key:", key)
    elif getattr(my_config, key) != getattr(ts_config, key):
        print("different default value:", key)

# compare state_dict
ts_dict = ts_retnet.state_dict()
my_dict = my_retnet.state_dict()

# test the names
assert set(ts_dict.keys()) - set(my_dict.keys()) == set()
assert set(my_dict.keys()) - set(ts_dict.keys()) == set()

# test the shapes
for keys in ts_dict.keys():
    if ts_dict[keys].shape != my_dict[keys].shape:
        print(keys)
