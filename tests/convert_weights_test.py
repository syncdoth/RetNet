"""
NOTE: this creates a hf_retnet folder containing weights. You can delete it after test.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from retnet.configuration_retnet import RetNetConfig
from retnet.modeling_retnet import RetNetForCausalLM, RetNetModel
from torchscale.config import RetNetConfig as TSRetNetConfig
from torchscale.retnet import RetNetModel as TSRetNetModel

from convert_weights import main as convert_weight

ts_config = TSRetNetConfig(
    decoder_layers=6,
    decoder_embed_dim=128,
    decoder_value_embed_dim=256,
    decoder_ffn_embed_dim=256,
    decoder_retention_heads=2,
    recurrent_chunk_size=2,
    vocab_size=32000,
    no_output_layer=False,
)
ts_retnet = TSRetNetModel(ts_config,
                          embed_tokens=torch.nn.Embedding(32000, 128, ts_config.pad_token_id),
                          output_projection=torch.nn.Linear(128, 32000, bias=False))
torch.save(ts_retnet.state_dict(), 'ts_retnet.pth')

# convert model weight
convert_weight('ts_retnet.pth', 'hf_retnet/pytorch_model.bin')
os.remove('ts_retnet.pth')

# 1. load weight
my_config = RetNetConfig()
my_config.override(ts_config)
my_retnet = RetNetForCausalLM(my_config)

my_retnet.load_state_dict(torch.load('hf_retnet/pytorch_model.bin', map_location='cpu'))

# 2. load using RetNetForCausalLM.from_pretrained
my_config.save_pretrained('hf_retnet')
from_retnet = RetNetForCausalLM.from_pretrained('hf_retnet')


def test_load1():
    for k, v in my_retnet.state_dict().items():
        assert torch.all(torch.eq(v, from_retnet.state_dict()[k])), k


# 3. load using RetNetModel.from_pretrained
def test_load2():
    from_retnet_model = RetNetModel.from_pretrained('hf_retnet')
    for k, v in from_retnet.model.state_dict().items():
        assert torch.all(torch.eq(v, from_retnet_model.state_dict()[k])), k
