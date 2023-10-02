import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from retnet.modeling_retnet import RetNetModel
from retnet.configuration_retnet import RetNetConfig
from torchscale.retnet import RetNetModel as TSRetNetModel
from torchscale.config import RetNetConfig as TSRetNetConfig
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
my_config = RetNetConfig(
    decoder_layers=6,
    decoder_embed_dim=128,
    decoder_value_embed_dim=256,
    decoder_ffn_embed_dim=256,
    decoder_retention_heads=2,
    recurrent_chunk_size=2,
    vocab_size=50257,
)
my_retnet = RetNetModel(my_config)

ts_config = TSRetNetConfig(
    decoder_layers=6,
    decoder_embed_dim=128,
    decoder_value_embed_dim=256,
    decoder_ffn_embed_dim=256,
    decoder_retention_heads=2,
    recurrent_chunk_size=2,
    vocab_size=50257,
)
ts_retnet = TSRetNetModel(ts_config, embed_tokens=torch.nn.Embedding(50257, 128))

my_retnet.load_state_dict(ts_retnet.state_dict())

text = "Let's compare the models on this dummy text!"
input_ids = tokenizer(text, return_tensors='pt')['input_ids']

my_retnet.eval()
ts_retnet.eval()

# device
input_ids.to(device)
my_retnet.to(device)
ts_retnet.to(device)


def test_same_parameter():
    # quick check: are the parameters the same?
    for p1, p2 in zip(my_retnet.parameters(), ts_retnet.parameters()):
        assert torch.allclose(p1, p2)


def test_parallel_forward():
    with torch.inference_mode():
        my_outputs = my_retnet(input_ids=input_ids, forward_impl='parallel')
        ts_outputs = ts_retnet(input_ids, features_only=True)
        my_parallel = my_outputs.last_hidden_state
        ts_parallel = ts_outputs[0]
    assert torch.allclose(my_parallel, ts_parallel)


def test_recurrent_forward():
    with torch.inference_mode():
        incremental_state = {}
        past_key_values = None
        my_rnn = []
        ts_rnn = []
        for i in range(input_ids.shape[1]):
            my_outputs = my_retnet(input_ids=input_ids[:, :i + 1],
                                   past_key_values=past_key_values,
                                   forward_impl='recurrent',
                                   use_cache=True)
            past_key_values = my_outputs.past_key_values
            my_rnn.append(my_outputs.last_hidden_state)
            ts_outputs = ts_retnet(input_ids[:, :i + 1],
                                   incremental_state=incremental_state,
                                   features_only=True)
            ts_rnn.append(ts_outputs[0])
        my_rnn = torch.cat(my_rnn, dim=1)
        ts_rnn = torch.cat(ts_rnn, dim=1)
    assert torch.allclose(my_rnn, ts_rnn)


def test_chunkwise_forward():
    with torch.inference_mode():
        ts_retnet.chunkwise_recurrent = True
        my_outputs = my_retnet(input_ids=input_ids, forward_impl='chunkwise')
        ts_outputs = ts_retnet(input_ids, features_only=True)
        my_chunk = my_outputs.last_hidden_state
        ts_chunk = ts_outputs[0]
    assert torch.allclose(my_chunk, ts_chunk)
