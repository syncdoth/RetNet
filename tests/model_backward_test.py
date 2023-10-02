"""From the root of the repo, run `pytest tests/model_backward_test.py`
to test the backward pass of the model. Simply run `pytest` to run all tests.
You might want to add -s to see the print outputs."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from retnet.modeling_retnet import RetNetForCausalLM
from retnet.configuration_retnet import RetNetConfig
from torchscale.retnet import RetNetModel as TSRetNetModel
from torchscale.config import RetNetConfig as TSRetNetConfig
from transformers import AutoTokenizer

torch.autograd.set_detect_anomaly(True)
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
my_config = RetNetConfig(
    decoder_layers=6,
    decoder_embed_dim=128,
    decoder_value_embed_dim=256,
    decoder_ffn_embed_dim=256,
    decoder_retention_heads=2,
    recurrent_chunk_size=2,
    vocab_size=50257,
    tie_word_embeddings=False,
)
my_retnet = RetNetForCausalLM(my_config)

ts_config = TSRetNetConfig(
    decoder_layers=6,
    decoder_embed_dim=128,
    decoder_value_embed_dim=256,
    decoder_ffn_embed_dim=256,
    decoder_retention_heads=2,
    recurrent_chunk_size=2,
    vocab_size=50257,
    no_output_layer=False,
)
ts_retnet = TSRetNetModel(ts_config,
                          embed_tokens=torch.nn.Embedding(50257, 128, ts_config.pad_token_id),
                          output_projection=torch.nn.Linear(128, 50257, bias=False))

incompatible_keys = my_retnet.model.load_state_dict(ts_retnet.state_dict(), strict=False)


def test_load_state_dict():
    assert len(incompatible_keys.unexpected_keys) == 1
    assert incompatible_keys.unexpected_keys[0] == 'output_projection.weight'
    assert len(incompatible_keys.missing_keys) == 0


my_retnet.lm_head.load_state_dict(ts_retnet.output_projection.state_dict())

my_retnet.train()
ts_retnet.train()
# device
my_retnet.to(device)
ts_retnet.to(device)

text = [
    "Let's compare the models on this dummy text!",
    "Let's compare the models on this dummy text!",
    "Let's compare the models on this dummy text! This one is longer",
]
loss_fct = torch.nn.CrossEntropyLoss()


def test_check_requires_grad():
    for key, param in my_retnet.named_parameters():
        assert param.requires_grad, f"Parameter {key} does not require grad"

    for key, param in ts_retnet.named_parameters():
        assert param.requires_grad, f"Parameter {key} does not require grad"


def get_data_sample(batch_text: list):
    inputs = tokenizer(batch_text, return_tensors='pt', padding=True)
    input_ids = inputs.input_ids.to(device)
    mask = inputs.attention_mask.to(device)
    labels = torch.where(mask == 1, input_ids, -100).to(device)
    return input_ids, labels


def compute_loss(logits, labels):
    """compute loss"""
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()
    loss = loss_fct(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1))
    return loss


def test_same_parameter():
    # quick check: are the parameters the same?
    for key, p1 in ts_retnet.named_parameters():
        if key == 'output_projection.weight':
            p2 = my_retnet.lm_head.state_dict()['weight']
        else:
            p2 = my_retnet.model.state_dict()[key]
        assert torch.allclose(p1, p2), f"Parameter {key} is not the same"


def test_parallel_gradient():
    input_ids, labels = get_data_sample(text)
    my_outputs = my_retnet(input_ids=input_ids, forward_impl='parallel')
    ts_outputs = ts_retnet(input_ids)
    my_logits = my_outputs.logits
    ts_logits = ts_outputs[0]

    assert torch.allclose(my_logits, ts_logits)

    my_loss = compute_loss(my_logits, labels)
    ts_loss = compute_loss(ts_logits, labels)
    assert torch.allclose(my_loss, ts_loss)

    my_loss.backward()
    ts_loss.backward()

    my_param_dict = dict(my_retnet.model.named_parameters())
    for key, p1 in ts_retnet.named_parameters():
        if key == 'output_projection.weight':
            p2 = my_retnet.lm_head.weight
        else:
            p2 = my_param_dict[key]

        # NOTE: To see the outputs, use `pytest -s` option
        if p1.grad is None and p2.grad is None:
            print(key, "both grad is None")
            continue
        if p1.grad is not None and p2.grad is None:
            print(key, "p2 grad is None")
            continue
        if p1.grad is None and p2.grad is not None:
            print(key, "p1 grad is None")
            continue
        assert torch.allclose(p1.grad, p2.grad), f"Parameter {key}'s grad is not the same"
    my_retnet.zero_grad()
    ts_retnet.zero_grad()


def test_recurrent_gradient():
    input_ids, labels = get_data_sample(text)
    incremental_state = {}
    past_key_values = None
    my_logits = []
    ts_logits = []
    for i in range(input_ids.shape[1]):
        my_outputs = my_retnet(input_ids=input_ids[:, :i + 1],
                               past_key_values=past_key_values,
                               forward_impl='recurrent',
                               use_cache=True)
        my_logits.append(my_outputs.logits)
        past_key_values = my_outputs.past_key_values
        ts_outputs = ts_retnet(input_ids[:, :i + 1], incremental_state=incremental_state)
        ts_logits.append(ts_outputs[0])

    my_logits = torch.cat(my_logits, dim=1)
    ts_logits = torch.cat(ts_logits, dim=1)
    assert torch.allclose(my_logits, ts_logits)

    my_loss = compute_loss(my_logits, labels)
    ts_loss = compute_loss(ts_logits, labels)
    assert torch.allclose(my_loss, ts_loss)

    my_loss.backward()
    ts_loss.backward()

    my_param_dict = dict(my_retnet.model.named_parameters())
    for key, p1 in ts_retnet.named_parameters():
        if key == 'output_projection.weight':
            p2 = my_retnet.lm_head.weight
        else:
            p2 = my_param_dict[key]

        # NOTE:To see the outputs, use `pytest -s` option
        if p1.grad is None and p2.grad is None:
            print(key, "both grad is None")
            continue
        if p1.grad is not None and p2.grad is None:
            print(key, "p2 grad is None")
            continue
        if p1.grad is None and p2.grad is not None:
            print(key, "p1 grad is None")
            continue
        assert torch.allclose(p1.grad, p2.grad), f"Parameter {key}'s grad is not the same"
    my_retnet.zero_grad()
    ts_retnet.zero_grad()


def test_chunkwise_gradient():
    input_ids, labels = get_data_sample(text)
    my_outputs = my_retnet(input_ids=input_ids, forward_impl='chunkwise')
    ts_retnet.chunkwise_recurrent = True
    ts_outputs = ts_retnet(input_ids)
    ts_retnet.chunkwise_recurrent = False
    my_logits = my_outputs.logits
    ts_logits = ts_outputs[0]

    assert torch.allclose(my_logits, ts_logits)

    my_loss = compute_loss(my_logits, labels)
    ts_loss = compute_loss(ts_logits, labels)
    assert torch.allclose(my_loss, ts_loss)

    my_loss.backward()
    ts_loss.backward()

    my_param_dict = dict(my_retnet.model.named_parameters())
    for key, p1 in ts_retnet.named_parameters():
        if key == 'output_projection.weight':
            p2 = my_retnet.lm_head.weight
        else:
            p2 = my_param_dict[key]

        # NOTE:To see the outputs, use `pytest -s` option
        if p1.grad is None and p2.grad is None:
            print(key, "both grad is None")
            continue
        if p1.grad is not None and p2.grad is None:
            print(key, "p2 grad is None")
            continue
        if p1.grad is None and p2.grad is not None:
            print(key, "p1 grad is None")
            continue
        assert torch.allclose(p1.grad, p2.grad), f"Parameter {key}'s grad is not the same"
    my_retnet.zero_grad()
    ts_retnet.zero_grad()


if __name__ == '__main__':
    test_parallel_gradient()