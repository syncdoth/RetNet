import torch

from retnet.configuration_retnet import RetNetConfig
from retnet.modeling_retnet import RetNetForCausalLM

torch.manual_seed(0)
config = RetNetConfig(decoder_layers=6,
                      decoder_embed_dim=256,
                      decoder_value_embed_dim=512,
                      decoder_retention_heads=2,
                      decoder_ffn_embed_dim=512)

model = RetNetForCausalLM(config)
model.eval()

test_cases = [
    dict(input_ids=torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8]])),
    dict(input_ids=torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]])),
    dict(input_ids=torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
         attention_mask=torch.LongTensor([[1, 1, 1, 1, 1, 1, 1, 1]])),
    dict(input_ids=torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
         attention_mask=torch.LongTensor([[0, 0, 1, 1, 1, 1, 1, 1]])),
    dict(input_ids=torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]]),
         attention_mask=torch.LongTensor([[0, 0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]])),
]


def test_greedy():
    generate_kwargs = dict(max_new_tokens=20, early_stopping=False, do_sample=False)

    for inputs in test_cases:
        generated1 = model.custom_generate(**inputs,
                                           parallel_compute_prompt=False,
                                           **generate_kwargs)
        generated2 = model.custom_generate(**inputs,
                                           parallel_compute_prompt=True,
                                           **generate_kwargs)
        generated3 = model.generate(**inputs, **generate_kwargs)
        assert (generated1 == generated2).all(), "recurrent vs parallel"
        assert (generated1 == generated3).all(), "recurrent vs huggingface"
        assert (generated2 == generated3).all(), "parallel vs huggingface"


# sample
def test_sampling():
    generate_kwargs = dict(max_new_tokens=20, early_stopping=False, do_sample=True, top_k=0)
    for inputs in test_cases:
        torch.manual_seed(0)
        generated1 = model.custom_generate(**inputs,
                                           parallel_compute_prompt=False,
                                           **generate_kwargs)
        torch.manual_seed(0)
        generated2 = model.custom_generate(**inputs,
                                           parallel_compute_prompt=True,
                                           **generate_kwargs)
        torch.manual_seed(0)
        generated3 = model.generate(**inputs, **generate_kwargs)
        assert (generated1 == generated2).all(), "recurrent vs parallel"
        assert (generated1 == generated3).all(), "recurrent vs huggingface"
        assert (generated2 == generated3).all(), "parallel vs huggingface"


def test_sampling_topk():
    generate_kwargs = dict(max_new_tokens=20, early_stopping=False, do_sample=True, top_k=50)
    for inputs in test_cases:
        torch.manual_seed(0)
        generated1 = model.custom_generate(**inputs,
                                           parallel_compute_prompt=False,
                                           **generate_kwargs)
        torch.manual_seed(0)
        generated2 = model.custom_generate(**inputs,
                                           parallel_compute_prompt=True,
                                           **generate_kwargs)
        torch.manual_seed(0)
        generated3 = model.generate(**inputs, **generate_kwargs)
        assert (generated1 == generated2).all(), "recurrent vs parallel"
        assert (generated1 == generated3).all(), "recurrent vs huggingface"
        assert (generated2 == generated3).all(), "parallel vs huggingface"


def test_sampling_top_p():
    generate_kwargs = dict(max_new_tokens=20,
                           early_stopping=False,
                           do_sample=True,
                           top_k=0,
                           top_p=0.8)
    for inputs in test_cases:
        torch.manual_seed(0)
        generated1 = model.custom_generate(**inputs,
                                           parallel_compute_prompt=False,
                                           **generate_kwargs)
        torch.manual_seed(0)
        generated2 = model.custom_generate(**inputs,
                                           parallel_compute_prompt=True,
                                           **generate_kwargs)
        torch.manual_seed(0)
        generated3 = model.generate(**inputs, **generate_kwargs)
        assert (generated1 == generated2).all(), "recurrent vs parallel"
        assert (generated1 == generated3).all(), "recurrent vs huggingface"
        assert (generated2 == generated3).all(), "parallel vs huggingface"


def test_sampling_temp():
    generate_kwargs = dict(max_new_tokens=20,
                           early_stopping=False,
                           do_sample=True,
                           top_k=0,
                           top_p=0.9,
                           temperature=1.2)
    for inputs in test_cases:
        torch.manual_seed(0)
        generated1 = model.custom_generate(**inputs,
                                           parallel_compute_prompt=False,
                                           **generate_kwargs)
        torch.manual_seed(0)
        generated2 = model.custom_generate(**inputs,
                                           parallel_compute_prompt=True,
                                           **generate_kwargs)
        torch.manual_seed(0)
        generated3 = model.generate(**inputs, **generate_kwargs)
        assert (generated1 == generated2).all(), "recurrent vs parallel"
        assert (generated1 == generated3).all(), "recurrent vs huggingface"
        assert (generated2 == generated3).all(), "parallel vs huggingface"


def test_beamsearch():
    for inputs in test_cases:
        generate_kwargs = dict(max_new_tokens=20, early_stopping=False, num_beams=4)
        generated = model.generate(**inputs, **generate_kwargs)
        # TODO: this just checks if there's no error, but we should check the output
