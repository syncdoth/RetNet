# RetNet

A huggingface transformer compatible implementation of Retention Networks. ([https://arxiv.org/pdf/2307.08621.pdf](https://arxiv.org/pdf/2307.08621.pdf))
Supports all types of forward implementations: `parallel`, `recurrent`, `chunkwise`

Check `play.ipynb` for minimal testing of parallel, recurrent, and chunkwise forward.

## Getting Started

Using `PyTorch` and huggingface `transformers`.

```bash
pip install torch transformers
```

You may want to use `conda`.

### Quick Examples

Take a look at `play.ipynb`.

```python
import torch
from retnet.modeling_retnet import RetNetModel
from retnet.configuration_retnet import RetNetConfig

config = RetNetConfig(num_layers=8,
                      hidden_size=512,
                      num_heads=4,
                      qk_dim=512,
                      v_dim=1024,
                      ffn_proj_size=1024,
                      use_default_gamma=False)
model = RetNetModel(config)

input_ids = torch.LongTensor([[1,2,3,4,5,6,7,8]])

parallel_outputs = model(input_ids, forward_impl='parallel', use_cache=True)
parallel_state = parallel_outputs.last_hidden_state
parallel_cache = parallel_outputs.past_key_values

past_kv = None
rnn_state = []
for i in range(input_ids.shape[1]):
    rnn_out = model(input_ids[:, i:i+1], forward_impl='recurrent', past_key_values=past_kv, use_cache=True, sequence_offset=i)
    rnn_state.append(rnn_out.last_hidden_state)
    past_kv = rnn_out.past_key_values
rnn_state = torch.cat(rnn_state, dim=1)
rnn_cache = rnn_out.past_key_values


chunk_outputs = model(input_ids, forward_impl='chunkwise', use_cache=True, chunk_size=4)
chunk_state = chunk_outputs.last_hidden_state
chunk_cache = chunk_outputs.past_key_values

```

### Language Generation


```python
import torch
from retnet.modeling_retnet import RetNetModelWithLMHead
from retnet.configuration_retnet import load_config_from_yaml
from transformers import AutoTokenizer

config = load_config_from_yaml('configs/retnet-1.3b.yml')
model = RetNetModelWithLMHead(config)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.model_max_length = 8192
tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer("Retention refers to", return_tensors='pt')

# parallel forward
generated = model.generate(**inputs, parallel_compute_prompt=True, max_new_tokens=20)
tokenizer.batch_decode(generated)
# NOTE: this should be gibberish, since the model is not trained.
```

- `parallel_compute_prompt = (default: True)`: Thanks to parallel forward being able
  to compute `past_kv`, we can compute parallel forward first, then feed the `past_kv`
  in to recurrent forward, which can save number of forwards for GPU with enough memory.

## Huggingface Integration

Because of `sequence_offset` parameter, it cannot utilize `GenerateMixin.generate` function.
Resorting to custom generate function for now.

### Minimal Training Example

You can train RetNet with huggingface `Trainer` API. Refer to `train.py`.

```bash
export CUDA_VISIBLE_DEVICES=0

python train.py \
    --model_size 300m \
    --output_dir checkpoints \
    --do_train --do_eval \
    --prediction_loss_only \
    --remove_unused_columns False \
    --learning_rate 6e-4 \
    --weight_decay 0.01 \
    --max_steps 20000 \
    --logging_steps 10 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16

```

## xpos note

The authors mention xpos as $e^{in\theta}, e^{-im\theta}$ (equation 5). At first glance, this is
a complex number, which is difficult to use and interpret. However, this is in fact xpos,
which was made clear by this [lecture note](https://banica.u-cergy.fr/pdf/la3.pdf) for me.
The gist is:

$$ R_{\theta} = e^{i\theta} = \begin{bmatrix} cos(\theta) & -sin(\theta); \\ sin(\theta) & cos(\theta) \end{bmatrix}$$

Since xpos (which builds on RoPE) precisely does such a rotation, this is in fact, xpos.
I used the implementation of xpos fould in [torchscale](https://github.com/microsoft/torchscale)
repo with 1 small change:
instead of negative `min_pos`, I used `min_pos=0` (line 53, 54), so that it is
recurrence friendly.

## Decay Note

Equation 7 omits an important detail: there should be an extra decay applied to
$K^T_{[i]}V_{[i]}$, which is $D_{B}$, i.e. the last row of the inner_chunk decay_mask.
So, it should be re-written as:

$$R_i = K^T_{[i]}V_{[i]} \odot D_{B} + \gamma ^B R_{i-1}$$

This is implemented in the `chunkwise_retention` function, named as `intra_decay`.

This idea can also be applied to `parallel_retention` to obtain the correct `past_kv` that can be
further fed into recurrent or chunkwise retention in the next token steps.

## Configs

The `configs/` folder includes example configurations listed in the paper for
different sizes. For simplicity, I used GPT2 tokenizer, and hence the model
has 50217 as vocab size for default (this can change when microsoft release the official
weight).

- Technically, I used `EleutherAI/gpt-j-6b` tokenizer, which is identical except for
  a few extra tokens.