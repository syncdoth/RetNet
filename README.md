# RetNet

A minimal yet complete implementation of Retention Networks. [https://arxiv.org/pdf/2307.08621.pdf](https://arxiv.org/pdf/2307.08621.pdf)
Supports all types of forward implementations: `parallel`, `recurrent`, `chunkwise`

Check `play.ipynb` for minimal testing of parallel, recurrent, and chunkwise forward.

## Getting Started

~~**All you need is PyTorch.**~~ Using `PyTorch` and huggingface `transformers`.

```bash
pip install torch transformers
```

You may want to use `conda`.

### Quick Examples

Take a look at `play.ipynb`.

```python
import torch
from model import RetNetModel, RetNetConfig

config = RetNetConfig(num_layers=8, vocab_size=100, hidden_size=512, num_heads=4, use_default_gamma=False, chunk_size=4)
model = RetNetModel(config)

input_ids = torch.LongTensor([[1,2,3,4]])

# parallel forward
out, parallel_past_kv = model(input_ids, forward_impl='parallel', return_kv=True)

# recurrent forward
past_kv = None
rnn_outs = []
for i in range(input_ids.shape[1]):
    rnn_out, past_kv = model(input_ids[:, i:i+1], forward_impl='recurrent', past_kv=past_kv, return_kv=True, sequence_offset=i)
    rnn_outs.append(rnn_out)
rnn_outs = torch.cat(rnn_outs, dim=1)

# chunkwise (implemented chunkwise within the forward)
chunk_out, chunk_past_kv = model(input_ids, forward_impl='chunkwise', return_kv=True)
```

### Language Generation


```python
import torch
from model import RetNetModelWithLMHead, RetNetConfig

config = RetNetConfig(num_layers=8, vocab_size=100, hidden_size=512, num_heads=4, use_default_gamma=False, chunk_size=4)
model = RetNetModel(config)

input_ids = torch.LongTensor([[1,2,3,4]])

# parallel forward
generated = model.generate(input_ids, parallel_compute_prompt=True, max_new_tokens=20)
```

- `parallel_compute_prompt = (default: True)`: Thanks to parallel forward being able
  to compute `past_kv`, we can compute parallel forward first, then feed the `past_kv`
  in to recurrent forward, which can save number of forwards for GPU with enough memory.

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
## TODOs
### Huggingface style

- make the implementation based on huggingface `PretrainedModel`.
- This includes handling `attention_mask` (for padding), etc.
