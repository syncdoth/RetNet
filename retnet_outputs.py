from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers.modeling_outputs import ModelOutput


@dataclass
class RetNetOutputWithPast(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(torch.FloatTensor)`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape
            `(batch_size, num_heads, qk_dim, v_dim)`.

            Contains pre-computed hidden-states (key and values in the multi-scale retention blocks)
            that can be used (see `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        retentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_retentions=True` is passed or when `config.output_retentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Retentions weights, used for visualization.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    retentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class RetNetCausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(torch.FloatTensor)`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape
            `(batch_size, num_heads, qk_dim, v_dim)`.

            Contains pre-computed hidden-states (key and values in the multi-scale retention blocks)
            that can be used (see `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        retentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_retentions=True` is passed or when `config.output_retentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Retentions weights, used for visualization.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    retentions: Optional[Tuple[torch.FloatTensor]] = None