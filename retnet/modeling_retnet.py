import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from timm.models.layers import drop_path
from torch import nn
from transformers import top_k_top_p_filtering
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_retnet import RetNetConfig
from .xpos_relative_position import XPOS

logger = logging.get_logger(__name__)


# helper functions
def split_chunks(*tensors, size, dim=0):
    return [torch.split(x, size, dim=dim) for x in tensors]


def split_heads(tensors, bsz, seqlen, num_heads):
    assert isinstance(tensors, (tuple, list))
    return [x.view(bsz, seqlen, num_heads, -1).transpose(1, 2) for x in tensors]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise NotImplementedError


class MultiScaleRetention(nn.Module):
    # TODO: normalization to decay in the paper
    def __init__(self, config: RetNetConfig, value_factor: int = 2):
        super().__init__()
        self.config = config
        self.embed_dim = config.decoder_embed_dim
        self.v_dim = config.decoder_embed_dim * value_factor
        self.num_heads = config.decoder_retention_heads

        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 2 + self.v_dim, bias=True)
        self.silu = nn.SiLU()
        self.gated = nn.Linear(self.embed_dim, self.v_dim, bias=True)
        self.proj = nn.Linear(self.v_dim, self.embed_dim, bias=True)
        self.gn = nn.GroupNorm(num_groups=self.num_heads, num_channels=self.v_dim, affine=False)
        self.xpos = XPOS(self.embed_dim)

        # initialize gamma
        s = torch.log(torch.tensor(1 / 32))
        e = torch.log(torch.tensor(1 / 512))
        gamma = 1 - torch.exp(torch.linspace(s, e, self.num_heads))  # [h,]
        self.decay = nn.Parameter(gamma, requires_grad=False)

    def get_parallel_decay_mask(self, length, retention_mask=None, return_scale=False):
        range_tensor = torch.arange(length, device=self.decay.device)
        range_tensor = range_tensor[None, :, None].expand(self.num_heads, length, 1)
        exponent = range_tensor - range_tensor.transpose(-1, -2)
        decay_mask = torch.exp(self.decay.view(-1, 1, 1) * exponent)  # FIX 1
        decay_mask = torch.tril(decay_mask, diagonal=0)  # [h, t, t]
        # FIX 2: rescale
        scale = decay_mask.sum(dim=-1, keepdim=True).sqrt()
        decay_mask = decay_mask / scale
        decay_mask = decay_mask.unsqueeze(0)
        if retention_mask is not None:
            retention_mask = retention_mask.float().view(-1, 1, 1, length)
            decay_mask *= retention_mask

        if return_scale:
            return decay_mask, scale.unsqueeze(0)
        return decay_mask

    def get_recurrent_decay(self):
        decay = self.decay.view(1, self.num_heads, 1, 1)
        return decay.exp()

    def get_chunkwise_decay(self, retention_mask=None):
        chunk_size = self.config.recurrent_chunk_size
        # within chunk decay
        decay_mask, scale = self.get_parallel_decay_mask(chunk_size,
                                                         retention_mask=retention_mask,
                                                         return_scale=True)
        # decay of the chunk
        chunk_decay = torch.exp(self.decay.view(1, self.num_heads, 1, 1) * chunk_size)  # FIX 1
        # cross-chunk decay
        exponent = torch.arange(chunk_size, dtype=torch.float,
                                device=decay_mask.device).unsqueeze(0) + 1
        inner_decay = torch.exp(self.decay.unsqueeze(-1) * exponent)
        inner_decay = inner_decay.view(1, self.num_heads, chunk_size,
                                       1) / (scale / scale[:, :, -1, None])
        return decay_mask, chunk_decay, inner_decay

    def parallel_retention(self, q, k, v, decay_mask):
        """
        q,  # bsz * num_head * len * qk_dim
        k,  # bsz * num_head * len * qk_dim
        v,  # bsz * num_head * len * v_dim
        decay_mask,  # (1 or bsz) * num_head * len * len
        """
        # [b, h, t, t]
        retention = q @ k.transpose(-1, -2) * k.size(-1)**-0.5  # (scaled dot-product)
        retention = retention * decay_mask
        output = retention @ v

        # kv cache
        current_kv = k.unsqueeze(-1) * v.unsqueeze(-2)
        # [bsz, num_head, qk_dim, v_dim]
        intra_decay = decay_mask[:, :, -1, :, None, None]
        current_kv = (current_kv * intra_decay).sum(2)
        return output, current_kv, retention

    def recurrent_retention(self, q, k, v, past_key_value=None, decay=None, retention_mask=None):
        """
        q, k, v, # bsz * num_head * 1 * qkv_dim
        past_key_value, # bsz * num_head * qk_dim * v_dim
        decay # num_head * 1 * 1
        retention_mask # bsz * 1
        """
        past_key_value = past_key_value if past_key_value is not None else 0
        decay = decay if decay is not None else 0
        retention_mask = retention_mask.view(-1, 1, 1, 1) if retention_mask is not None else 1
        # (b, h, d_k, d_v)
        current_kv = decay * past_key_value + retention_mask * (k.transpose(-1, -2) @ v)
        output = q @ current_kv * k.size(-1)**-0.5  # (b, h, 1, d_v)
        return output, current_kv

    def chunkwise_retention(self,
                            q,
                            k,
                            v,
                            decay_mask,
                            past_key_value=None,
                            chunk_decay=None,
                            inner_decay=None):
        """
        q, k, v,  # bsz * num_head * chunk_size * qkv_dim
        past_key_value,  # bsz * num_head * qk_dim * v_dim
        decay_mask,  # 1 * num_head * chunk_size * chunk_size
        chunk_decay,  # 1 * num_head * 1 * 1
        inner_decay,  # 1 * num_head * chunk_size * 1
        """
        # [bsz, num_head, chunk_size, chunk_size]
        retention = q @ k.transpose(-1, -2) * k.size(-1)**-0.5
        retention = retention * decay_mask
        inner_retention = retention @ v  # [bsz, num_head, chunk_size, v_dim]

        if past_key_value is None:
            cross_retention = 0
            past_chunk = 0
        else:
            cross_retention = (q @ past_key_value) * inner_decay * k.size(-1)**-0.5
            past_chunk = chunk_decay * past_key_value

        # [bsz, num_head, chunk_size, v_dim]
        retention = inner_retention + cross_retention
        # [bsz, num_head, chunk_size, qk_dim, v_dim]
        current_kv = k.unsqueeze(-1) * v.unsqueeze(-2)
        # NOTE: intra_decay is omitted in the paper; but this detail is important
        # [bsz, num_head, qk_dim, v_dim]
        intra_decay = decay_mask[:, :, -1, :, None, None]
        current_kv = (current_kv * intra_decay).sum(2)
        current_kv = past_chunk + current_kv
        return retention, current_kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        retention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        forward_impl: str = 'parallel',
        sequence_offset: Optional[int] = 0,
        recurrent_chunk_size: Optional[int] = None,
        output_retentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor]]:
        B, T, H = hidden_states.size()
        q, k, v = self.qkv(hidden_states).split([self.embed_dim, self.embed_dim, self.v_dim],
                                                dim=-1)
        q, k = self.xpos.rotate_queries_and_keys(q, k, offset=sequence_offset)
        q, k, v = split_heads((q, k, v), B, T, self.num_heads)
        # retention
        if forward_impl == 'parallel':
            decay_mask = self.get_parallel_decay_mask(T, retention_mask=retention_mask)
            retention_out, curr_kv, retention_weights = self.parallel_retention(q, k, v, decay_mask)
        elif forward_impl == 'recurrent':
            decay = self.get_recurrent_decay()
            retention_out, curr_kv = self.recurrent_retention(q,
                                                              k,
                                                              v,
                                                              past_key_value=past_key_value,
                                                              decay=decay,
                                                              retention_mask=retention_mask)
        elif forward_impl == 'chunkwise':
            assert recurrent_chunk_size is not None
            q_chunks, k_chunks, v_chunks = split_chunks(q, k, v, size=recurrent_chunk_size, dim=2)
            if retention_mask is not None:
                retention_mask_chunks = split_chunks(retention_mask,
                                                     size=recurrent_chunk_size,
                                                     dim=1)[0]
            ret_chunks = []
            for i, (_q, _k, _v) in enumerate(zip(q_chunks, k_chunks, v_chunks)):
                csz = _q.size(2)
                ret_mask = retention_mask_chunks[i] if retention_mask is not None else None
                decay_mask, chunk_decay, inner_decay = self.get_chunkwise_decay(
                    csz, retention_mask=ret_mask)
                out_chunk, past_key_value = self.chunkwise_retention(_q,
                                                                     _k,
                                                                     _v,
                                                                     decay_mask,
                                                                     past_key_value=past_key_value,
                                                                     chunk_decay=chunk_decay,
                                                                     inner_decay=inner_decay)
                ret_chunks.append(out_chunk)
            # [bsz, num_head, seqlen, v_dim]
            retention_out = torch.cat(ret_chunks, dim=2)
            curr_kv = past_key_value
        else:
            raise ValueError(f'forward_impl {forward_impl} not supported.')
        # concaat heads
        retention_out = retention_out.transpose(1, 2).contiguous().view(B, T, self.v_dim)
        # group norm (merge batch, length dimension -> group norm -> split back)
        normed = self.gn(retention_out.view(B * T, self.v_dim))
        normed = normed.view(B, T, self.v_dim)
        # out gate & proj
        out = self.silu(self.gated(hidden_states)) * normed

        outputs = (self.proj(out), curr_kv)
        if output_retentions:
            outputs += (retention_weights,) if forward_impl == 'parallel' else (None,)
        return outputs


class FeedForwardNetwork(nn.Module):

    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn,
        dropout,
        activation_dropout,
        layernorm_eps,
        subln=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = get_activation_fn(activation=str(activation_fn))
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.ffn_layernorm = nn.LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x):
        x_shape = x.shape
        x = x.reshape(-1, x.size(-1))
        x = self.fc1(x)
        x = self.activation_fn(x.float()).type_as(x)
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = x.view(x_shape)
        x = self.dropout_module(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return "p={}".format(self.drop_prob)


class RetNetDecoderLayer(nn.Module):

    def __init__(self, config: RetNetConfig, depth: int):
        super().__init__()
        self.config = config
        self.embed_dim = config.decoder_embed_dim
        self.dropout_module = torch.nn.Dropout(config.dropout)

        if config.drop_path_rate > 0:
            drop_path_prob = np.linspace(0, config.drop_path_rate, config.decoder_layers)[depth]
            self.drop_path = DropPath(drop_path_prob)
        else:
            self.drop_path = None

        self.retention = MultiScaleRetention(
            config,
            value_factor=2,  # TODO
        )

        self.normalize_before = config.decoder_normalize_before

        self.retention_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layernorm_eps)

        self.ffn_dim = config.decoder_ffn_embed_dim

        self.ffn = FeedForwardNetwork(
            self.embed_dim,
            self.ffn_dim,
            self.config.activation_fn,
            self.config.dropout,
            self.config.activation_dropout,
            self.config.layernorm_eps,
            self.config.subln,
        )

        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layernorm_eps)

        if config.deepnorm:
            self.alpha = math.pow(2.0 * config.decoder_layers, 0.25)
        else:
            self.alpha = 1.0

    def residual_connection(self, x, residual):
        return residual * self.alpha + x

    def forward(
        self,
        hidden_states: torch.Tensor,
        incremental_state=None,
        chunkwise_recurrent=False,
        retention_rel_pos=None,
        retention_mask: Optional[torch.Tensor] = None,
        forward_impl: str = 'parallel',
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        sequence_offset: Optional[int] = 0,
        recurrent_chunk_size: Optional[int] = None,
        output_retentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor]]:
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.retention_layer_norm(hidden_states)

        # TODO: resolve this by using the same interface as the original RetNet
        # hidden_states = self.retention(
        #     hidden_states,
        #     incremental_state=incremental_state,
        #     rel_pos=retention_rel_pos,
        #     chunkwise_recurrent=chunkwise_recurrent,
        # )
        msr_outs = self.retention(hidden_states,
                                  retention_mask=retention_mask,
                                  past_key_value=past_key_value,
                                  forward_impl=forward_impl,
                                  sequence_offset=sequence_offset,
                                  recurrent_chunk_size=recurrent_chunk_size,
                                  output_retentions=output_retentions)
        hidden_states = msr_outs[0]
        curr_kv = msr_outs[1]

        hidden_states = self.dropout_module(hidden_states)

        if self.drop_path is not None:
            hidden_states = self.drop_path(hidden_states)

        hidden_states = self.residual_connection(hidden_states, residual)
        if not self.normalize_before:
            hidden_states = self.retention_layer_norm(hidden_states)

        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.ffn(hidden_states)

        if self.drop_path is not None:
            hidden_states = self.drop_path(hidden_states)

        hidden_states = self.residual_connection(hidden_states, residual)
        if not self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states, curr_kv)

        if output_retentions:
            outputs += (msr_outs[2],)
        return outputs


class RetNetPreTrainedModel(PreTrainedModel):
    # copied from LlamaPretrainedModel
    config_class = RetNetConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RetNetDecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, RetNetModel):
            module.gradient_checkpointing = value


@dataclass
class RetNetOutputWithPast(ModelOutput):
    """
    class for RetNet model's outputs that may also contain a past key/values (to speed up sequential decoding).

    config:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, decoder_embed_dim)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            decoder_embed_dim)` is output.
        past_key_values (`tuple(torch.FloatTensor)`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape
            `(batch_size, num_heads, qk_dim, v_dim)`.

            Contains pre-computed hidden-states (key and values in the multi-scale retention blocks)
            that can be used (see `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, decoder_embed_dim)`.

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


class RetNetModel(RetNetPreTrainedModel):

    def __init__(self, config: RetNetConfig, embed_tokens: nn.Embedding = None):
        super().__init__(config)
        self.config = config

        self.dropout_module = torch.nn.Dropout(config.dropout)

        self.embed_dim = config.decoder_embed_dim
        self.embed_scale = 1.0 if config.no_scale_embedding else math.sqrt(self.embed_dim)

        if embed_tokens is None:
            embed_tokens = nn.Embedding(config.vocab_size, config.decoder_embed_dim,
                                        config.pad_token_id)
        self.embed_tokens = embed_tokens

        if config.layernorm_embedding:
            self.layernorm_embedding = nn.LayerNorm(self.embed_dim, eps=config.layernorm_eps)
        else:
            self.layernorm_embedding = None

        self.layers = nn.ModuleList([])

        for i in range(config.decoder_layers):
            self.layers.append(RetNetDecoderLayer(config, depth=i))

        self.decoder_layers = len(self.layers)

        if config.decoder_normalize_before:
            self.layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layernorm_eps)
        else:
            self.layer_norm = None

        self.retnet_rel_pos = RetNetRelPos(config)
        self.chunkwise_recurrent = config.chunkwise_recurrent
        self.recurrent_chunk_size = config.recurrent_chunk_size

        if config.deepnorm:
            init_scale = math.pow(8.0 * config.decoder_layers, 0.25)
            for name, p in self.named_parameters():
                if ("fc1" in name or "fc2" in name or "out_proj" in name or "v_proj" in name):
                    p.data.div_(init_scale)

        if config.subln:
            init_scale = math.sqrt(math.log(config.decoder_layers * 2))
            for name, p in self.named_parameters():
                if ("fc1" in name or "fc2" in name or "out_proj" in name or "v_proj" in name):
                    p.data.mul_(init_scale)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward_embedding(
        self,
        input_ids,
        inputs_embeds=None,
        incremental_state=None,
    ):
        if incremental_state is not None and not self.is_first_step(incremental_state):
            input_ids = input_ids[:, -1:]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        embed = self.embed_scale * inputs_embeds

        if self.layernorm_embedding is not None:
            embed = self.layernorm_embedding(embed)

        embed = self.dropout_module(embed)

        return embed

    def is_first_step(self, incremental_state):
        if incremental_state is None:
            return False
        return incremental_state.get("is_first_step", False)

    def forward(
        self,
        incremental_state=None,  # TODO: merge with past_key_values
        input_ids: torch.LongTensor = None,
        retention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_retentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        forward_impl: Optional[str] = 'parallel',
        sequence_offset: Optional[int] = 0,
        recurrent_chunk_size: Optional[int] = None,
    ) -> Union[Tuple, RetNetOutputWithPast]:

        output_retentions = output_retentions if output_retentions is not None else self.config.output_retentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else
                                self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if forward_impl == 'recurrent' and seq_length > 1:
            raise ValueError('Recurrent forward only supports sequence length 1.')

        # embed tokens
        if inputs_embeds is None:
            inputs_embeds = self.forward_embedding(input_ids, inputs_embeds, incremental_state)

        if retention_mask is None:
            if attention_mask is not None:
                retention_mask = attention_mask
            else:
                # TODO: might not need this
                retention_mask = torch.ones((batch_size, seq_length),
                                            dtype=torch.bool,
                                            device=inputs_embeds.device)

        is_first_step = self.is_first_step(incremental_state)
        hidden_states = inputs_embeds

        need_pad_for_chunkwise = (self.chunkwise_recurrent and
                                  seq_length % self.recurrent_chunk_size != 0)
        if need_pad_for_chunkwise:
            padding_len = self.recurrent_chunk_size - seq_length % self.recurrent_chunk_size
            slen = seq_length + padding_len
            hidden_states = F.pad(hidden_states, (0, 0, 0, padding_len))
        else:
            slen = seq_length
        # relative position
        retention_rel_pos = self.retnet_rel_pos(slen,
                                                incremental_state is not None and not is_first_step,
                                                chunkwise_recurrent=self.chunkwise_recurrent)

        # start running through the decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_retentions = () if output_retentions else None
        # layers * [bsz, num_head, qk_dim, decoder_embed_dim]
        next_decoder_cache = () if use_cache else None

        for idx, layer in enumerate(self.layers):
            if incremental_state is None or is_first_step:
                if is_first_step and incremental_state is not None:
                    if idx not in incremental_state:
                        incremental_state[idx] = {}
            else:
                if idx not in incremental_state:
                    incremental_state[idx] = {}

            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs, sequence_offset, recurrent_chunk_size,
                                      output_retentions)

                    return custom_forward

                block_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    incremental_state[idx] if incremental_state is not None else None,
                    retention_rel_pos,
                    self.chunkwise_recurrent,
                    retention_mask,
                    forward_impl,
                    past_key_value,
                )
            else:
                block_outputs = layer(hidden_states,
                                      incremental_state=incremental_state[idx]
                                      if incremental_state is not None else None,
                                      retention_rel_pos=retention_rel_pos,
                                      chunkwise_recurrent=self.chunkwise_recurrent,
                                      retention_mask=retention_mask,
                                      forward_impl=forward_impl,
                                      past_key_value=past_key_value,
                                      sequence_offset=sequence_offset,
                                      recurrent_chunk_size=recurrent_chunk_size,
                                      output_retentions=output_retentions)

            hidden_states = block_outputs[0]

            if use_cache:
                next_decoder_cache += (block_outputs[1],)

            if output_retentions:
                all_retentions += (block_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if need_pad_for_chunkwise:
            hidden_states = hidden_states[:, :seq_length, :]

        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_retentions]
                         if v is not None)
        return RetNetOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            retentions=all_retentions,
        )


@dataclass
class RetNetCausalLMOutputWithPast(ModelOutput):
    """
    class for RetNet causal language model (or autoregressive) outputs.

    config:
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
            one for the output of each layer) of shape `(batch_size, sequence_length, decoder_embed_dim)`.

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


class RetNetModelWithLMHead(RetNetPreTrainedModel):

    def __init__(self, config: RetNetConfig) -> None:
        super().__init__(config)
        self.model = RetNetModel(config)
        self.lm_head = nn.Linear(config.decoder_embed_dim, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embedding

    def set_input_embeddings(self, value):
        self.model.embedding = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        retention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_retentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        forward_impl: Optional[str] = None,
        sequence_offset: Optional[int] = 0,
        recurrent_chunk_size: Optional[int] = None,
    ) -> Union[Tuple, RetNetCausalLMOutputWithPast]:
        output_retentions = output_retentions if output_retentions is not None else self.config.output_retentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else
                                self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        forward_impl = forward_impl if forward_impl is not None else self.config.forward_impl
        recurrent_chunk_size = recurrent_chunk_size if recurrent_chunk_size is not None else self.config.recurrent_chunk_size

        if retention_mask is None and attention_mask is not None:
            retention_mask = attention_mask

        outputs = self.model(input_ids,
                             retention_mask=retention_mask,
                             past_key_values=past_key_values,
                             inputs_embeds=inputs_embeds,
                             output_retentions=output_retentions,
                             output_hidden_states=output_hidden_states,
                             return_dict=return_dict,
                             forward_impl=forward_impl,
                             use_cache=use_cache,
                             sequence_offset=sequence_offset,
                             recurrent_chunk_size=recurrent_chunk_size)

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return RetNetCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            retentions=outputs.retentions,
        )

    def sample_token(self, logit, do_sample=False, top_k=1, top_p=1.0, temperature=1.0):
        if not do_sample:
            return torch.argmax(logit, dim=-1, keepdim=True)
        filtered = top_k_top_p_filtering(logit / temperature, top_k=top_k, top_p=top_p)
        return torch.multinomial(torch.softmax(filtered, dim=-1), num_samples=1)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        retention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        parallel_compute_prompt=True,
        max_new_tokens=20,
        bos_token_id=0,
        eos_token_id=0,
        do_sample=False,
        top_k=0,
        top_p=1.0,
        temperature=1.0,
        early_stopping=True,
    ):
        if retention_mask is None and attention_mask is not None:
            retention_mask = attention_mask

        generated = []
        if input_ids is not None:
            if input_ids.shape[1] == 1:
                past_key_values = None
            elif parallel_compute_prompt:
                ret_mask = retention_mask[:, :-1] if retention_mask is not None else None
                outputs = self(input_ids[:, :-1],
                               retention_mask=ret_mask,
                               forward_impl='parallel',
                               return_dict=True,
                               use_cache=True)
                past_key_values = outputs.past_key_values
            else:
                past_key_values = None
                for p_i in range(input_ids.shape[1] - 1):
                    ret_mask = retention_mask[:,
                                              p_i:p_i + 1] if retention_mask is not None else None
                    outputs = self(input_ids[:, p_i:p_i + 1],
                                   retention_mask=ret_mask,
                                   forward_impl='recurrent',
                                   past_key_values=past_key_values,
                                   sequence_offset=p_i,
                                   return_dict=True,
                                   use_cache=True)
                    past_key_values = outputs.past_key_values
            token = input_ids[:, -1].unsqueeze(-1)  # [B, 1]
            prompt_len = input_ids.shape[1] - 1
        else:
            prompt_len = 0
            token = torch.tensor([[bos_token_id]]).to(self.lm_head.weight.device)
            past_key_values = None

        for i in range(max_new_tokens):
            outputs = self(token,
                           forward_impl='recurrent',
                           past_key_values=past_key_values,
                           use_cache=True,
                           return_dict=True,
                           sequence_offset=prompt_len + i)
            logit = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            past_key_values = outputs.past_key_values
            token = self.sample_token(logit,
                                      do_sample=do_sample,
                                      top_k=top_k,
                                      top_p=top_p,
                                      temperature=temperature)
            generated.append(token)
            if early_stopping and (token == eos_token_id).all():
                break
        generated = torch.cat(generated, dim=-1)
        return generated
