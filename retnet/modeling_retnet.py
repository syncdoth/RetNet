import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

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

try:
    from apex.normalization import FusedLayerNorm as LayerNorm
except ModuleNotFoundError:
    from torch.nn import LayerNorm

from .configuration_retnet import RetNetConfig

logger = logging.get_logger(__name__)


# helper functions
def split_heads(tensors, bsz, seqlen, num_heads):
    assert isinstance(tensors, (tuple, list))
    return [x.view(bsz, seqlen, num_heads, -1).transpose(1, 2) for x in tensors]


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\


def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "swish":
        return F.silu
    else:
        raise NotImplementedError


class RetNetRelPos(nn.Module):

    def __init__(self, config: RetNetConfig):
        super().__init__()
        num_heads = config.decoder_retention_heads

        angle = 1.0 / (10000**torch.linspace(0, 1, config.decoder_embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        # decay (gamma)
        if config.use_lm_decay:
            # NOTE: alternative way described in the paper
            s = torch.log(torch.tensor(1 / 32))
            e = torch.log(torch.tensor(1 / 512))
            decay = torch.log(1 - torch.exp(torch.linspace(s, e, num_heads)))  # [h,]
        else:
            decay = torch.log(1 - 2**(-5 - torch.arange(num_heads, dtype=torch.float)))
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)
        self.recurrent_chunk_size = config.recurrent_chunk_size

    def forward(self,
                slen,
                forward_impl='parallel',
                recurrent_chunk_size=None,
                retention_mask=None):
        if forward_impl == 'recurrent':
            sin = torch.sin(self.angle * (slen - 1))
            cos = torch.cos(self.angle * (slen - 1))
            retention_rel_pos = ((sin, cos), self.decay.view(1, -1, 1, 1).exp())
        elif forward_impl == 'chunkwise':
            if recurrent_chunk_size is None:
                recurrent_chunk_size = self.recurrent_chunk_size
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])

            block_index = torch.arange(recurrent_chunk_size).to(self.decay)
            mask = torch.tril(torch.ones(recurrent_chunk_size, recurrent_chunk_size).to(self.decay))
            mask = torch.masked_fill(block_index[:, None] - block_index[None, :], ~mask.bool(),
                                     float("inf"))
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)
            mask = mask.unsqueeze(0)  # [1, h, t, t]
            # TODO: need to handle retention_mask
            # scaling
            # TODO: removing scaling from chunkwise and parallel makes them equivalent.
            # but scaling is required for recurrent.
            scale = mask.sum(dim=-1, keepdim=True).sqrt()
            mask = mask / scale

            cross_decay = torch.exp(self.decay * recurrent_chunk_size)
            inner_decay = torch.exp(self.decay[:, None] * (block_index + 1))
            cross_decay = cross_decay[None, :, None, None]
            inner_decay = inner_decay[None, :, :, None] / (scale / scale[:, :, -1, None])
            # decay_scale (used for kv cache)
            decay_scale = self.compute_decay_scale(slen, retention_mask)
            retention_rel_pos = ((sin, cos), (mask, cross_decay, inner_decay, decay_scale))
        else:  # parallel
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])
            mask = torch.tril(torch.ones(slen, slen).to(self.decay))
            mask = torch.masked_fill(index[:, None] - index[None, :], ~mask.bool(), float("inf"))
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)
            mask = mask.unsqueeze(0)  # [1, h, t, t]
            if retention_mask is not None:
                # this is required for left padding
                mask = mask * retention_mask.float().view(-1, 1, 1, slen)

            # scaling
            mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
            mask = torch.nan_to_num(mask, nan=0.0)
            # decay_scale (used for kv cache)
            decay_scale = self.compute_decay_scale(slen, retention_mask)
            # mask processing for intra decay
            if retention_mask is not None:
                max_non_zero = torch.cumsum(retention_mask, dim=-1).max(dim=-1).indices  # [b,]
                intra_decay = mask[range(mask.shape[0]), :, max_non_zero]
            else:
                intra_decay = mask[:, :, -1]

            retention_rel_pos = ((sin, cos), (mask, intra_decay, decay_scale))

        return retention_rel_pos

    def compute_decay_scale(self, slen, retention_mask=None):
        exponent = torch.arange(slen, device=self.decay.device).float()
        decay_scale = self.decay.exp().view(-1, 1)**exponent.view(1, -1)
        if retention_mask is not None:
            seqlen = retention_mask.sum(dim=-1)  # [b,]
            for pos in seqlen:
                # the formula for decay_scale is `sum(gamma^i) for i in [0, slen).`
                # Since the retention_mask is 0 for padding, we can set the decay_scale
                # to 0 for the padding positions.
                decay_scale[:, pos.item():] = 0
        decay_scale = decay_scale.sum(-1).view(1, -1, 1, 1)
        return decay_scale


class MultiScaleRetention(nn.Module):

    def __init__(
        self,
        config: RetNetConfig,
        value_factor: int = 2,
        gate_fn="swish",
    ):
        super().__init__()
        self.config = config
        self.factor = value_factor
        self.embed_dim = config.decoder_embed_dim
        self.v_dim = config.decoder_embed_dim * value_factor
        self.num_heads = config.decoder_retention_heads
        self.key_dim = self.embed_dim // self.num_heads
        self.head_dim = self.v_dim // self.num_heads
        self.scaling = self.key_dim**-0.5

        self.gate_fn = get_activation_fn(activation=str(gate_fn))

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.v_dim, bias=True)
        self.g_proj = nn.Linear(self.embed_dim, self.v_dim, bias=True)

        self.out_proj = nn.Linear(self.v_dim, self.embed_dim, bias=True)

        self.group_norm = LayerNorm(self.head_dim, eps=1e-6, elementwise_affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.g_proj.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def parallel_retention(self, q, k, v, decay_mask):
        """
        q,  # bsz * num_head * len * qk_dim
        k,  # bsz * num_head * len * qk_dim
        v,  # bsz * num_head * len * v_dim
        decay_mask,  # (1 or bsz) * num_head * len * len
        """
        decay_mask, intra_decay, scale = decay_mask
        # [b, h, t, t]
        retention = q @ k.transpose(-1, -2)  # (scaled dot-product)
        retention = retention * decay_mask

        # invariant after normalization
        retention = retention / retention.detach().sum(dim=-1, keepdim=True).abs().clamp(min=1)

        output = retention @ v  # [b, h, t, v_dim / h]
        output = output.transpose(1, 2)  # [b, t, h, v_dim / h]

        # kv cache: [b, h, t, v_dim, qk_dim]
        current_kv = k.unsqueeze(-2) * v.unsqueeze(-1)
        intra_decay = intra_decay[:, :, :, None, None]  # [b, h, t, 1, 1]
        current_kv = (current_kv * intra_decay).sum(2)  # [b, h, v_dim, qk_dim]

        cache = {"prev_key_value": current_kv, "scale": scale}
        return output, cache, retention

    def recurrent_retention(self, q, k, v, decay, past_key_value=None, retention_mask=None):
        """
        q, k, v, # bsz * num_head * 1 * qkv_dim
        past_key_value:
            - "prev_key_value"  # bsz * num_head * v_dim * qk_dim
            - "scale"  # (1 or bsz) * num_head * 1 * 1
        decay # (1 or bsz) * num_head * 1 * 1
        retention_mask # bsz * 1
        """
        if retention_mask is not None:
            retention_mask = retention_mask.float().view(-1, 1, 1, 1)
        else:
            retention_mask = torch.ones(k.size(0), 1, 1, 1).to(decay)
        # (b, h, v_dim, qk_dim)
        current_kv = k * v.transpose(-1, -2) * retention_mask

        if past_key_value is not None and "prev_key_value" in past_key_value:
            prev_kv = past_key_value["prev_key_value"]
            prev_scale = past_key_value["scale"]
            scale = torch.where(retention_mask == 0, prev_scale, prev_scale * decay + 1)
            # connect prev_kv and current_kv
            # how much to decay prev_kv
            decay_amount = prev_scale.sqrt() * decay / scale.sqrt()
            decay_amount = torch.where(retention_mask == 0, 1, decay_amount)
            prev_kv *= decay_amount  # decay prev_kv
            current_kv /= scale.sqrt()  # scale current_kv
            current_kv = torch.nan_to_num(current_kv, nan=0.0)  # remove nan, scale might be 0

            current_kv = prev_kv + current_kv
        else:
            scale = torch.ones_like(decay)
            # when retention_mask is 0 at the beginning, setting scale to 1 will
            # make the first retention to use the padding incorrectly. Hence,
            # setting it to 0 here. This is a little ugly, so we might want to
            # change this later. TODO: improve
            scale = torch.where(retention_mask == 0, torch.zeros_like(decay), scale)

        output = torch.sum(q * current_kv, dim=3).unsqueeze(1)  # (b, 1, h, d_v)

        cache = {"prev_key_value": current_kv, "scale": scale}
        return output, cache

    def chunkwise_retention(self, q, k, v, decay_mask):
        """
        q, k, v,  # bsz * num_head * seqlen * qkv_dim
        past_key_value:
            - "prev_key_value"  # bsz * num_head * v_dim * qk_dim
            - "scale"  # (1 or bsz) * num_head * 1 * 1
        decay_mask,  # 1 * num_head * chunk_size * chunk_size
        chunk_decay,  # 1 * num_head * 1 * 1
        inner_decay,  # 1 * num_head * chunk_size * 1
        """
        # TODO: not working properly
        decay_mask, chunk_decay, inner_decay, decay_scale = decay_mask
        bsz, _, tgt_len, _ = v.size()
        chunk_len = decay_mask.size(-1)
        assert tgt_len % chunk_len == 0
        num_chunks = tgt_len // chunk_len

        # [b, n_c, h, t_c, qkv_dim]
        q = q.view(bsz, self.num_heads, num_chunks, chunk_len, self.key_dim).transpose(1, 2)
        k = k.view(bsz, self.num_heads, num_chunks, chunk_len, self.key_dim).transpose(1, 2)
        v = v.view(bsz, self.num_heads, num_chunks, chunk_len, self.head_dim).transpose(1, 2)

        k_t = k.transpose(-1, -2)

        qk_mat = q @ k_t  # [b, n_c, h, t_c, t_c]
        qk_mat = qk_mat * decay_mask.unsqueeze(1)
        inner_scale = qk_mat.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1)
        qk_mat = qk_mat / inner_scale
        # [b, n_c, h, t_c, v_dim]
        inner_output = torch.matmul(qk_mat, v)

        # reduce kv in one chunk
        # [b, n_c, h, qk_dim, v_dim]
        kv = k_t @ (v * decay_mask[:, None, :, -1, :, None])
        kv = kv.view(bsz, num_chunks, self.num_heads, self.key_dim, self.head_dim)

        kv_recurrent = []
        cross_scale = []
        kv_state = torch.zeros(bsz, self.num_heads, self.key_dim, self.head_dim).to(v)
        kv_scale = torch.ones(bsz, self.num_heads, 1, 1).to(v)

        # accumulate kv by loop
        for i in range(num_chunks):
            kv_recurrent.append(kv_state / kv_scale)
            cross_scale.append(kv_scale)
            kv_state = kv_state * chunk_decay + kv[:, i]
            kv_scale = kv_state.detach().abs().sum(dim=-2, keepdim=True).max(
                dim=-1, keepdim=True).values.clamp(min=1)

        kv_recurrent = torch.stack(kv_recurrent, dim=1)
        cross_scale = torch.stack(cross_scale, dim=1)

        all_scale = torch.maximum(inner_scale, cross_scale)
        align_inner_scale = all_scale / inner_scale
        align_cross_scale = all_scale / cross_scale

        cross_output = (q * inner_decay.unsqueeze(1)) @ kv_recurrent
        output = inner_output / align_inner_scale + cross_output / align_cross_scale
        output = output.transpose(2, 3)  # [b, n_c, t_c, h, v_dim]

        cache = {"prev_key_value": kv_state.transpose(-2, -1), "scale": decay_scale}
        return output, cache

    def forward(
        self,
        hidden_states: torch.Tensor,
        rel_pos: Tuple[Tuple[torch.Tensor]],
        retention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        forward_impl: str = 'parallel',
        output_retentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor]]:
        B, T, H = hidden_states.size()
        (sin, cos), decay_mask = rel_pos
        # projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        g = self.g_proj(hidden_states)
        # multi-head
        q, k, v = split_heads((q, k, v), B, T, self.num_heads)
        k = k * self.scaling  # for scaled dot product
        # rotate
        # NOTE: theta_shift has bug with mps device.
        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        # retention
        if forward_impl == 'parallel':
            retention_out, curr_kv, retention_weights = self.parallel_retention(
                qr, kr, v, decay_mask)
        elif forward_impl == 'recurrent':
            retention_out, curr_kv = self.recurrent_retention(qr,
                                                              kr,
                                                              v,
                                                              decay_mask,
                                                              past_key_value=past_key_value,
                                                              retention_mask=retention_mask)
        elif forward_impl == 'chunkwise':
            retention_out, curr_kv = self.chunkwise_retention(qr, kr, v, decay_mask)
        else:
            raise ValueError(f'forward_impl {forward_impl} not supported.')

        # concaat heads
        normed = self.group_norm(retention_out).reshape(B, T, self.v_dim)
        # out gate & proj
        out = self.gate_fn(g) * normed
        out = self.out_proj(out)

        outputs = (out, curr_kv)
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
        self.ffn_layernorm = LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None

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
            value_factor=config.value_factor,
        )

        self.normalize_before = config.decoder_normalize_before

        self.retention_layer_norm = LayerNorm(self.embed_dim, eps=config.layernorm_eps)

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

        self.final_layer_norm = LayerNorm(self.embed_dim, eps=config.layernorm_eps)

        if config.deepnorm:
            self.alpha = math.pow(2.0 * config.decoder_layers, 0.25)
        else:
            self.alpha = 1.0

    def residual_connection(self, x, residual):
        return residual * self.alpha + x

    def forward(
        self,
        hidden_states: torch.Tensor,
        retention_rel_pos: Tuple[Tuple[torch.Tensor]],
        retention_mask: Optional[torch.Tensor] = None,
        forward_impl: str = 'parallel',
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_retentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor]]:
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.retention_layer_norm(hidden_states)

        msr_outs = self.retention(hidden_states,
                                  retention_rel_pos,
                                  retention_mask=retention_mask,
                                  past_key_value=past_key_value,
                                  forward_impl=forward_impl,
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
        past_key_values (`List(Dict(str, torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            - "prev_key_value": shape=(bsz * num_head * v_dim * qk_dim)
            - "scale": shape=((1 or bsz) * num_head * 1 * 1)

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
    past_key_values: Optional[List[Dict[str, torch.FloatTensor]]] = None
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
            self.layernorm_embedding = LayerNorm(self.embed_dim, eps=config.layernorm_eps)
        else:
            self.layernorm_embedding = None

        self.layers = nn.ModuleList([])

        for i in range(config.decoder_layers):
            self.layers.append(RetNetDecoderLayer(config, depth=i))

        self.decoder_layers = len(self.layers)

        if config.decoder_normalize_before:
            self.layer_norm = LayerNorm(self.embed_dim, eps=config.layernorm_eps)
        else:
            self.layer_norm = None

        self.retnet_rel_pos = RetNetRelPos(config)
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
        past_key_values=None,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        embed = self.embed_scale * inputs_embeds

        if self.layernorm_embedding is not None:
            embed = self.layernorm_embedding(embed)

        embed = self.dropout_module(embed)

        return embed

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        retention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Dict[str, torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_retentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        forward_impl: Optional[str] = 'parallel',
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

        # embed tokens
        if inputs_embeds is None:
            inputs_embeds = self.forward_embedding(input_ids, inputs_embeds, past_key_values)

        if retention_mask is None:
            if attention_mask is not None:
                retention_mask = attention_mask

        hidden_states = inputs_embeds

        # handling chunking here
        if recurrent_chunk_size is None:
            recurrent_chunk_size = self.recurrent_chunk_size
        need_pad_for_chunkwise = (forward_impl == 'chunkwise' and
                                  seq_length % recurrent_chunk_size != 0)
        if need_pad_for_chunkwise:
            padding_len = recurrent_chunk_size - seq_length % recurrent_chunk_size
            slen = seq_length + padding_len
            hidden_states = F.pad(hidden_states, (0, 0, 0, padding_len))
        else:
            slen = seq_length
        # relative position
        retention_rel_pos = self.retnet_rel_pos(slen,
                                                forward_impl=forward_impl,
                                                recurrent_chunk_size=recurrent_chunk_size,
                                                retention_mask=retention_mask)

        # start running through the decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_retentions = () if output_retentions else None
        # layers * [bsz, num_head, qk_dim, decoder_embed_dim]
        next_decoder_cache = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs, output_retentions)

                    return custom_forward

                block_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    retention_rel_pos,
                    retention_mask,
                    forward_impl,
                    past_key_value,
                )
            else:
                block_outputs = layer(hidden_states,
                                      retention_rel_pos,
                                      retention_mask=retention_mask,
                                      forward_impl=forward_impl,
                                      past_key_value=past_key_value,
                                      output_retentions=output_retentions)

            hidden_states = block_outputs[0]

            if use_cache:
                next_decoder_cache.append(block_outputs[1])

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
        past_key_values (`List(Dict(str, torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            - "prev_key_value": shape=(bsz * num_head * v_dim * qk_dim)
            - "scale": shape=((1 or bsz) * num_head * 1 * 1)

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
    past_key_values: Optional[List[Dict[str, torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    retentions: Optional[Tuple[torch.FloatTensor]] = None


class RetNetModelWithLMHead(RetNetPreTrainedModel):

    def __init__(self, config: RetNetConfig, embed_tokens: nn.Embedding = None) -> None:
        super().__init__(config)
        self.model = RetNetModel(config, embed_tokens=embed_tokens)
        self.lm_head = nn.Linear(config.decoder_embed_dim, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

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
                    outputs = self(input_ids[:, :p_i + 1],
                                   retention_mask=ret_mask,
                                   forward_impl='recurrent',
                                   past_key_values=past_key_values,
                                   return_dict=True,
                                   use_cache=True)
                    past_key_values = outputs.past_key_values

            generated = input_ids[:, -1].unsqueeze(-1)  # [B, 1]
        else:
            generated = torch.tensor([[bos_token_id]]).to(self.lm_head.weight.device)
            past_key_values = None

        for i in range(max_new_tokens):
            outputs = self(generated,
                           forward_impl='recurrent',
                           past_key_values=past_key_values,
                           use_cache=True,
                           return_dict=True)
            logit = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            past_key_values = outputs.past_key_values
            token = self.sample_token(logit,
                                      do_sample=do_sample,
                                      top_k=top_k,
                                      top_p=top_p,
                                      temperature=temperature)
            generated = torch.cat([generated, token], dim=-1)
            if early_stopping and (token == eos_token_id).all():
                break
        return generated
