# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from fairscale.nn import checkpoint_wrapper, wrap
from timm.models.layers import drop_path
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from apex.normalization import FusedLayerNorm as LayerNorm
except ModuleNotFoundError:
    from torch.nn import LayerNorm


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


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output


class RetNetRelPos(nn.Module):

    def __init__(self, config):
        super().__init__()
        num_heads = config.decoder_retention_heads

        angle = 1.0 / (10000**torch.linspace(0, 1, config.decoder_embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
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

    def forward(self, slen, activate_recurrent=False, chunkwise_recurrent=False):
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen - 1))
            cos = torch.cos(self.angle * (slen - 1))
            retention_rel_pos = ((sin, cos), self.decay.exp())
        elif chunkwise_recurrent:
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])

            block_index = torch.arange(self.recurrent_chunk_size).to(self.decay)
            mask = torch.tril(torch.ones(self.recurrent_chunk_size,
                                         self.recurrent_chunk_size)).to(self.decay)
            mask = torch.masked_fill(block_index[:, None] - block_index[None, :], ~mask.bool(),
                                     float("inf"))
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)

            value_inner_decay = mask[:, -1] / mask[:, -1].sum(dim=-1, keepdim=True)
            value_inner_decay = value_inner_decay.unsqueeze(-1)
            scale = mask.sum(dim=-1, keepdim=True).sqrt()
            inner_mask = mask / scale

            cross_decay = torch.exp(self.decay * self.recurrent_chunk_size)
            query_inner_decay = torch.exp(self.decay[:, None] * (block_index + 1))
            query_inner_decay = query_inner_decay[:, :, None] / (
                scale / mask[:, -1].sum(dim=-1)[:, None, None])
            cross_decay = cross_decay[:, None, None]
            retention_rel_pos = ((sin, cos), (inner_mask, cross_decay, query_inner_decay,
                                              value_inner_decay))
        else:
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])
            mask = torch.tril(torch.ones(slen, slen)).to(self.decay)
            mask = torch.masked_fill(index[:, None] - index[None, :], ~mask.bool(), float("inf"))
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)
            mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos


class MultiScaleRetention(nn.Module):

    def __init__(
        self,
        config,
        embed_dim,
        value_dim,
        num_heads,
        gate_fn="swish",
    ):
        super().__init__()
        self.config = config
        self.embed_dim = embed_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.head_dim = self.value_dim // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim**-0.5

        self.gate_fn = get_activation_fn(activation=str(gate_fn))

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, value_dim, bias=False)
        self.g_proj = nn.Linear(embed_dim, value_dim, bias=False)

        self.out_proj = nn.Linear(value_dim, embed_dim, bias=False)

        self.group_norm = RMSNorm(self.head_dim, eps=config.layernorm_eps, elementwise_affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.g_proj.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def parallel_forward(self, qr, kr, v, mask):
        bsz, tgt_len, embed_dim = v.size()

        vr = v.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        qk_mat = qr @ kr.transpose(-1, -2)  # bsz * m * tgt_len * tgt_len
        qk_mat = qk_mat * mask
        # invariant after normalization
        qk_mat = qk_mat / qk_mat.detach().sum(dim=-1, keepdim=True).abs().clamp(min=1)
        output = torch.matmul(qk_mat, vr)
        output = output.transpose(1, 2)
        return output

    def recurrent_forward(self, qr, kr, v, decay, incremental_state):
        bsz = v.size(0)

        v = v.view(bsz, self.num_heads, self.head_dim, 1)
        kv = kr * v
        if "prev_key_value" in incremental_state:
            prev_kv = incremental_state["prev_key_value"]
            prev_scale = incremental_state["scale"]
            scale = prev_scale * decay + 1
            kv = prev_kv * (prev_scale.sqrt() * decay / scale.sqrt()).view(
                self.num_heads, 1, 1) + kv / scale.sqrt().view(self.num_heads, 1, 1)
            # kv = prev_kv * decay.view(self.num_heads, 1, 1) + kv
        else:
            scale = torch.ones_like(decay)

        incremental_state["prev_key_value"] = kv
        incremental_state["scale"] = scale

        output = torch.sum(qr * kv, dim=3)
        return output

    def chunk_recurrent_forward(self, qr, kr, v, inner_mask):
        mask, cross_decay, query_inner_decay, value_inner_decay = inner_mask
        bsz, tgt_len, embed_dim = v.size()
        chunk_len = mask.size(1)
        num_chunks = tgt_len // chunk_len

        assert tgt_len % chunk_len == 0

        qr = qr.view(bsz, self.num_heads, num_chunks, chunk_len, self.key_dim).transpose(1, 2)
        kr = kr.view(bsz, self.num_heads, num_chunks, chunk_len, self.key_dim).transpose(1, 2)
        v = v.view(bsz, num_chunks, chunk_len, self.num_heads, self.head_dim).transpose(2, 3)

        kr_t = kr.transpose(-1, -2)

        qk_mat = qr @ kr_t  # bsz * num_heads * chunk_len * chunk_len
        qk_mat = qk_mat * mask
        inner_scale = qk_mat.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1)
        qk_mat = qk_mat / inner_scale
        inner_output = torch.matmul(qk_mat,
                                    v)  # bsz * num_heads * num_value_heads * chunk_len * head_dim

        # reduce kv in one chunk
        kv = kr_t @ (v * value_inner_decay)

        kv_recurrent = []
        cross_scale = []
        kv_state = torch.zeros(bsz, self.num_heads, self.key_dim, self.head_dim).to(v)
        kv_scale = torch.ones(bsz, self.num_heads, 1, 1).to(v)

        # accumulate kv by loop
        for i in range(num_chunks):
            kv_recurrent.append(kv_state / kv_scale)
            cross_scale.append(kv_scale)
            kv_state = kv_state * cross_decay + kv[:, i]
            kv_scale = kv_state.detach().abs().sum(dim=-2, keepdim=True).max(
                dim=-1, keepdim=True).values.clamp(min=1)

        kv_recurrent = torch.stack(kv_recurrent, dim=1)
        cross_scale = torch.stack(cross_scale, dim=1)

        all_scale = torch.maximum(inner_scale, cross_scale)
        align_inner_scale = all_scale / inner_scale
        align_cross_scale = all_scale / cross_scale

        cross_output = (qr * query_inner_decay) @ kv_recurrent
        output = inner_output / align_inner_scale + cross_output / align_cross_scale
        # output = inner_output / cross_scale + cross_output / inner_scale

        output = output.transpose(2, 3)
        return output

    def forward(self, x, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        bsz, tgt_len, _ = x.size()
        (sin, cos), inner_mask = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        g = self.g_proj(x)

        k *= self.scaling
        q = q.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        k = k.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)

        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        if incremental_state is not None:
            output = self.recurrent_forward(qr, kr, v, inner_mask, incremental_state)
        elif chunkwise_recurrent:
            output = self.chunk_recurrent_forward(qr, kr, v, inner_mask)
        else:
            output = self.parallel_forward(qr, kr, v, inner_mask)

        output = self.group_norm(output).reshape(bsz, tgt_len, self.head_dim * self.num_heads)

        output = self.gate_fn(g) * output

        output = self.out_proj(output)

        return output


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
        use_rms_norm=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = get_activation_fn(activation=str(activation_fn))
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        if subln:
            if use_rms_norm:
                self.ffn_layernorm = RMSNorm(self.embed_dim, eps=layernorm_eps)
            else:
                self.ffn_layernorm = LayerNorm(self.embed_dim, eps=layernorm_eps)
        else:
            self.ffn_layernorm = None

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


class GLU(nn.Module):

    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn,
        dropout,
        activation_dropout,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = get_activation_fn(activation=str(activation_fn))
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim, bias=False)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim, bias=False)
        self.gate = nn.Linear(self.embed_dim, ffn_dim, bias=False)

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.gate.reset_parameters()

    def forward(self, x):
        x_shape = x.shape
        x = x.reshape(-1, x.size(-1))
        g = self.gate(x)
        x = self.fc1(x)
        x = self.activation_fn(x.float()).type_as(x) * g
        x = self.activation_dropout_module(x)
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

    def __init__(
        self,
        config,
        depth,
    ):
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
            self.embed_dim,
            config.decoder_value_embed_dim,
            config.decoder_retention_heads,
        )

        self.normalize_before = config.decoder_normalize_before

        self.retention_layer_norm = RMSNorm(self.embed_dim, eps=config.layernorm_eps)

        self.ffn_dim = config.decoder_ffn_embed_dim

        self.ffn = self.build_ffn()

        self.final_layer_norm = RMSNorm(self.embed_dim, eps=config.layernorm_eps)

        if config.deepnorm:
            self.alpha = math.pow(2.0 * config.decoder_layers, 0.25)
        else:
            self.alpha = 1.0

    def build_ffn(self):
        if self.config.use_glu:
            return GLU(
                self.embed_dim,
                self.ffn_dim,
                self.config.activation_fn,
                self.config.dropout,
                self.config.activation_dropout,
            )
        else:
            return FeedForwardNetwork(
                self.embed_dim,
                self.ffn_dim,
                self.config.activation_fn,
                self.config.dropout,
                self.config.activation_dropout,
                self.config.layernorm_eps,
                self.config.subln,
                self.config.use_ffn_rms_norm,
            )

    def residual_connection(self, x, residual):
        return residual * self.alpha + x

    def forward(
        self,
        x,
        incremental_state=None,
        chunkwise_recurrent=False,
        retention_rel_pos=None,
    ):
        residual = x
        if self.normalize_before:
            x = self.retention_layer_norm(x)

        x = self.retention(
            x,
            incremental_state=incremental_state,
            rel_pos=retention_rel_pos,
            chunkwise_recurrent=chunkwise_recurrent,
        )
        x = self.dropout_module(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.retention_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.ffn(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x


class RetNetModel(nn.Module):

    def __init__(self, config, embed_tokens=None, output_projection=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.dropout_module = torch.nn.Dropout(config.dropout)

        embed_dim = config.decoder_embed_dim
        self.embed_dim = embed_dim
        self.embed_scale = 1.0 if config.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_tokens = embed_tokens

        if (output_projection is None and not config.no_output_layer and config.vocab_size > 0):
            self.output_projection = self.build_output_projection(config)
        else:
            self.output_projection = output_projection

        if config.layernorm_embedding:
            self.layernorm_embedding = RMSNorm(embed_dim, eps=config.layernorm_eps)
        else:
            self.layernorm_embedding = None

        self.layers = nn.ModuleList([])

        for i in range(config.decoder_layers):
            self.layers.append(self.build_decoder_layer(
                config,
                depth=i,
            ))

        self.num_layers = len(self.layers)

        if config.decoder_normalize_before:
            self.layer_norm = RMSNorm(embed_dim, eps=config.layernorm_eps)
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

        if config.subln and not config.use_glu:
            init_scale = math.sqrt(math.log(config.decoder_layers * 2))
            for name, p in self.named_parameters():
                if ("fc1" in name or "fc2" in name or "out_proj" in name or "v_proj" in name):
                    p.data.mul_(init_scale)

    def build_output_projection(
        self,
        config,
    ):
        if config.share_decoder_input_output_embed:
            output_projection = torch.nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            output_projection.weight = self.embed_tokens.weight
        else:
            output_projection = torch.nn.Linear(config.decoder_embed_dim,
                                                config.vocab_size,
                                                bias=False)
            torch.nn.init.normal_(output_projection.weight,
                                  mean=0,
                                  std=config.decoder_embed_dim**-0.5)
        return output_projection

    def build_decoder_layer(self, config, depth):
        layer = RetNetDecoderLayer(
            config,
            depth,
        )
        # if config.checkpoint_activations:
        #     layer = checkpoint_wrapper(layer)
        # if config.fsdp:
        #     layer = wrap(layer)
        return layer

    def forward_embedding(
        self,
        tokens,
        token_embedding=None,
        incremental_state=None,
    ):
        if incremental_state is not None and not self.is_first_step(incremental_state):
            tokens = tokens[:, -1:]

        if token_embedding is None:
            token_embedding = self.embed_tokens(tokens)

        x = embed = self.embed_scale * token_embedding

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        return x, embed

    def is_first_step(self, incremental_state):
        if incremental_state is None:
            return False
        return incremental_state.get("is_first_step", False)

    def forward(self,
                prev_output_tokens,
                incremental_state=None,
                features_only=False,
                token_embeddings=None):
        # embed tokens
        x, _ = self.forward_embedding(prev_output_tokens, token_embeddings, incremental_state)
        is_first_step = self.is_first_step(incremental_state)

        if self.chunkwise_recurrent and prev_output_tokens.size(1) % self.recurrent_chunk_size != 0:
            padding_len = self.recurrent_chunk_size - prev_output_tokens.size(
                1) % self.recurrent_chunk_size
            slen = prev_output_tokens.size(1) + padding_len
            x = F.pad(x, (0, 0, 0, padding_len))
        else:
            slen = prev_output_tokens.size(1)
        # relative position
        retention_rel_pos = self.retnet_rel_pos(slen,
                                                incremental_state is not None and not is_first_step,
                                                chunkwise_recurrent=self.chunkwise_recurrent)

        # decoder layers
        inner_states = [x]

        for idx, layer in enumerate(self.layers):
            if incremental_state is None or is_first_step:
                if is_first_step and incremental_state is not None:
                    if idx not in incremental_state:
                        incremental_state[idx] = {}
            else:
                if idx not in incremental_state:
                    incremental_state[idx] = {}

            x = layer(
                x,
                incremental_state[idx] if incremental_state is not None else None,
                retention_rel_pos=retention_rel_pos,
                chunkwise_recurrent=self.chunkwise_recurrent,
            )
            inner_states.append(x)

        if self.chunkwise_recurrent and prev_output_tokens.size(1) % self.recurrent_chunk_size != 0:
            x = x[:, :prev_output_tokens.size(1), :]

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if not features_only:
            x = self.output_layer(x)

        return x, {
            "inner_states": inner_states,
            "attn": None,
        }

    def output_layer(self, features):
        return self.output_projection(features)


class RetNetForCausalLM(nn.Module):

    def __init__(self, config, embed_tokens=None, output_projection=None, **kwargs):
        super().__init__(**kwargs)
        assert config.vocab_size > 0, "you must specify vocab size"
        if output_projection is None:
            config.no_output_layer = False
        if embed_tokens is None:
            embed_tokens = nn.Embedding(config.vocab_size, config.decoder_embed_dim,
                                        config.pad_token_id)

        self.config = config
        self.model = RetNetModel(config,
                                 embed_tokens=embed_tokens,
                                 output_projection=output_projection,
                                 **kwargs)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.model.output_projection

    def set_output_embeddings(self, new_embeddings):
        self.model.output_projection = new_embeddings

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
        recurrent_chunk_size: Optional[int] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        outputs = self.model(
            input_ids,
            incremental_state=past_key_values,
            features_only=False,
            token_embeddings=inputs_embeds,
        )

        logits, inner_hidden_states = outputs[0], outputs[1]['inner_states']

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

            if self.config.z_loss_coeff > 0:
                # z_loss from PaLM paper
                # z_loss = 1e-4 * log(log(z)), where z = sum(exp(logits))
                z_loss = torch.logsumexp(shift_logits, dim=-1).log().mean()
                loss += self.config.z_loss_coeff * z_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=inner_hidden_states,
            attentions=None,
        )
