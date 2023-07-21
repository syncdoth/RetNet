from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import top_k_top_p_filtering

from utils import split_chunks, split_heads
from xpos_relative_position import XPOS


@dataclass
class RetNetConfig:
    num_layers: int
    vocab_size: int
    hidden_size: int
    num_heads: int
    qk_dim: int = None
    v_dim: int = None
    ffn_proj_size: int = None
    chunk_size: int = None
    use_bias_in_msr: bool = False
    use_bias_in_mlp: bool = True
    use_bias_in_msr_out: bool = True
    use_default_gamma: bool = False
    tie_weights: bool = False

    def __post_init__(self):
        if self.ffn_proj_size is None:
            self.ffn_proj_size = self.hidden_size * 2
        if self.qk_dim is None:
            self.qk_dim = self.hidden_size
        if self.v_dim is None:
            self.v_dim = self.hidden_size * 2


class MultiScaleRetention(nn.Module):
    # TODO: normalization to decay in the paper
    def __init__(self, config: RetNetConfig):
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.hidden_size,
                             config.qk_dim * 2 + config.v_dim,
                             bias=config.use_bias_in_msr)
        self.silu = nn.SiLU()
        self.gated = nn.Linear(config.hidden_size, config.v_dim, bias=False)
        self.proj = nn.Linear(config.v_dim, config.hidden_size, bias=config.use_bias_in_msr_out)
        self.gn = nn.GroupNorm(num_groups=config.num_heads, num_channels=config.v_dim, affine=False)
        self.xpos = XPOS(config.qk_dim)

        # initialize gamma
        if config.use_default_gamma:
            gamma = 1 - 2**(-5 - torch.arange(0, config.num_heads, dtype=torch.float))
        else:
            s = torch.log(torch.tensor(1 / 32))
            e = torch.log(torch.tensor(1 / 512))
            gamma = 1 - torch.exp(torch.linspace(s, e, config.num_heads))  # [h,]
        self.decay = nn.Parameter(gamma, requires_grad=False)

    def get_parallel_decay_mask(self, length):
        range_tensor = torch.arange(length, device=self.decay.device)
        range_tensor = range_tensor[None, :, None].expand(self.config.num_heads, length, 1)
        exponent = range_tensor - range_tensor.transpose(-1, -2)
        decay_mask = self.decay.view(-1, 1, 1)**exponent
        decay_mask = torch.tril(decay_mask, diagonal=0)
        return decay_mask

    def get_recurrent_decay(self):
        return self.decay.view(1, self.config.num_heads, 1, 1)

    def get_chunkwise_decay(self, chunk_size):
        # within chunk decay
        decay_mask = self.get_parallel_decay_mask(chunk_size)
        # decay of the chunk
        chunk_decay = self.decay.view(1, self.config.num_heads, 1, 1)**chunk_size
        # cross-chunk decay
        exponent = torch.arange(chunk_size, dtype=torch.float,
                                device=decay_mask.device).unsqueeze(0) + 1
        inner_decay = (self.decay.unsqueeze(-1)**exponent).view(1, self.config.num_heads,
                                                                chunk_size, 1)
        return decay_mask, chunk_decay, inner_decay

    def parallel_retention(self, q, k, v, decay_mask):
        """
        q,  # bsz * num_head * len * qk_dim
        k,  # bsz * num_head * len * qk_dim
        v,  # bsz * num_head * len * v_dim
        decay_mask,  # num_head * len * len
        """
        # [b, h, t, t]
        retention = q @ k.transpose(-1, -2) * k.size(-1)**-0.5  # (scaled dot-product)
        retention = retention * decay_mask
        output = retention @ v

        # kv cache
        current_kv = k.unsqueeze(-1) * v.unsqueeze(-2)
        # [bsz, num_head, qk_dim, v_dim]
        intra_decay = decay_mask[None, :, -1, :, None, None]
        current_kv = (current_kv * intra_decay).sum(2)
        return output, current_kv

    def recurrent_retention(self, q, k, v, past_kv=None, decay=None):
        """
        q, k, v, # bsz * num_head * 1 * qkv_dim
        past_kv, # bsz * num_head * qk_dim * v_dim
        decay # num_head * 1 * 1
        """
        past_kv = past_kv if past_kv is not None else 0
        decay = decay if decay is not None else 0
        current_kv = decay * past_kv + k.transpose(-1, -2) @ v  # (b, h, d_k, d_v)
        output = q @ current_kv * k.size(-1)**-0.5  # (b, h, 1, d_v)
        return output, current_kv

    def chunkwise_retention(self,
                            q,
                            k,
                            v,
                            decay_mask,
                            past_kv=None,
                            chunk_decay=None,
                            inner_decay=None):
        """
        q, k, v,  # bsz * num_head * chunk_size * qkv_dim
        past_kv,  # bsz * num_head * qk_dim * v_dim
        decay_mask,  # 1 * num_head * chunk_size * chunk_size
        chunk_decay,  # 1 * num_head * 1 * 1
        inner_decay,  # 1 * num_head * chunk_size * 1
        """
        # [bsz, num_head, chunk_size, chunk_size]
        retention = q @ k.transpose(-1, -2) * k.size(-1)**-0.5
        retention = retention * decay_mask
        inner_retention = retention @ v  # [bsz, num_head, chunk_size, v_dim]

        if past_kv is None:
            cross_retention = 0
            past_chunk = 0
        else:
            cross_retention = (q @ past_kv) * inner_decay * k.size(-1)**-0.5
            past_chunk = chunk_decay * past_kv

        # [bsz, num_head, chunk_size, v_dim]
        retention = inner_retention + cross_retention
        # [bsz, num_head, chunk_size, qk_dim, v_dim]
        current_kv = k.unsqueeze(-1) * v.unsqueeze(-2)
        # NOTE: intra_decay is omitted in the paper; but this detail is important
        # [bsz, num_head, qk_dim, v_dim]
        intra_decay = decay_mask[None, :, -1, :, None, None]
        current_kv = (current_kv * intra_decay).sum(2)
        current_kv = past_chunk + current_kv
        return retention, current_kv

    def forward(self, x, past_kv=None, forward_impl='parallel', sequence_offset=0):
        B, T, H = x.size()
        q, k, v = self.qkv(x).split([self.config.qk_dim, self.config.qk_dim, self.config.v_dim],
                                    dim=-1)
        q, k = self.xpos.rotate_queries_and_keys(q, k, offset=sequence_offset)
        q, k, v = split_heads((q, k, v), B, T, self.config.num_heads)
        # retention
        if forward_impl == 'parallel':
            decay_mask = self.get_parallel_decay_mask(T)
            retention_out, curr_kv = self.parallel_retention(q, k, v, decay_mask)
        elif forward_impl == 'recurrent':
            decay = self.get_recurrent_decay()
            retention_out, curr_kv = self.recurrent_retention(q, k, v, past_kv=past_kv, decay=decay)
        elif forward_impl == 'chunkwise':
            assert self.config.chunk_size is not None
            q_chunks, k_chunks, v_chunks = split_chunks(q, k, v, size=self.config.chunk_size, dim=2)
            ret_chunks = []
            for _, (_q, _k, _v) in enumerate(zip(q_chunks, k_chunks, v_chunks)):
                csz = _q.size(2)
                decay_mask, chunk_decay, inner_decay = self.get_chunkwise_decay(csz)
                out_chunk, past_kv = self.chunkwise_retention(_q,
                                                              _k,
                                                              _v,
                                                              decay_mask,
                                                              past_kv=past_kv,
                                                              chunk_decay=chunk_decay,
                                                              inner_decay=inner_decay)
                ret_chunks.append(out_chunk)
            # [bsz, num_head, seqlen, v_dim]
            retention_out = torch.cat(ret_chunks, dim=2)
            curr_kv = past_kv
        else:
            raise ValueError(f'forward_impl {forward_impl} not supported.')
        # concaat heads
        retention_out = retention_out.transpose(1, 2).contiguous().view(B, T, self.config.v_dim)
        # group norm (merge batch, length dimension -> group norm -> split back)
        normed = self.gn(retention_out.view(B * T, self.config.v_dim))
        normed = normed.view(B, T, self.config.v_dim)
        # out gate & proj
        out = self.silu(self.gated(x)) * normed
        return self.proj(out), curr_kv


class RetNetBlock(nn.Module):

    def __init__(self, config: RetNetConfig):
        super().__init__()
        self.config = config
        self.msr = MultiScaleRetention(config)

        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.ffn_proj_size, bias=config.use_bias_in_mlp),
            nn.GELU(),
            nn.Linear(config.ffn_proj_size, config.hidden_size, bias=config.use_bias_in_mlp),
        )
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x, past_kv=None, forward_impl='parallel', sequence_offset=0):
        msr, curr_kv = self.msr(self.ln1(x),
                                past_kv=past_kv,
                                forward_impl=forward_impl,
                                sequence_offset=sequence_offset)
        y = x + msr
        y = y + self.ffn(self.ln2(y))
        return y, curr_kv


class RetNetModel(nn.Module):

    def __init__(self, config: RetNetConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([RetNetBlock(config) for _ in range(config.num_layers)])

    def forward(self,
                input_ids,
                forward_impl='parallel',
                past_kv=None,
                return_kv=True,
                sequence_offset=0):
        h = self.embedding(input_ids)

        kv_cache = []  # layers * [bsz, num_head, qk_dim, hidden_size]
        for i, block in enumerate(self.blocks):
            p_kv = past_kv[i] if past_kv is not None else None
            h, kv = block(h,
                          forward_impl=forward_impl,
                          past_kv=p_kv,
                          sequence_offset=sequence_offset)
            kv_cache.append(kv)

        if return_kv:
            return h, kv_cache
        return h


class RetNetModelWithLMHead(RetNetModel):

    def __init__(self, config: RetNetConfig) -> None:
        super().__init__(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.embedding.weight

    def forward(self,
                input_ids,
                forward_impl='parallel',
                past_kv=None,
                return_kv=True,
                sequence_offset=0):
        h, kv_cache = super().forward(input_ids,
                                      forward_impl=forward_impl,
                                      past_kv=past_kv,
                                      return_kv=True,
                                      sequence_offset=sequence_offset)
        lm_logits = self.lm_head(h)
        if return_kv:
            return lm_logits, kv_cache
        return lm_logits

    def sample_token(self, logit, do_sample=False, top_k=1, top_p=1.0, temperature=1.0):
        if not do_sample:
            return torch.argmax(logit, dim=-1)
        filtered = top_k_top_p_filtering(logit / temperature, top_k=top_k, top_p=top_p)
        return torch.multinomial(torch.softmax(filtered, dim=-1), num_samples=1)

    @torch.inference_mode()
    def generate(
        self,
        input_ids=None,
        parallel_compute_prompt=True,
        max_new_tokens=20,
        bos_token_id=0,
        eos_token_id=0,
        do_sample=False,
        top_k=1,
        top_p=1.0,
        temperature=1.0,
        early_stopping=True,
    ):
        generated = []
        if input_ids is not None:
            if parallel_compute_prompt:
                _, past_kv = self(input_ids[:, :-1], forward_impl='parallel', return_kv=True)
            else:
                past_kv = None
                for p_i in range(input_ids.size(1) - 1):
                    _, past_kv = self(input_ids[:, p_i:p_i + 1],
                                      forward_impl='recurrent',
                                      past_kv=past_kv,
                                      return_kv=True,
                                      sequence_offset=p_i)
            token = input_ids[:, -1].unsqueeze(-1)  # [B, 1]
        else:
            token = torch.tensor([[bos_token_id]]).to(self.lm_head.weight.device)
            past_kv = None

        for i in range(max_new_tokens):
            logit, past_kv = self(token,
                                  forward_impl='recurrent',
                                  past_kv=past_kv,
                                  return_kv=True,
                                  sequence_offset=i)
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
