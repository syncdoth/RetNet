import torch
from torch import nn

from utils import split_chunks, split_heads


class MultiScaleRetention(nn.Module):
    # TODO: normalization in the paper

    def __init__(self,
                 hidden_size,
                 qk_dim,
                 num_heads,
                 use_bias=True,
                 use_bias_in_out=True,
                 chunk_size=None,
                 default_gamma=False) -> None:
        super().__init__()
        assert qk_dim % num_heads == 0, 'qk_dim must be divisible by num_heads'
        assert hidden_size % num_heads == 0, 'hidden_size must be divisible by num_heads'
        # normalization & activation
        self.gn = nn.GroupNorm(num_heads, hidden_size)  # NOTE: also see self.group_norm
        self.swish = nn.SiLU()
        # retention
        self.w_query = nn.Linear(hidden_size, qk_dim, bias=use_bias)
        self.w_key = nn.Linear(hidden_size, qk_dim, bias=use_bias)
        self.w_value = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        # output parameters
        self.w_out = nn.Linear(hidden_size, hidden_size, bias=use_bias_in_out)
        self.w_gate = nn.Linear(hidden_size, hidden_size, bias=use_bias_in_out)
        # hyperparameters
        self.qk_dim = qk_dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.default_gamma = default_gamma

        # initialize gamma
        if default_gamma:
            gamma = 1 - 2**(-5 - torch.arange(0, num_heads, dtype=torch.float))
        else:
            s = -torch.log(torch.tensor(32))
            e = -torch.log(torch.tensor(512))
            gamma = 1 - torch.exp(-torch.linspace(s, e, num_heads))  # [h,]
        self.gamma = nn.Parameter(gamma, requires_grad=False)

    def forward(self, x, forward_impl='parallel', past_kv=None):
        bsz, seqlen = x.size(0), x.size(1)
        # [bsz, seqlen, qkv_dim]
        query = self.w_query(x)
        key = self.w_key(x)
        value = self.w_value(x)
        # [bsz, num_head, seqlen, qkv_dim/num_head]
        query, key, value = split_heads((query, key, value), bsz, seqlen, self.num_heads)
        if forward_impl == 'parallel':
            decay_mask = self.get_parallel_decay_mask(seqlen)
            retention_out = self.parallel_retention(query, key, value, decay_mask)
            past_kv = None
        elif forward_impl == 'recurrent':
            decay = self.get_recurrent_decay()
            retention_out, past_kv = self.recurrent_retention(query,
                                                              key,
                                                              value,
                                                              past_kv=past_kv,
                                                              decay=decay)
        elif forward_impl == 'chunkwise':
            q_chunks, k_chunks, v_chunks = split_chunks(query,
                                                        key,
                                                        value,
                                                        size=self.chunk_size,
                                                        dim=2)
            past_kv = None
            ret_chunks = []
            for i, (q, k, v) in enumerate(zip(q_chunks, k_chunks, v_chunks)):
                csz = q.size(2)
                decay_mask, chunk_decay, inner_decay = self.get_chunkwise_decay(csz, i)
                out_chunk, past_kv = self.chunkwise_retention(q,
                                                              k,
                                                              v,
                                                              decay_mask,
                                                              past_kv=past_kv,
                                                              chunk_decay=chunk_decay,
                                                              inner_decay=inner_decay)
                ret_chunks.append(out_chunk)
            # [bsz, seqlen, hidden_size]
            retention_out = torch.cat(ret_chunks, dim=2)
        else:
            raise ValueError(f'forward_impl {forward_impl} not supported.')

        out = self.w_out(self.swish(self.w_gate(x)) * retention_out)

        return out, past_kv

    def get_parallel_decay_mask(self, length):
        decay_mask = torch.zeros(self.num_heads, length, length)
        for n in range(length):
            for m in range(0, n + 1):
                decay_mask[:, n, m] = self.gamma**(n - m)
        return decay_mask

    def get_recurrent_decay(self):
        return self.gamma.view(self.num_heads, 1, 1)

    def get_chunkwise_decay(self, chunk_size, chunk_idx):
        # within chunk decay
        decay_mask = self.get_parallel_decay_mask(chunk_size)
        # decay of the chunk
        chunk_decay = self.gamma.view(self.num_heads, 1, 1)**chunk_size
        # cross-chunk decay
        inner_decay = self.gamma.unsqueeze(-1)**(chunk_idx + 1)

        return decay_mask, chunk_decay, inner_decay

    def group_norm(self, x):
        """performs group normalization on x after concatenating the heads"""
        transposed = x.transpose(1, 2)
        # [bsz, seqlen, hidden_size]
        concat = transposed.reshape(transposed.size(0), transposed.size(1), -1)
        norm = self.gn(concat.transpose(1, 2)).transpose(1, 2)
        return norm

    def parallel_retention(self, q, k, v, decay_mask):
        """
        q,  # bsz * num_head * len * qk_dim
        k,  # bsz * num_head * len * qk_dim
        v,  # bsz * num_head * len * v_dim
        decay_mask,  # num_head * len * len
        """
        retention = q @ k.transpose(-1, -2)
        retention = retention * decay_mask
        output = retention @ v
        output = self.group_norm(output)

        return output

    def recurrent_retention(self, q, k, v, past_kv=None, decay=None):
        """
        q, k, v, # bsz * num_head * len * qkv_dim
        past_kv, # bsz * num_head * qk_dim * v_dim
        decay # num_head * 1 * 1
        """
        past = decay * past_kv if past_kv is not None else 0
        current_kv = past + k.unsqueeze(-1) * v.unsqueeze(-2)
        output = torch.sum(q.unsqueeze(-1) * current_kv, dim=-2)
        output = self.group_norm(output)
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
        decay_mask,  # num_head * chunk_size * chunk_size
        chunk_decay,  # num_head * 1 * 1
        inner_decay,  # num_head * chunk_size
        """
        retention = q @ k.transpose(-1, -2)  # [bsz, num_head, chunk_size, chunk_size]
        retention = retention * decay_mask
        inner_retention = retention @ v  # [bsz, num_head, chunk_size, v_dim]
        # [bsz, num_head, chunk_size, v_dim]
        cross_retention = (q @ past_kv) * inner_decay if past_kv is not None else 0
        retention = inner_retention + cross_retention
        output = self.group_norm(retention)
        # [bsz, num_head, qk_dim, v_dim]
        past = chunk_decay * past_kv if past_kv is not None else 0
        current_kv = past + k.transpose(-1, -2) @ v
        return output, current_kv


class RetNetMLP(nn.Module):

    def __init__(self, hidden_size, proj_size, use_bias=True) -> None:
        super().__init__()
        self.activation = nn.GELU()
        self.w1 = nn.Linear(hidden_size, proj_size, bias=use_bias)
        self.w2 = nn.Linear(proj_size, hidden_size, bias=use_bias)

    def forward(self, x):
        return self.w2(self.activation(self.w1(x)))


class RetNetBlock(nn.Module):

    def __init__(self,
                 hidden_size,
                 qk_dim,
                 mlp_proj_size,
                 num_heads,
                 chunk_size=None,
                 use_bias_in_msr=False,
                 use_bias_in_msr_out=True,
                 use_bias_in_mlp=True) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.msr = MultiScaleRetention(hidden_size,
                                       qk_dim,
                                       num_heads,
                                       chunk_size=chunk_size,
                                       use_bias=use_bias_in_msr,
                                       use_bias_in_out=use_bias_in_msr_out)
        self.ffn = RetNetMLP(hidden_size, mlp_proj_size, use_bias=use_bias_in_mlp)

    def forward(self, x, forward_impl='parallel', past_kv=None):
        y, curr_kv = self.msr(self.layer_norm(x), forward_impl=forward_impl, past_kv=past_kv)
        y = y + x
        out = self.ffn(self.layer_norm(y)) + y
        return out, curr_kv


class RetNet(nn.Module):

    def __init__(self,
                 num_layers,
                 vocab_size,
                 hidden_size,
                 qk_dim,
                 mlp_proj_size,
                 num_heads,
                 chunk_size=None,
                 use_bias_in_msr=False,
                 use_bias_in_msr_out=True,
                 use_bias_in_mlp=True) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList(
            RetNetBlock(hidden_size,
                        qk_dim,
                        mlp_proj_size,
                        num_heads,
                        chunk_size=chunk_size,
                        use_bias_in_msr=use_bias_in_msr,
                        use_bias_in_mlp=use_bias_in_mlp,
                        use_bias_in_msr_out=use_bias_in_msr_out) for _ in range(num_layers))

    def forward(self, input_ids, forward_impl='parallel', past_kv=None, return_kv=True):
        h = self.embedding(input_ids)

        kv_cache = []  # layers * [bsz, num_head, qk_dim, hidden_size]
        for i, block in enumerate(self.blocks):
            h, kv = block(h, forward_impl=forward_impl, past_kv=past_kv)
            if return_kv:
                kv_cache.append(kv)
        if return_kv:
            return h, kv
        return h
