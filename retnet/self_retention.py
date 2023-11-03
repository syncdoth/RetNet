import torch
import torch.nn as nn
from einops import rearrange
import numpy as np 
try:
    from torch_discounted_cumsum import  discounted_cumsum3_left
except:
    print("WARNING: torch_discounted_cumsum not installed, using pure python implementation.")


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = dim
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


class Discounted_Cumsum(nn.Module):
    """
    Assume input it (B, H, S, D) or (B, H, S, D1, D2)
                 or (B, D, H, S) or (B, D1, D2, H, S)
    ---> firstly, convert to
        - input (B*D, S)
        - gamma (B*D)
    ---> then, compute discounted cumsum by
        discounted_cumsum_left(input, gamma)
    ---> finally, convert back to original shape
    """
    def __init__(self, dim_head = -2, dim_leng = -1):
        super().__init__()
        self.dim_head  = dim_head
        self.dim_leng  = dim_leng
        
    def forward(self, tensor, gamma):
        _shape = tensor.shape
        assert _shape[self.dim_head] == gamma.shape[-1]
        ## then permute the target dim into 
        if self.dim_head == -2 and self.dim_leng == -1: #(B, D, H, S) or (B, D1, D2, H, S)
            tensor = tensor.view(-1, _shape[-1]) # (B*D*H, S)
        elif self.dim_head == 1 and self.dim_leng == 2:
            if   len(_shape) == 4:tensor = rearrange(tensor, 'B H S D -> (B D) H S')
            elif len(_shape) == 5:tensor = rearrange(tensor, 'B H S D1 D2 -> (B D1 D2) H S')
            else:raise NotImplementedError
        else:
            raise NotImplementedError
        #gamma  = gamma.repeat(len(tensor)//len(gamma)) #(H,) -> (B*D*H,) ## same as gamma.unsqueeze(0).unsqueeze(0).repeat(B,D,1).view(-1)
        #tensor = discounted_cumsum_left(tensor, gamma)
        assert len(gamma.shape)==1
        tensor = discounted_cumsum3_left(tensor, gamma)
        if   len(_shape) == 4:
            B,H,S,D = _shape
            tensor = rearrange(tensor, '(B D) H S -> B H S D', B=B)
        elif len(_shape) == 5:
            B,H,S,D1,D2 = _shape
            tensor = rearrange(tensor, '(B D1 D2) H S -> B H S D1 D2',  B=B, D1=D1)
        else:
            tensor = tensor.view(*_shape)
        return tensor

class RetNetRelPosV1(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
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
        self.cache = {
            'parallel':{},
            'recurrent':{},
            'chunkwise':{},
            'recurrent_chunk':{}
        }
    def forward(self,
                slen,
                forward_impl='parallel',
                recurrent_chunk_size=None,
                retention_mask=None,
                get_decay_scale=True):
        
        if forward_impl == 'recurrent':
            if slen in self.cache[forward_impl]:
                return self.cache[forward_impl][slen]
            sin = torch.sin(self.angle * (slen - 1))
            cos = torch.cos(self.angle * (slen - 1))
            retention_rel_pos = ((sin, cos), self.decay.view(1, -1, 1, 1).exp())
            self.cache[forward_impl][slen] = retention_rel_pos
        elif forward_impl in ['chunkwise', 'recurrent_chunk']:
            
            if recurrent_chunk_size is None:recurrent_chunk_size = self.recurrent_chunk_size
            if forward_impl == 'chunkwise':
                index = torch.arange(slen,slen).to(self.decay)
            else:   
                index = torch.arange(slen-recurrent_chunk_size,slen).to(self.decay)
            #index = torch.ones_like(index)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])

            block_index = torch.arange(recurrent_chunk_size).to(self.decay)
            mask = torch.tril(torch.ones(recurrent_chunk_size, recurrent_chunk_size)).to(self.decay)
            mask = torch.masked_fill(block_index[:, None] - block_index[None, :], ~mask.bool(),float("inf"))
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)
            mask = mask.unsqueeze(0)  # [1, h, t, t]
            # TODO: need to handle retention_mask
            # scaling
            value_inner_decay = mask[:, :, -1] / mask[:, :, -1].sum(dim=-1, keepdim=True)
            value_inner_decay = value_inner_decay.unsqueeze(-1)
            scale = mask.sum(dim=-1, keepdim=True).sqrt()
            inner_mask = mask / scale

            cross_decay = torch.exp(self.decay * recurrent_chunk_size)
            query_inner_decay = torch.exp(self.decay[:, None] * (block_index + 1))
            cross_decay = cross_decay[None, :, None, None]
            query_inner_decay = query_inner_decay[None, :, :, None] / (scale / mask[:, :, -1].sum(dim=-1)[:, :, None, None])
            # decay_scale (used for kv cache)

            decay_scale = self.compute_decay_scale(slen, retention_mask) if get_decay_scale else None

            retention_rel_pos = ((sin, cos), (inner_mask, cross_decay, query_inner_decay, value_inner_decay, #decay_scale
                                              ))
        else:  # parallel
            if slen in self.cache[forward_impl]:
                return self.cache[forward_impl][slen]
            index = torch.arange(slen).to(self.decay)
            #index = torch.ones_like(index)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])

            block_index = torch.arange(slen).to(self.decay)
            mask = torch.tril(torch.ones(slen, slen)).to(self.decay)
            mask = torch.masked_fill(block_index[:, None] - block_index[None, :], ~mask.bool(), float("inf"))
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)
            mask = mask.unsqueeze(0)  # [1, h, t, t]
            if retention_mask is not None:
                # this is required for left padding
                mask = mask * retention_mask.float().view(-1, 1, 1, slen).to(mask)
            gamma = mask[0,:,1,0]
            L     = mask.sum(dim=-1, keepdim=True).sqrt()
            # scaling
            mask = mask / L
            mask = torch.nan_to_num(mask, nan=0.0)
            # decay_scale (used for kv cache)
            decay_scale = self.compute_decay_scale(slen, retention_mask) if get_decay_scale else None
            # mask processing for intra decay
            if retention_mask is not None:
                max_non_zero = torch.cumsum(retention_mask, dim=-1).max(dim=-1).indices  # [b,]
                intra_decay = mask[range(mask.shape[0]), :, max_non_zero]
            else:
                intra_decay = mask[:, :, -1]

            retention_rel_pos = ((sin, cos), (mask, intra_decay, decay_scale,gamma, L))
            self.cache[forward_impl][slen] = retention_rel_pos
        return retention_rel_pos

    def compute_decay_scale(self, slen, retention_mask=None):
        exponent = torch.arange(slen, device=self.decay.device).float()
        decay_scale = self.decay.exp().view(-1, 1)**exponent.view(1, -1)  # [h, t]
        if retention_mask is not None:
            seqlen = retention_mask.sum(dim=-1)  # [b,]
            bsz = seqlen.size(0)
            decay_scale = decay_scale.unsqueeze(0).repeat(bsz, 1, 1)  # [b, h, t]
            for i, pos in enumerate(seqlen):
                # the formula for decay_scale is `sum(gamma^i) for i in [0, slen).`
                # Since the retention_mask is 0 for padding, we can set the decay_scale
                # to 0 for the padding positions.
                decay_scale[i, :, pos.item():] = 0
        else:
            bsz = 1
        decay_scale = decay_scale.sum(-1).view(bsz, -1, 1, 1)  # [b, h, 1, 1]
        return decay_scale

class SelfRetentionV1(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.embed_dim = config.decoder_embed_dim
        self.value_dim = config.decoder_value_embed_dim
        self.num_heads = config.decoder_retention_heads
        self.head_dim = self.value_dim // self.num_heads
        self.key_dim = self.embed_dim // self.num_heads
        self.scaling = self.key_dim**-0.5
        self.use_flash_retention = config.use_flash_retention
        self.gamma_cusum_1 = Discounted_Cumsum(1,2)
        self.gamma_cusum_2 = Discounted_Cumsum(1,2)
        self.group_norm = RMSNorm(self.head_dim, eps=config.layernorm_eps, elementwise_affine=False)

    def forward(self, q, k, v, 
                decay_mask,
                past_key_value=None, 
                retention_mask = None, 
                forward_impl= 'parallel'):     
        if forward_impl   == 'parallel':
            """
            q,  # bsz * num_head * len * qk_dim
            k,  # bsz * num_head * len * qk_dim
            v,  # bsz * num_head * len * v_dim
            decay_mask,  # (1 or bsz) * num_head * len * len
            """
            assert past_key_value is None, "parallel retention does not support past_key_value."
            assert retention_mask is None, "parallel retention does not support retention_mask."
            decay_mask, intra_decay, scale, gamma, L = decay_mask
            # just return retention_rel_pos projected
            # TODO: for shardformer
            #if self.decay_proj is not None:decay_mask = self.decay_proj(decay_mask.transpose(-1, -3)).transpose(-3, -1)
            
                
            if self.use_flash_retention and self.training:
                raise NotImplementedError("do not use in any case, under developing")
                B,H,L,D1 = q.shape
                B,H,L,D2 = v.shape
                assert D1*D2 < L/3, "do not use flash retention when D1*D2 > L/3"
                gamma = gamma.to(k.device).float()
                L     = L.to(q)
                qL    = q/L
                Tbf   = self.gamma_cusum_1(k,gamma)
                P     = torch.einsum('BHia, BHia->BHi',qL, Tbf)
                P     = P[...,None].detach().abs().clamp(min=1)
                D     = torch.einsum('BHia,BHic->BHiac',k, v)
                D     = self.gamma_cusum_2(D,gamma)
                O     = torch.einsum('BHia,BHiac->BHic',qL,D)/P
                output= rearrange(O,'B H i c->B i H c')
                return output, None, None, scale
            else:
                
                # [b, h, t, t]
                retention = q @ k.transpose(-1, -2)  # (scaled dot-product)
                retention = retention * decay_mask # invariant after normalization
                retention = retention / retention.detach().sum(dim=-1, keepdim=True).abs().clamp(min=1)
                output = retention @ v  # [b, h, t, v_dim / h]
                output = output.transpose(1, 2)  # [b, t, h, v_dim / h]
                output = self.group_norm(output)
                if self.training:  # skip cache
                    curr_kv = {"prev_key_value": None, "scale": scale}
                    return output, retention, curr_kv

                #if self.decay_proj is not None:intra_decay = self.decay_proj(intra_decay.transpose(-1, -2)).transpose(-2, -1)

                # kv cache: [b, h, t, v_dim, qk_dim]
                current_kv = k.unsqueeze(-2) * v.unsqueeze(-1)
                intra_decay = intra_decay[:, :, :, None, None]  # [b, h, t, 1, 1]
                current_kv = (current_kv * intra_decay).sum(2)  # [b, h, v_dim, qk_dim]

            curr_kv = {"prev_key_value": current_kv, "scale": scale}
            return output, retention, curr_kv
        elif forward_impl == 'recurrent':
            """
            q, k, v, # bsz * num_head * 1 * qkv_dim
            past_key_value:
                - "prev_key_value"  # bsz * num_head * v_dim * qk_dim
                - "scale"  # (1 or bsz) * num_head * 1 * 1
            decay # (1 or bsz) * num_head * 1 * 1
            retention_mask # bsz * 1
            """
            assert isinstance(decay_mask, torch.Tensor)
            decay = decay_mask 
            if retention_mask is not None:
                retention_mask = retention_mask.float().view(-1, 1, 1, 1).to(decay)
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
                prev_kv = prev_kv * decay_amount  # decay prev_kv
                current_kv = current_kv / scale.sqrt()  # scale current_kv
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
            #cache = {"prev_key_value": current_kv, "scale": scale}
            output = self.group_norm(output)
            curr_kv = {"prev_key_value": current_kv, "scale": scale}
            return output, None, curr_kv
        elif forward_impl == 'chunkwise':
            """
            q, k, v,  # bsz * num_head * seqlen * qkv_dim
            past_key_value:
                - "prev_key_value"  # bsz * num_head * v_dim * qk_dim
                - "scale"  # (1 or bsz) * num_head * 1 * 1
            decay_mask,    # 1 * num_head * chunk_size * chunk_size
            cross_decay,   # 1 * num_head * 1 * 1
            inner_decay,   # 1 * num_head * chunk_size * 1
            """
            # TODO: not working properly
            decay_mask, cross_decay, query_inner_decay, value_inner_decay = decay_mask
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
            kv = k_t @ (v * value_inner_decay)
            # kv = kv.view(bsz, num_chunks, self.num_heads, self.key_dim, self.head_dim)

            kv_recurrent = []
            cross_scale = []
            kv_state = torch.zeros(bsz, self.num_heads, self.key_dim, self.head_dim).to(v)
            kv_scale = torch.ones(bsz, self.num_heads, 1, 1).to(v)

            # accumulate kv by loop
            for i in range(num_chunks):
                kv_recurrent.append(kv_state / kv_scale)
                cross_scale.append(kv_scale)
                kv_state = kv_state * cross_decay + kv[:, i]
                kv_scale = kv_state.detach().abs().sum(dim=-2, keepdim=True).max(dim=-1, keepdim=True).values.clamp(min=1)

            kv_recurrent = torch.stack(kv_recurrent, dim=1)
            cross_scale = torch.stack(cross_scale, dim=1)

            all_scale = torch.maximum(inner_scale, cross_scale)
            align_inner_scale = all_scale / inner_scale
            align_cross_scale = all_scale / cross_scale

            cross_output = (q * query_inner_decay.unsqueeze(1)) @ kv_recurrent
            output = inner_output / align_inner_scale + cross_output / align_cross_scale
            output = output.transpose(2, 3)  # [b, n_c, t_c, h, v_dim]
            current_kv = kv_state.transpose(-2, -1)
            
            #cache = {"prev_key_value": current_kv, "scale": scale}
            output = self.group_norm(output)
            scale = None
            curr_kv = {"prev_key_value": current_kv, "scale": scale}
            return output, None, curr_kv
        elif forward_impl == 'recurrent_chunk':
            
            mask, cross_decay, query_inner_decay, value_inner_decay = decay_mask
            
            
            qk_mat      = torch.einsum('BHia,BHja->BHij', q, k)
            qk_mat      = qk_mat * mask
            inner_scale = qk_mat.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1) #(B,H,S,1)
            #inner_output = torch.matmul(qk_mat, v) # bsz * num_heads * num_value_heads * chunk_len * head_dim
            qk_mat      = qk_mat/ inner_scale
            inner_output = torch.einsum('BHij,BHjc->BHic', qk_mat, v)
            inner_output = inner_output
            
            
            ############## cross_retention_between_the_chunk_and_past ##################
            if past_key_value is not None:
                kv_state = past_key_value["prev_key_value"]
                kv_scale = past_key_value["scale"]
                kv_recurrent = kv_state / kv_scale
                cross_scale  = kv_scale
            else:
                B,H,_,kv_dim = k.shape
                B,H,_,v_dim  = v.shape
                kv_state = kv_recurrent = torch.zeros(B, H, kv_dim, v_dim).to(v) #(B,H,D1,D2)
                kv_scale = cross_scale  =  torch.ones(B, H, 1, 1).to(v)
            #print(kv_scale.flatten())
            # kv = kr_t @ (v * value_inner_decay)
            kv           = torch.einsum('BHja, BHjc-> BHac', k, v * value_inner_decay)
            next_state     = kv_state * cross_decay + kv 
            next_scale     = next_state.detach().abs().sum(dim=-2, keepdim=True).max(dim=-1, keepdim=True).values.clamp(min=1)
            
            
            
            all_scale = torch.maximum(inner_scale, cross_scale)
            align_inner_scale = all_scale / inner_scale
            align_cross_scale = all_scale / cross_scale
            cross_output = (q * query_inner_decay) @ kv_recurrent
            output = inner_output / align_inner_scale + cross_output / align_cross_scale
            output = output.transpose(1, 2)
            output = self.group_norm(output)
            curr_kv = {"prev_key_value": next_state, "scale": next_scale}
            return output, None, curr_kv
        else:
            raise ValueError(f'forward_impl {forward_impl} not supported.')


class RetNetRelPosV2(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        num_heads = config.decoder_retention_heads

        angle = 1.0 / (10000**torch.linspace(0, 1,
                       config.decoder_embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        # decay (gamma)
        if config.use_lm_decay:
            ###### lets control the perception window
            disapear_limit = torch.Tensor([0.0001])
            min_perception_distance = 100
            max_perception_distance = config.use_lm_decay
            decay = torch.log(disapear_limit)/torch.linspace(min_perception_distance, max_perception_distance, num_heads)
            
            # NOTE: alternative way described in the paper
            # s = torch.log(torch.tensor(1 / 32))
            # e = torch.log(torch.tensor(1 / 512))
            # decay = torch.log(1 - torch.exp(torch.linspace(s, e, num_heads)))  # [h,]
        else:
            decay = torch.log(1 - 2**(-5 - torch.arange(num_heads, dtype=torch.float)))
        local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
        if local_rank == 0:
            print(f"use decay {decay.exp()}")
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)
        self.recurrent_chunk_size = config.recurrent_chunk_size
        self.cache_sincos = {}
        self.cache_mask = {}

    def forward(self,
                slen,
                forward_impl='parallel',
                recurrent_chunk_size=None,
                retention_mask=None,
                get_decay_scale=True):
        assert slen > 0, "check your sequence length, it must big then 0"
        assert retention_mask is None
        if forward_impl == 'recurrent':
            assert recurrent_chunk_size == 1
        elif forward_impl == 'parallel':
            if recurrent_chunk_size is None or not recurrent_chunk_size:
                recurrent_chunk_size = slen
            assert recurrent_chunk_size == slen, f"recurrent_chunk_size:{recurrent_chunk_size} != slen:{slen}"
        elif forward_impl == 'chunkwise_recurrent':
            assert recurrent_chunk_size is not None, "must assign a recurrent_chunk_size"
        else:
            raise NotImplementedError

        if slen in self.cache_sincos and recurrent_chunk_size in self.cache_sincos[slen]:
            sin, cos = self.cache_sincos[slen][recurrent_chunk_size]
        else:
            index = torch.arange(slen-recurrent_chunk_size, slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])
            if slen not in self.cache_sincos:
                self.cache_sincos[slen] = {}
            self.cache_sincos[slen][recurrent_chunk_size] = (sin, cos)

        if slen in self.cache_mask and recurrent_chunk_size in self.cache_mask[slen]:
            chunk_gamma, mask, L = self.cache_mask[slen][recurrent_chunk_size]
        else:
            block_index = torch.arange(recurrent_chunk_size).to(self.decay)
            mask = self.create_tril_up_decay_mask(self.decay, block_index)

            last_mask = self.create_tril_up_decay_mask(
                self.decay, torch.arange(slen).to(self.decay), recurrent_chunk_size)
            L = last_mask.sum(dim=-1).sqrt()
            chunk_gamma = torch.einsum('H,C->HC', self.decay, block_index+1).exp()
            if slen not in self.cache_mask:
                self.cache_mask[slen] = {}
            self.cache_mask[slen][recurrent_chunk_size] = (
                chunk_gamma, mask, L)

        retention_rel_pos = ((sin, cos), (chunk_gamma, mask, L))

        return retention_rel_pos

    @staticmethod
    def create_tril_up_decay_mask(decay, block_index, recurrent_chunk_size=None):
        S = len(block_index)
        if recurrent_chunk_size is None:
            recurrent_chunk_size = len(block_index)
        mask = torch.tril(torch.ones(S, S)).to(decay)
        mask = mask[-recurrent_chunk_size:]
        mask = torch.masked_fill(
            block_index[-recurrent_chunk_size:, None] - block_index[None, :], ~mask.bool(), float("inf"))
        mask = torch.exp(mask * decay[:, None, None])
        mask = torch.nan_to_num(mask)
        return mask


class SelfRetentionV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.decoder_embed_dim
        self.value_dim = config.decoder_value_embed_dim
        self.num_heads = config.decoder_retention_heads
        self.head_dim = self.value_dim // self.num_heads
        self.key_dim = self.embed_dim // self.num_heads
        self.scaling = self.key_dim**-0.5
        self.use_flash_retention = config.use_flash_retention
        self.group_norm = RMSNorm(self.head_dim, eps=config.layernorm_eps, elementwise_affine=False)
        self.normlize_for_stable = config.normlize_for_stable

    def forward(self, q, k, v,
                decay_system,
                past_key_value=None,
                retention_mask=None,
                forward_impl='parallel',
                mode = 'qk_first',
                normlize_for_stable=None,
                output_increment=False ):
        normlize_for_stable = self.normlize_for_stable if normlize_for_stable is None else normlize_for_stable
        if normlize_for_stable == -1:
            normlize_for_stable = self.training ## only enable when training.
        kargs = {
            'past_key_value':past_key_value,
            'retention_mask':retention_mask,
            'normlize_for_stable':normlize_for_stable,
            'only_output': forward_impl == 'parallel' and self.training
        }
        if mode == 'kv_first':
            o, cache =  self.kv_first_forward(q, k, v, decay_system, **kargs)
        elif mode == 'qk_first':
            o, cache = self.qk_first_forward(q, k, v, decay_system, **kargs)
        elif mode == 'readable_qk_first':
            o, cache = self.readable_qk_first_forward(q, k, v, decay_system, **kargs)
        elif mode == 'readable_kv_first':
            o, cache = self.readbale_kv_first_forward(q, k, v, decay_system, **kargs)
        elif mode == 'kv_reduce':
            o, cache = self.kv_reduce_forward(q, k, v, decay_system, **kargs)
        else:
            raise NotImplementedError("mode must be 'kv_first' or 'qk_first'")
        retention = None 
        increment = None
        if output_increment:# notice we return the kv table and named it as retention
            increment = k.unsqueeze(-1)*v.unsqueeze(-2) #(B, H, C2, D2, D2)
            
        return self.group_norm(o), retention, cache, increment
    
    @staticmethod
    def readable_qk_first_forward(q, k, v,
                         decay_system,
                         past_key_value=None,
                         retention_mask=None,only_output=False,
                         normlize_for_stable=True):
        """
        q,    # (B,H,C1,D1)
        k,    # (B,H,C2,D1)
        v,    # (B,H,C2,D2)
        decay_system:
            - chunk_gamma:            (H,  C1    )
            - unnormlized_decay_mask: (H,  C1,  C2)
            - mask_normlizer:         (H,  C1    )
            
        # the real mask that hold \gamma^{i-j} is (normlized_decay_mask*mask_normlizer)
        ------------------------------------------------------------------

        """
        # (b, h, v_dim, qk_dim)
        
        chunk_gamma, unnormlized_decay_mask, mask_normlizer = decay_system
        H,C1,C2 = unnormlized_decay_mask.shape
        q = q/mask_normlizer.view(1,H,C1,1)

        normlized_qk   = torch.einsum('BHia,BHja->BHij',q, k)*unnormlized_decay_mask.view(1,H,C1,C2)
        numerator      = torch.einsum('BHij,BHjc->BHic',normlized_qk, v) 
        denominator    = normlized_qk.detach().sum(-1) if normlize_for_stable else None #(B,H,C1)
        current_scale  =  mask_normlizer# let the norm be directly assigned by decay_system is designed.
        
        if not only_output:last_unnormlized_kv = torch.einsum('BHja,Hj,BHjc->BHac', k ,unnormlized_decay_mask[:,-1], v) # (B, H ,D1, D2)
        if not only_output:last_unnormlized_gk = torch.einsum('BHja,Hj->BHa', k ,unnormlized_decay_mask[:,-1]).detach() if normlize_for_stable else None #(B,H,C1)
        if past_key_value is not None:
            assert "unnormlized_kv" in past_key_value
            if normlize_for_stable: assert past_key_value["unnormlized_gk"] is not None
            numerator   = numerator   + torch.einsum('BHia,BHab,Hi->BHib',q,past_key_value["unnormlized_kv"],chunk_gamma)
            #(B,H,C1,D2)=(B,H,C1,D2)  + (B,H,C1,D1)@(B,H,D1,D2)@(H, C1)->(B,H,C1,D2) 
            if not only_output:last_unnormlized_kv = last_unnormlized_kv + torch.einsum('BHab,H->BHab',past_key_value["unnormlized_kv"],chunk_gamma[:,-1])
            
            if past_key_value["unnormlized_gk"] is not None:
                denominator = denominator + torch.einsum('BHia,BHa,Hi->BHi'  ,q,past_key_value["unnormlized_gk"],chunk_gamma) #(B,H,C1)
                # (B,H,C1) =  (B,H,C1)   + #(B,H,C1,D1)@(B,H,D1)@(H, C1)->(B,H,C1) 
                if not only_output:last_unnormlized_gk = last_unnormlized_gk + torch.einsum( 'BHa,H->BHa' ,past_key_value["unnormlized_gk"],chunk_gamma[:,-1])
        numerator  = numerator
        denominator= denominator.abs().clamp(min=1).unsqueeze(-1) if denominator is not None else 1
        output = numerator/denominator

        output = output.permute(0,2,1,3)
        cache = {"unnormlized_kv": last_unnormlized_kv if not only_output else None, 
                 "unnormlized_gk": last_unnormlized_gk if not only_output else None, 
                "normlize_scale": current_scale # <-- used for check flow correct
                }
        #output = self.group_norm(output).reshape(output.size(0), -1, self.value_dim)
        return output, cache

    @staticmethod
    def qk_first_forward(q, k, v,
                         decay_system,
                         past_key_value=None,
                         retention_mask=None,only_output=False,
                         normlize_for_stable=True):
        """
        q,    # (B,H,C1,D1)
        k,    # (B,H,C2,D1)
        v,    # (B,H,C2,D2)
        decay_system:
            - chunk_gamma:            (H,  C1    )
            - unnormlized_decay_mask: (H,  C1,  C2)
            - mask_normlizer:         (H,  C1    )
            
        # the real mask that hold \gamma^{i-j} is (normlized_decay_mask*mask_normlizer)
        ------------------------------------------------------------------

        """
        # (b, h, v_dim, qk_dim)

        chunk_gamma, unnormlized_decay_mask, mask_normlizer = decay_system
        H, C1, C2 = unnormlized_decay_mask.shape
        q = q/mask_normlizer.view(1, H, C1, 1)
        
        normlized_qk = q@(k.mT)*unnormlized_decay_mask.view(1,H,C1,C2)
        #normlized_qk = torch.einsum('BHia,BHja->BHij', q, k)*unnormlized_decay_mask.view(1, H, C1, C2)
        #numerator = torch.einsum('BHij,BHjc->BHic', normlized_qk, v)
        numerator = normlized_qk@v
        denominator = normlized_qk.detach().sum(-1) if normlize_for_stable else None  # (B,H,C1)
        # let the norm be directly assigned by decay_system is designed.
        current_scale = mask_normlizer

        #last_unnormlized_kv = torch.einsum('BHja,Hj,BHjc->BHac', k, unnormlized_decay_mask[:, -1], v)  # (B, H ,D1, D2)
        #last_unnormlized_gk = torch.einsum('BHja,Hj->BHa', k, unnormlized_decay_mask[:, -1]).detach() if normlize_for_stable else None  # (B,H,C1)
        k = k*(unnormlized_decay_mask[:, -1].view(1,H,C2,1))
        if not only_output:last_unnormlized_gk = k.detach().sum(-2) if normlize_for_stable else None #(B,H,C1)
        if not only_output:last_unnormlized_kv = k.mT@v 
        if past_key_value is not None:
            assert "unnormlized_kv" in past_key_value
            if normlize_for_stable:assert past_key_value["unnormlized_gk"] is not None
            q = q*chunk_gamma.view(1,H,C1,1)
            #numerator = numerator + torch.einsum('BHia,BHab,Hi->BHib', q,past_key_value["unnormlized_kv"], chunk_gamma)
            numerator = numerator + q@past_key_value["unnormlized_kv"]
            #(B,H,C1,D2)=(B,H,C1,D2)  + (B,H,C1,D1)@(B,H,D1,D2)@(H, C1)->(B,H,C1,D2)
            #last_unnormlized_kv = last_unnormlized_kv + torch.einsum('BHab,H->BHab', past_key_value["unnormlized_kv"], chunk_gamma[:, -1])
            if not only_output:last_unnormlized_kv = last_unnormlized_kv + past_key_value["unnormlized_kv"]*chunk_gamma[:, -1].view(1,H,1,1)

            if past_key_value["unnormlized_gk"] is not None:
                # denominator = denominator + torch.einsum('BHia,BHa,Hi->BHi', q, past_key_value["unnormlized_gk"], chunk_gamma)  # (B,H,C1)
                denominator = denominator + (q@past_key_value["unnormlized_gk"][...,None])[...,0]
                # (B,H,C1) =  (B,H,C1)   + #(B,H,C1,D1)@(B,H,D1)@(H, C1)->(B,H,C1)
                #last_unnormlized_gk = last_unnormlized_gk + torch.einsum('BHa,H->BHa', past_key_value["unnormlized_gk"], chunk_gamma[:, -1])
                if not only_output:last_unnormlized_gk = last_unnormlized_gk + past_key_value["unnormlized_gk"]*chunk_gamma[:, -1].view(1,H,1)
        numerator = numerator
        denominator = denominator.abs().clamp(min=1).unsqueeze(-1) if denominator is not None else 1
        output = numerator/denominator

        output = output.permute(0, 2, 1, 3)
        cache = {"unnormlized_kv": last_unnormlized_kv if not only_output else None,
                 "unnormlized_gk": last_unnormlized_gk if not only_output else None,
                 "normlize_scale": current_scale  # <-- used for check flow correct
                 }
        #output = self.group_norm(output).reshape(output.size(0), -1, self.value_dim)
        return output, cache

    @staticmethod
    def readbale_kv_first_forward(q, k, v,
                         decay_system,
                         past_key_value=None,
                         retention_mask=None,only_output=False,
                         normlize_for_stable=True):
        """
        q,    # (B,H,C1,D1)
        k,    # (B,H,C1,D1)
        v,    # (B,H,C2,D2)
        decay_system:
            - chunk_gamma:            (H,  C1    )
            - unnormlized_decay_mask: (H,  C1,  C2)
            - mask_normlizer:         (H,  C1    )
            
        # the real mask that hold \gamma^{i-j} is (normlized_decay_mask*mask_normlizer)
        ------------------------------------------------------------------
        """
        # (b, h, v_dim, qk_dim)
        
        chunk_gamma, unnormlized_decay_mask, mask_normlizer = decay_system
        B,H,C1,D1 = q.shape
        B,H,C2,D2 = v.shape
        H, C1, C2 = unnormlized_decay_mask.shape

        unnormlized_kv = torch.einsum('BHja,Hij,BHjc->BHiac', k, unnormlized_decay_mask, v)  # (B, H ,D1, D2)
        unnormlized_gk = torch.einsum('BHja,Hij->BHia'     , k, unnormlized_decay_mask).detach() if normlize_for_stable else None  # (B, H ,1, D1) -> (B, H, D1)

        # let the norm be directly assigned by decay_system is designed.
        current_scale = mask_normlizer
        if past_key_value is not None:
            assert "unnormlized_kv" in past_key_value
            if normlize_for_stable:assert past_key_value["unnormlized_gk"] is not None
            # we need firstly revert the nomrlized_kv to unnormlized_kv by mutiple the scale
            # current_scale= ((past_key_value["normlize_scale"]**2) * decay + 1  ).sqrt()
            # print("normlizer_error",torch.dist(mask_normlizer,((past_key_value["normlize_scale"]**2) * normlized_decay_mask + 1  ).sqrt()))
            unnormlized_kv = unnormlized_kv + torch.einsum('BHab,Hi->BHiab',past_key_value["unnormlized_kv"], chunk_gamma) 
            unnormlized_gk = unnormlized_gk + torch.einsum('BHa,Hi->BHia',  past_key_value["unnormlized_gk"], chunk_gamma) if past_key_value["unnormlized_gk"] is not None else None


        q = q/mask_normlizer.view(1, H, C1, 1)
        # torch.sum(q * current_kv, dim=3).unsqueeze(1)  # (B, 1,H, D2)
        numerator   = torch.einsum("BHia,BHiab->BiHb", q, unnormlized_kv)
        denominator = torch.einsum("BHia,BHia ->BiH",  q.detach(), unnormlized_gk).abs().clamp(min=1).unsqueeze(-1) if unnormlized_gk is not None else 1
        output = numerator/denominator  # (B,H,C,D2)/(B,H,C,1)

        cache = {"unnormlized_kv": unnormlized_kv[:, :, -1],
                 "unnormlized_gk": unnormlized_gk[:, :, -1] if unnormlized_gk is not None else None ,
                 "normlize_scale": current_scale  # <-- used for check flow correct
                 }
        #output = self.group_norm(output).reshape(output.size(0), -1, self.value_dim)
        return output, cache

    @staticmethod
    def kv_first_forward(q, k, v,
                         decay_system,
                         past_key_value=None,
                         retention_mask=None,only_output=False,
                         normlize_for_stable=True):
        """
        q,    # (B,H,C1,D1)
        k,    # (B,H,C1,D1)
        v,    # (B,H,C2,D2)
        decay_system:
            - chunk_gamma:            (H,  C1    )
            - unnormlized_decay_mask: (H,  C1,  C2)
            - mask_normlizer:         (H,  C1    )
            
        # the real mask that hold \gamma^{i-j} is (normlized_decay_mask*mask_normlizer)
        ------------------------------------------------------------------
        """
        # (b, h, v_dim, qk_dim)
        
        chunk_gamma, unnormlized_decay_mask, mask_normlizer = decay_system
        B,H,C1,D1 = q.shape
        B,H,C2,D2 = v.shape
        H, C1, C2 = unnormlized_decay_mask.shape

        #unnormlized_kv = torch.einsum('BHja,Hij,BHjc->BHiac', k, unnormlized_decay_mask, v)  # (B, H ,D1, D2)
        unnormlized_kv = k.unsqueeze(-1)*v.unsqueeze(-2) # (B,H,C2, D1, 1)*(B,H,C2,1,D2) -> (B,H,C2,D1,D2)
        unnormlized_kv = torch.einsum('BHjac,Hij->BHiac', unnormlized_kv, unnormlized_decay_mask)  
        #unnormlized_gk = torch.einsum('BHja,Hij->BHia'  , k, unnormlized_decay_mask).detach() if normlize_for_stable else None  # (B, H ,1, D1) -> (B, H, D1)
        unnormlized_gk =  unnormlized_decay_mask @(k.detach())  if normlize_for_stable else None  # (B, H ,1, D1) -> (B, H, D1)
        
        # let the norm be directly assigned by decay_system is designed.
        current_scale = mask_normlizer
        if past_key_value is not None:
            assert "unnormlized_kv" in past_key_value
            if normlize_for_stable:assert past_key_value["unnormlized_gk"] is not None
            # we need firstly revert the nomrlized_kv to unnormlized_kv by mutiple the scale
            # current_scale= ((past_key_value["normlize_scale"]**2) * decay + 1  ).sqrt()
            # print("normlizer_error",torch.dist(mask_normlizer,((past_key_value["normlize_scale"]**2) * normlized_decay_mask + 1  ).sqrt()))
            #unnormlized_kv = unnormlized_kv + torch.einsum('BHab,Hi->BHiab',past_key_value["unnormlized_kv"], chunk_gamma) 
            unnormlized_kv = unnormlized_kv + past_key_value["unnormlized_kv"].view(B,H,1,D1,D2)*chunk_gamma.view(1,H,C1,1,1)
            if past_key_value["unnormlized_gk"] is not None:
                #unnormlized_gk = unnormlized_gk + torch.einsum('BHa,Hi->BHia',  past_key_value["unnormlized_gk"], chunk_gamma)  is not None else None
                unnormlized_gk = unnormlized_gk + past_key_value["unnormlized_gk"].view(B,H,1,D1)*chunk_gamma.view(1,H,C1,1)
            else:
                unnormlized_gk = None

        q = q/mask_normlizer.view(1, H, C1, 1)
        # torch.sum(q * current_kv, dim=3).unsqueeze(1)  # (B, 1,H, D2)
        # numerator   = torch.einsum("BHia,BHiab->BHib", q, unnormlized_kv)
        numerator = (q.unsqueeze(-2)@unnormlized_kv).squeeze(-2)
        #denominator = torch.einsum("BHia,BHia ->BHi",  q.detach(), unnormlized_gk).abs().clamp(min=1).unsqueeze(-1) if unnormlized_gk is not None else 1
        denominator = (q.detach()*unnormlized_gk).sum(-1) if unnormlized_gk is not None else None
        denominator = denominator.abs().clamp(min=1).unsqueeze(-1) if denominator is not None else 1
        output = numerator/denominator  # (B,H,C,D2)/(B,H,C,1)
        output = output.permute(0, 2, 1, 3)
        cache = {"unnormlized_kv": unnormlized_kv[:, :, -1],
                 "unnormlized_gk": unnormlized_gk[:, :, -1] if unnormlized_gk is not None else None ,
                 "normlize_scale": current_scale  # <-- used for check flow correct
                 }
        #output = self.group_norm(output).reshape(output.size(0), -1, self.value_dim)
        return output, cache

    @staticmethod
    def kv_reduce_forward(q, k, v,
                         decay_system,
                         past_key_value=None,
                         retention_mask=None,only_output=False,
                         normlize_for_stable=True):
        """
        q_bar_coef = omask[...,:,0]/omask.sum(dim=-1).sqrt()
        k_bar_coef = 1/(omask[...,:,0])#<----this will overflow~~~~!!!!
        q_bar = q_bar_coef[...,None]*q
        k_bar = k_bar_coef[...,None]*k
        T = torch.cumsum(k_bar,dim=-2)
        P = torch.einsum('BHia,BHia->BHi', T,q_bar)
        P = P[...,None].detach().abs().clamp(min=1)
        q_bar = q_bar/P
        D = torch.einsum('BHia,BHic->BHiac',k_bar, v)
        D = torch.cumsum(D,dim=-3)
        O = torch.einsum('BHia,BHiac->BHic',q_bar,D)
        ------------------------------------------------------------------
        """
        # (b, h, v_dim, qk_dim)
        chunk_gamma, unnormlized_decay_mask, mask_normlizer = decay_system
        
        B,H,C1,D1 = q.shape
        B,H,C2,D2 = v.shape
        H, C1, C2 = unnormlized_decay_mask.shape
        assert C1 == C2
        decay_mask = unnormlized_decay_mask[...,0]
        q = q
        k = k/decay_mask.view(1, H, C1, 1)
        unnormlized_kv  = decay_mask.view(1, H, C1, 1, 1)*torch.cumsum(k.unsqueeze(-1)*v.unsqueeze(-2),dim=-3)
        unnormlized_gk  = decay_mask.view(1, H, C1, 1)*torch.cumsum(k.detach(),dim=-2) if normlize_for_stable else None  # (B, H ,1, D1) -> (B, H, D1)

        
        # let the norm be directly assigned by decay_system is designed.
        current_scale = mask_normlizer
        if past_key_value is not None:
            assert "unnormlized_kv" in past_key_value
            if normlize_for_stable:assert past_key_value["unnormlized_gk"] is not None
            # we need firstly revert the nomrlized_kv to unnormlized_kv by mutiple the scale
            # current_scale= ((past_key_value["normlize()_scale"]**2) * decay + 1  ).sqrt
            # print("normlizer_error",torch.dist(mask_normlizer,((past_key_value["normlize_scale"]**2) * normlized_decay_mask + 1  ).sqrt()))
            #unnormlized_kv = unnormlized_kv + torch.einsum('BHab,Hi->BHiab',past_key_value["unnormlized_kv"], chunk_gamma) 
            unnormlized_kv = unnormlized_kv + past_key_value["unnormlized_kv"].view(B,H,1,D1,D2)*chunk_gamma.view(1,H,C1,1,1)
            if past_key_value["unnormlized_gk"] is not None:
                #unnormlized_gk = unnormlized_gk + torch.einsum('BHa,Hi->BHia',  past_key_value["unnormlized_gk"], chunk_gamma)  is not None else None
                unnormlized_gk = unnormlized_gk + past_key_value["unnormlized_gk"].view(B,H,1,D1)*chunk_gamma.view(1,H,C1,1)
            else:
                unnormlized_gk = None

        q = q/mask_normlizer.view(1, H, C1, 1)
        # torch.sum(q * current_kv, dim=3).unsqueeze(1)  # (B, 1,H, D2)
        # numerator   = torch.einsum("BHia,BHiab->BHib", q, unnormlized_kv)
        numerator = (q.unsqueeze(-2)@unnormlized_kv).squeeze(-2)
        #denominator = torch.einsum("BHia,BHia ->BHi",  q.detach(), unnormlized_gk).abs().clamp(min=1).unsqueeze(-1) if unnormlized_gk is not None else 1
        denominator = (q.detach()*unnormlized_gk).sum(-1) if unnormlized_gk is not None else None
        denominator = denominator.abs().clamp(min=1).unsqueeze(-1) if denominator is not None else 1
        output = numerator/denominator  # (B,H,C,D2)/(B,H,C,1)
        output = output.permute(0, 2, 1, 3)
        cache = {"unnormlized_kv": unnormlized_kv[:, :, -1],
                 "unnormlized_gk": unnormlized_gk[:, :, -1] if unnormlized_gk is not None else None ,
                 "normlize_scale": current_scale  # <-- used for check flow correct
                 }
        #output = self.group_norm(output).reshape(output.size(0), -1, self.value_dim)
        return output, cache

       


if __name__ == "__main__":
    import numpy as np
    from configuration_retnet import RetNetConfig
    from tqdm.auto import tqdm
    import time
    def meta_test(q,k,v,retnet_rel_pos1, model1,
                  retnet_rel_pos2, model2,
                  use_gk = True,
                  mode   = 'qk_first'):
        
        (cos,sin), decay_system = retnet_rel_pos1(S,forward_impl='parallel')
        parallel_output = model1(q,k,v,decay_system)[0]

        (cos,sin), (chunk_gamma, unnormlized_decay_mask,mask_normlizer) = retnet_rel_pos2(S,recurrent_chunk_size=S,forward_impl='chunkwise_recurrent')
        chunkwise_output,_,chunkwise_cache =  model2(q,k,v,(chunk_gamma, unnormlized_decay_mask,mask_normlizer),normlize_for_stable=use_gk,mode=mode)
        print("     ================= whole chunk size S=S test ====================")
        print(f"     error before group_norm {torch.dist(parallel_output,chunkwise_output):.3f}")
        print(f"     error after  group_norm {torch.dist(group_norm(parallel_output),group_norm(chunkwise_output)):.3f}")
        print("     ================= rnn chunk size=1 test ====================")
        past_kv = None
        full_rnn_state = []
        for i in range(0,S):
            (cos,sin), (chunk_gamma, unnormlized_decay_mask,mask_normlizer) = retnet_rel_pos2(i+1,recurrent_chunk_size=1,forward_impl='chunkwise_recurrent')
            one_step_output, _, past_kv = model2(q[:,:,i:i+1],k[:,:,i:i+1],v[:,:,i:i+1],
                                                                (chunk_gamma, unnormlized_decay_mask,mask_normlizer),
                                                                past_key_value= past_kv,
                                                                normlize_for_stable=use_gk, mode=mode)
            #print(past_kv['normlize_scale'].squeeze()[:2].cpu().numpy())
            full_rnn_state.append(one_step_output)
        full_rnn_state = torch.cat(full_rnn_state, dim=1)
        print(f"     error of first element  {torch.dist(parallel_output[:, 0], full_rnn_state[:, 0]):.3f}")
        print(f"     error before group_norm {torch.dist(parallel_output,full_rnn_state):.3f}")
        print(f"     error after  group_norm {torch.dist(group_norm(parallel_output),group_norm(full_rnn_state)):.3f}")
        print("     ================= parallel+rnn chunk size=3 test ====================")
        offset = 3
        past_kv= None
        (cos,sin), (chunk_gamma, unnormlized_decay_mask,mask_normlizer) = retnet_rel_pos2(S-offset,recurrent_chunk_size=S-offset,forward_impl='chunkwise_recurrent')
        start_output, _, start_cache =  model2(q[:,:,:-offset],k[:,:,:-offset],v[:,:,:-offset],
                                                                (chunk_gamma, unnormlized_decay_mask,mask_normlizer),
                                                                past_key_value= past_kv,
                                            normlize_for_stable=use_gk, mode=mode)

        (cos,sin), (chunk_gamma, unnormlized_decay_mask,mask_normlizer) = retnet_rel_pos2(S,recurrent_chunk_size=offset,forward_impl='chunkwise_recurrent')
        next_output, _, next_cache =  model2(q[:,:,-offset:],k[:,:,-offset:],v[:,:,-offset:],
                                                            (chunk_gamma, unnormlized_decay_mask,mask_normlizer),
                                                            past_key_value= start_cache,
                                                            normlize_for_stable=use_gk, mode=mode)
        rnn_output = torch.cat([start_output, next_output],1)
        print(f"     error of start element  {torch.dist(parallel_output[:,:-offset], start_output):.3f}")
        print(f"     error before group_norm {torch.dist(parallel_output,rnn_output):.3f}")
        print(f"     error after  group_norm {torch.dist(group_norm(parallel_output),group_norm(rnn_output)):.3f}")

        print("     ================= random chunksize recurrent test ====================")
        partition = np.sort(np.random.choice(np.arange(2,S-2),(5,),replace=False)).tolist() + [S]
        print(f"     partition: {partition}")
        past_kv = None
        full_rnn_state = []
        last = 0
        for i in partition:
            qm = q[:,:,last:i]
            km = k[:,:,last:i]
            vm = v[:,:,last:i]
            (cos, sin), (chunk_gamma, unnormlized_decay_mask, mask_normlizer) = retnet_rel_pos2(
                i, recurrent_chunk_size=qm.shape[-2], forward_impl='chunkwise_recurrent')
            one_step_output, _, past_kv = model2(qm, km, vm,
                                                (chunk_gamma, unnormlized_decay_mask,mask_normlizer),
                                                past_key_value= past_kv,
                                              normlize_for_stable=use_gk, mode=mode)
            full_rnn_state.append(one_step_output)
            last = i
        full_rnn_state = torch.cat(full_rnn_state, dim=1)
        print(f"     error before group_norm {torch.dist(parallel_output,full_rnn_state):.3f}")
        print(f"     error after  group_norm {torch.dist(group_norm(parallel_output),group_norm(full_rnn_state)):.3f}")

    def whole_recurrent(q, k, v, retnet_rel_pos, model,use_gk=True, mode='qk_first'):
        past_kv = None
        full_rnn_state = []
        for i in range(0,S):
            (cos,sin), (chunk_gamma, unnormlized_decay_mask,mask_normlizer) = retnet_rel_pos(i+1,recurrent_chunk_size=1,forward_impl='chunkwise_recurrent')
            one_step_output, _, past_kv = model(q[:,:,i:i+1],k[:,:,i:i+1],v[:,:,i:i+1],
                                                                (chunk_gamma, unnormlized_decay_mask,mask_normlizer),
                                                                past_key_value= past_kv,
                                                                normlize_for_stable=use_gk, mode=mode)
            #print(past_kv['normlize_scale'].squeeze()[:2].cpu().numpy())
            full_rnn_state.append(one_step_output)
        full_rnn_state = torch.cat(full_rnn_state, dim=1)
        return full_rnn_state

    def timecost_profile(fun,*arg,**kargs):
        first_run = fun(*arg,**kargs)
        costs = []
        for _ in tqdm(range(20),leave=False):
            now = time.time()
            out = whole_recurrent(*arg,**kargs)
            costs.append(time.time()-now)
        mode = kargs.get('mode', 'none')
        sign = '+' if kargs.get('use_gk',True) else '-'
        print(f"[{mode} {sign} gk]:time cost: {np.mean(costs):.3f}+-{np.std(costs):.3f}")
        return out
    config = RetNetConfig(decoder_layers=1,
                      decoder_embed_dim=256,
                      decoder_value_embed_dim=256,
                      decoder_retention_heads=8,
                      decoder_ffn_embed_dim=128)
    S = 30
    B = 2
    H = 8
    qk_dim = 32
    v_dim  = 64
    q = torch.randn(B,H,S,qk_dim).cuda()
    k = torch.randn(B,H,S,qk_dim).cuda()
    v = torch.randn(B,H,S, v_dim).cuda()

    retention_origin = SelfRetentionV1(config)
    retention_v2     = SelfRetentionV2(config)
    group_norm       = RMSNorm(H,0,False)
    retention_origin.group_norm = nn.Identity()
    retention_v2.group_norm = nn.Identity()

    
    retnet_rel_posV1 = RetNetRelPosV1(config).cuda()
    retnet_rel_posV2 = RetNetRelPosV2(config).cuda()

    # print("===================================================================================")
    # print("================= check the timecost among different mode [qk] ====================")
    # print("===================================================================================")
    # fqk_first_with_gk = timecost_profile(whole_recurrent, q, k, v, retnet_rel_posV2, retention_v2, use_gk=True , mode='qk_first')
    # fqk_first_wito_gk = timecost_profile(whole_recurrent, q, k, v, retnet_rel_posV2, retention_v2, use_gk=False, mode='qk_first')
    # rqk_first_with_gk = timecost_profile(whole_recurrent, q, k, v, retnet_rel_posV2, retention_v2, use_gk=True , mode='readable_qk_first')
    # rqk_first_wito_gk = timecost_profile(whole_recurrent, q, k, v, retnet_rel_posV2, retention_v2, use_gk=False, mode='readable_qk_first')
    
    # print("================= check the consistancy between different mode [before group norm] ====================")
    # print(f"fast qk + gk <=> fast qk - gk:{torch.dist(fqk_first_with_gk,fqk_first_wito_gk).item():.4f}")
    # print(f"fast qk + gk <=> read qk + gk:{torch.dist(fqk_first_with_gk,rqk_first_with_gk).item():.4f}")
    # print(f"fast qk + gk <=> read qk - gk:{torch.dist(fqk_first_with_gk,rqk_first_wito_gk).item():.4f}")
    # print(f"fast qk - gk <=> read qk - gk:{torch.dist(fqk_first_wito_gk,rqk_first_wito_gk).item():.4f}")
    # print("================= check the consistancy between different mode [after group norm] ====================")
    # print(f"fast qk + gk <=> fast qk - gk:{torch.dist(group_norm(fqk_first_with_gk),group_norm(fqk_first_wito_gk)).item():.4f}")
    # print(f"fast qk + gk <=> read qk + gk:{torch.dist(group_norm(fqk_first_with_gk),group_norm(rqk_first_with_gk)).item():.4f}")
    # print(f"fast qk + gk <=> read qk - gk:{torch.dist(group_norm(fqk_first_with_gk),group_norm(rqk_first_wito_gk)).item():.4f}")
    # print(f"fast qk - gk <=> read qk - gk:{torch.dist(group_norm(fqk_first_wito_gk),group_norm(rqk_first_wito_gk)).item():.4f}")
    
    # print("===================================================================================")
    # print("================= check the timecost among different mode [kv] ====================")
    # print("===================================================================================")
    # fkv_first_with_gk = timecost_profile(whole_recurrent, q, k, v, retnet_rel_posV2, retention_v2, use_gk=True , mode='kv_first')
    # fkv_first_wito_gk = timecost_profile(whole_recurrent, q, k, v, retnet_rel_posV2, retention_v2, use_gk=False, mode='kv_first')
    # rkv_first_with_gk = timecost_profile(whole_recurrent, q, k, v, retnet_rel_posV2, retention_v2, use_gk=True , mode='readable_kv_first')
    # rkv_first_wito_gk = timecost_profile(whole_recurrent, q, k, v, retnet_rel_posV2, retention_v2, use_gk=False, mode='readable_kv_first')
    
    # print("================= check the consistancy between different mode [before group norm] ====================")
    # print(f"fast kv + gk <=> fast kv - gk:{torch.dist(fkv_first_with_gk,fkv_first_wito_gk).item():.4f}")
    # print(f"fast kv + gk <=> read kv + gk:{torch.dist(fkv_first_with_gk,rkv_first_with_gk).item():.4f}")
    # print(f"fast kv + gk <=> read kv - gk:{torch.dist(fkv_first_with_gk,rkv_first_wito_gk).item():.4f}")
    # print(f"fast kv - gk <=> read kv - gk:{torch.dist(fkv_first_wito_gk,rkv_first_wito_gk).item():.4f}")
    # print("================= check the consistancy between different mode [after group norm] ====================")
    # print(f"fast kv + gk <=> fast kv - gk:{torch.dist(group_norm(fkv_first_with_gk),group_norm(fkv_first_wito_gk)).item():.4f}")
    # print(f"fast kv + gk <=> read kv + gk:{torch.dist(group_norm(fkv_first_with_gk),group_norm(rkv_first_with_gk)).item():.4f}")
    # print(f"fast kv + gk <=> read kv - gk:{torch.dist(group_norm(fkv_first_with_gk),group_norm(rkv_first_wito_gk)).item():.4f}")
    # print(f"fast kv - gk <=> read kv - gk:{torch.dist(group_norm(fkv_first_wito_gk),group_norm(rkv_first_wito_gk)).item():.4f}")
    
    # print("===================================================================================")
    # print("================= check the timecost among different mode [kv] ====================")
    # print("===================================================================================")
    # fkv_first_with_gk = timecost_profile(whole_recurrent, q, k, v, retnet_rel_posV2, retention_v2, use_gk=True , mode='kv_first')
    # fkv_first_wito_gk = timecost_profile(whole_recurrent, q, k, v, retnet_rel_posV2, retention_v2, use_gk=False, mode='kv_first')
    # rkv_first_with_gk = timecost_profile(whole_recurrent, q, k, v, retnet_rel_posV2, retention_v2, use_gk=True , mode='kv_reduce')
    # rkv_first_wito_gk = timecost_profile(whole_recurrent, q, k, v, retnet_rel_posV2, retention_v2, use_gk=False, mode='kv_reduce')
    
    # print("================= check the consistancy between different mode [before group norm] ====================")
    # print(f"fast kv + gk <=> fast kv - gk:{torch.dist(fkv_first_with_gk,fkv_first_wito_gk).item():.4f}")
    # print(f"fast kv + gk <=> redu kv + gk:{torch.dist(fkv_first_with_gk,rkv_first_with_gk).item():.4f}")
    # print(f"fast kv + gk <=> redu kv - gk:{torch.dist(fkv_first_with_gk,rkv_first_wito_gk).item():.4f}")
    # print(f"fast kv - gk <=> redu kv - gk:{torch.dist(fkv_first_wito_gk,rkv_first_wito_gk).item():.4f}")
    # print("================= check the consistancy between different mode [after group norm] ====================")
    # print(f"fast kv + gk <=> fast kv - gk:{torch.dist(group_norm(fkv_first_with_gk),group_norm(fkv_first_wito_gk)).item():.4f}")
    # print(f"fast kv + gk <=> redu kv + gk:{torch.dist(group_norm(fkv_first_with_gk),group_norm(rkv_first_with_gk)).item():.4f}")
    # print(f"fast kv + gk <=> redu kv - gk:{torch.dist(group_norm(fkv_first_with_gk),group_norm(rkv_first_wito_gk)).item():.4f}")
    # print(f"fast kv - gk <=> redu kv - gk:{torch.dist(group_norm(fkv_first_wito_gk),group_norm(rkv_first_wito_gk)).item():.4f}")
    
    # (cos,sin), (decay_mask,intra_decay, scale,gamma, L) = retnet_rel_posV1(S,forward_impl='parallel')
    # parallel_output_origin, _ , _ = retention_origin(q, k, v, (decay_mask, intra_decay, scale, gamma, L), past_key_value=None,forward_impl='parallel')

    # (cos, sin), (chunk_gamma, unnormlized_decay_mask, mask_normlizer) = retnet_rel_posV2(S, forward_impl='parallel')
    # parallel_output_qk_with_gk,_, parallel_cache =  retention_v2(q,k,v,(chunk_gamma, unnormlized_decay_mask,mask_normlizer),mode='qk_first',normlize_for_stable=True)
    # parallel_output_qk_wito_gk,_, parallel_cache =  retention_v2(q,k,v,(chunk_gamma, unnormlized_decay_mask,mask_normlizer),mode='qk_first',normlize_for_stable=False)
    
    # print("========== check the consistancy between origin implement output and qk version with gk ==========")
    # print(f" qk + gk <=> origin: before group_norm {torch.dist(parallel_output_qk_with_gk,parallel_output_origin):.3f}")
    # print(f" qk + gk <=> origin: after  group_norm {torch.dist(group_norm(parallel_output_qk_with_gk),group_norm(parallel_output_origin)):.3f}")
    # print(f" qk - gk <=> origin: before group_norm {torch.dist(parallel_output_qk_wito_gk,parallel_output_origin):.3f}")
    # print(f" qk - gk <=> origin: after  group_norm {torch.dist(group_norm(parallel_output_qk_wito_gk),group_norm(parallel_output_origin)):.3f}")

    # parallel_output_kv_with_gk,_, parallel_cache =  retention_v2(q,k,v,(chunk_gamma, unnormlized_decay_mask,mask_normlizer),mode='kv_first',normlize_for_stable=True)
    # parallel_output_kv_wito_gk,_, parallel_cache =  retention_v2(q,k,v,(chunk_gamma, unnormlized_decay_mask,mask_normlizer),mode='kv_first',normlize_for_stable=False)
    
    # print("========== check the consistancy between origin implement output and kv version with gk ==========")
    # print(f" kv + gk <=> origin: before group_norm {torch.dist(parallel_output_kv_with_gk,parallel_output_origin):.3f}")
    # print(f" kv + gk <=> origin: after  group_norm {torch.dist(group_norm(parallel_output_kv_with_gk),group_norm(parallel_output_origin)):.3f}")
    # print(f" kv - gk <=> origin: before group_norm {torch.dist(parallel_output_kv_wito_gk,parallel_output_origin):.3f}")
    # print(f" kv - gk <=> origin: after  group_norm {torch.dist(group_norm(parallel_output_kv_wito_gk),group_norm(parallel_output_origin)):.3f}")

    

    use_gk = True
    mode   = 'qk_first'
    print(f"============= use_gk:{use_gk} mode:{mode} ==============")
    meta_test(q,k,v,retnet_rel_posV1, retention_origin,retnet_rel_posV2, retention_v2, use_gk = use_gk, mode=mode)
    
    use_gk = True
    mode   = 'kv_first'
    print(f"============= use_gk:{use_gk} mode:{mode} ==============")
    meta_test(q,k,v,retnet_rel_posV1, retention_origin,retnet_rel_posV2, retention_v2, use_gk = use_gk, mode=mode)

    use_gk = False
    mode   = 'qk_first'
    print(f"============= use_gk:{use_gk} mode:{mode} ==============")
    meta_test(q,k,v,retnet_rel_posV1, retention_origin,retnet_rel_posV2, retention_v2, use_gk = use_gk, mode=mode)

    use_gk = False
    mode   = 'kv_first'
    print(f"============= use_gk:{use_gk} mode:{mode} ==============")
    meta_test(q,k,v,retnet_rel_posV1, retention_origin,retnet_rel_posV2, retention_v2, use_gk = use_gk, mode=mode)

    use_gk = False
    mode   = 'kv_reduce'
    print(f"============= use_gk:{use_gk} mode:{mode} ==============")
    meta_test(q,k,v,retnet_rel_posV1, retention_origin,retnet_rel_posV2, retention_v2, use_gk = use_gk, mode=mode)
    
    exit()
