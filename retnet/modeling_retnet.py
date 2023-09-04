from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
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

    def get_parallel_decay_mask(self, length, retention_mask=None):
        range_tensor = torch.arange(length, device=self.decay.device)
        range_tensor = range_tensor[None, :, None].expand(self.config.num_heads, length, 1)
        exponent = range_tensor - range_tensor.transpose(-1, -2)
        decay_mask = self.decay.view(-1, 1, 1)**exponent
        decay_mask = torch.tril(decay_mask, diagonal=0)  # [h, t, t]
        if retention_mask is not None:
            retention_mask = retention_mask.float().view(-1, 1, 1, length)
            decay_mask = decay_mask.unsqueeze(0) * retention_mask
        else:
            decay_mask = decay_mask.unsqueeze(0)
        return decay_mask

    def get_recurrent_decay(self):
        decay = self.decay.view(1, self.config.num_heads, 1, 1)
        return decay

    def get_chunkwise_decay(self, chunk_size, retention_mask=None):
        # within chunk decay
        decay_mask = self.get_parallel_decay_mask(chunk_size, retention_mask=retention_mask)
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
        chunk_size: Optional[int] = None,
        output_retentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor]]:
        B, T, H = hidden_states.size()
        q, k, v = self.qkv(hidden_states).split(
            [self.config.qk_dim, self.config.qk_dim, self.config.v_dim], dim=-1)
        q, k = self.xpos.rotate_queries_and_keys(q, k, offset=sequence_offset)
        q, k, v = split_heads((q, k, v), B, T, self.config.num_heads)
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
            assert chunk_size is not None
            q_chunks, k_chunks, v_chunks = split_chunks(q, k, v, size=chunk_size, dim=2)
            if retention_mask is not None:
                retention_mask_chunks = split_chunks(retention_mask, size=chunk_size, dim=1)[0]
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
        retention_out = retention_out.transpose(1, 2).contiguous().view(B, T, self.config.v_dim)
        # group norm (merge batch, length dimension -> group norm -> split back)
        normed = self.gn(retention_out.view(B * T, self.config.v_dim))
        normed = normed.view(B, T, self.config.v_dim)
        # out gate & proj
        out = self.silu(self.gated(hidden_states)) * normed

        outputs = (self.proj(out), curr_kv)
        if output_retentions:
            outputs += (retention_weights,) if forward_impl == 'parallel' else (None,)
        return outputs


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

    def forward(
        self,
        hidden_states: torch.Tensor,
        retention_mask: Optional[torch.Tensor] = None,
        forward_impl: str = 'parallel',
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        sequence_offset: Optional[int] = 0,
        chunk_size: Optional[int] = None,
        output_retentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor]]:

        msr_outs = self.msr(self.ln1(hidden_states),
                            retention_mask=retention_mask,
                            past_key_value=past_key_value,
                            forward_impl=forward_impl,
                            sequence_offset=sequence_offset,
                            chunk_size=chunk_size,
                            output_retentions=output_retentions)
        msr = msr_outs[0]
        curr_kv = msr_outs[1]
        y = hidden_states + msr
        y = y + self.ffn(self.ln2(y))

        outputs = (y, curr_kv)

        if output_retentions:
            outputs += (msr_outs[2],)
        return outputs


class RetNetPreTrainedModel(PreTrainedModel):
    # copied from LlamaPretrainedModel
    config_class = RetNetConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RetNetBlock"]
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


class RetNetModel(RetNetPreTrainedModel):

    def __init__(self, config: RetNetConfig) -> None:
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.blocks = nn.ModuleList([RetNetBlock(config) for _ in range(config.num_layers)])

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, value):
        self.embedding = value

    def forward(
        self,
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
        chunk_size: Optional[int] = None,
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

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if retention_mask is None:
            if attention_mask is not None:
                retention_mask = attention_mask
            else:
                # TODO: might not need this
                retention_mask = torch.ones((batch_size, seq_length),
                                            dtype=torch.bool,
                                            device=inputs_embeds.device)

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_retentions = () if output_retentions else None
        # layers * [bsz, num_head, qk_dim, hidden_size]
        next_decoder_cache = () if use_cache else None

        for i, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs, sequence_offset, chunk_size, output_retentions)

                    return custom_forward

                block_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    retention_mask,
                    forward_impl,
                    past_key_value,
                )
            else:
                block_outputs = block(hidden_states,
                                      retention_mask=retention_mask,
                                      forward_impl=forward_impl,
                                      past_key_value=past_key_value,
                                      sequence_offset=sequence_offset,
                                      chunk_size=chunk_size,
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


class RetNetModelWithLMHead(RetNetPreTrainedModel):

    def __init__(self, config: RetNetConfig) -> None:
        super().__init__(config)
        self.model = RetNetModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

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
        chunk_size: Optional[int] = None,
    ) -> Union[Tuple, RetNetCausalLMOutputWithPast]:
        output_retentions = output_retentions if output_retentions is not None else self.config.output_retentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else
                                self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        forward_impl = forward_impl if forward_impl is not None else self.config.forward_impl
        chunk_size = chunk_size if chunk_size is not None else self.config.chunk_size

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
                             chunk_size=chunk_size)

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
