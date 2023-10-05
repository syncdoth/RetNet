# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]


class RetNetConfig(object):

    def __init__(self, **kwargs):
        self.decoder_embed_dim = kwargs.pop("decoder_embed_dim", 768)
        self.decoder_value_embed_dim = kwargs.pop("decoder_value_embed_dim", 1280)
        self.decoder_retention_heads = kwargs.pop("decoder_retention_heads", 3)
        self.decoder_ffn_embed_dim = kwargs.pop("decoder_ffn_embed_dim", 1280)
        self.decoder_layers = kwargs.pop("decoder_layers", 12)
        self.decoder_normalize_before = kwargs.pop("decoder_normalize_before", True)
        self.activation_fn = kwargs.pop("activation_fn", "gelu")
        self.dropout = kwargs.pop("dropout", 0.0)
        self.drop_path_rate = kwargs.pop("drop_path_rate", 0.0)
        self.activation_dropout = kwargs.pop("activation_dropout", 0.0)
        self.no_scale_embedding = kwargs.pop("no_scale_embedding", True)
        self.layernorm_embedding = kwargs.pop("layernorm_embedding", False)
        self.moe_freq = kwargs.pop("moe_freq", 0)
        self.moe_top1_expert = kwargs.pop("moe_top1_expert", False)
        self.moe_expert_count = kwargs.pop("moe_expert_count", 0)
        self.moe_gating_use_fp32 = kwargs.pop("moe_gating_use_fp32", True)
        self.moe_eval_capacity_token_fraction = kwargs.pop("moe_eval_capacity_token_fraction", 0.25)
        self.moe_second_expert_policy = kwargs.pop("moe_second_expert_policy", "random")
        self.moe_normalize_gate_prob_before_dropping = kwargs.pop(
            "moe_normalize_gate_prob_before_dropping", False)
        self.use_xmoe = kwargs.pop("use_xmoe", False)
        self.rel_pos_buckets = kwargs.pop("rel_pos_buckets", 0)
        self.max_rel_pos = kwargs.pop("max_rel_pos", 0)
        self.deepnorm = kwargs.pop("deepnorm", False)
        self.subln = kwargs.pop("subln", True)
        self.use_ffn_rms_norm = kwargs.pop("use_ffn_rms_norm", False)
        self.use_glu = kwargs.pop("use_glu", True)
        self.use_lm_decay = kwargs.pop("use_lm_decay", False)
        self.z_loss_coeff = kwargs.pop("z_loss_coeff", 0.0)  # TODO: 1e-4
        self.multiway = kwargs.pop("multiway", False)
        self.share_decoder_input_output_embed = kwargs.pop("share_decoder_input_output_embed",
                                                           False)
        self.max_target_positions = kwargs.pop("max_target_positions", 1024)
        self.no_output_layer = kwargs.pop("no_output_layer", True)
        self.layernorm_eps = kwargs.pop("layernorm_eps", 1e-6)
        # Blockwise
        self.chunkwise_recurrent = kwargs.pop("chunkwise_recurrent", False)
        self.recurrent_chunk_size = kwargs.pop("recurrent_chunk_size", 512)
        # Text
        self.vocab_size = kwargs.pop("vocab_size", -1)
        # Fairscale
        self.checkpoint_activations = kwargs.pop("checkpoint_activations", False)
        self.fsdp = kwargs.pop("fsdp", False)
        self.ddp_rank = kwargs.pop("ddp_rank", 0)
        self.xpos_rel_pos = kwargs.pop("xpos_rel_pos", False)
        self.xpos_scale_base = kwargs.pop("xpos_scale_base", 512)
        # token id
        self.pad_token_id = kwargs.pop("pad_token_id", 0)

        if self.deepnorm:
            self.decoder_normalize_before = False
            self.subln = False
        if self.subln:
            self.decoder_normalize_before = True
            self.deepnorm = False
        if self.use_xmoe:
            self.moe_normalize_gate_prob_before_dropping = True
            self.moe_second_expert_policy = "random"
            assert self.moe_freq > 0 and self.moe_expert_count > 0

    def override(self, args):
        for hp in self.__dict__.keys():
            if getattr(args, hp, None) is not None:
                self.__dict__[hp] = getattr(args, hp, None)