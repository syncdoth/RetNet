import yaml

from transformers.configuration_utils import PretrainedConfig


def load_config_from_yaml(config_file):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = RetNetConfig.from_dict(config)
    return config


class RetNetConfig(PretrainedConfig):
    model_type = "retnet"

    def __init__(
            self,
            vocab_size: int = 50257,
            initializer_range: float = 0.02,
            is_decoder: bool = True,
            pad_token_id: int = 50256,
            eos_token_id: int = 50256,
            output_retentions: bool = False,
            use_cache: bool = True,
            forward_impl: str = 'parallel',
            activation_fn: str = "gelu",
            dropout: float = 0.0,  # dropout probability
            activation_dropout: float = 0.0,  # dropout probability after activation in FFN.
            drop_path_rate: float = 0.0,
            decoder_embed_dim: int = 768,  # decoder embedding dimension
            value_factor: int = 2,
            decoder_ffn_embed_dim: int = 1536,  # decoder embedding dimension for FFN
            decoder_layers: int = 12,  # num decoder layers
            decoder_retention_heads: int = 2,  # num decoder retention heads
            decoder_normalize_before: bool = True,  # apply layernorm before each decoder block
            layernorm_embedding: bool = False,  # add layernorm to embedding
            no_scale_embedding: bool = True,  # if True, dont scale embeddings
            recurrent_chunk_size: int = 512,
            use_lm_decay: bool = False,
            deepnorm: bool = False,
            subln: bool = True,
            layernorm_eps: float = 1e-5,
            **kwargs):
        self.vocab_size = vocab_size
        self.initializer_range = initializer_range
        self.output_retentions = output_retentions
        self.use_lm_decay = use_lm_decay
        # size related
        self.decoder_embed_dim = decoder_embed_dim
        self.value_factor = value_factor
        self.decoder_retention_heads = decoder_retention_heads
        self.decoder_ffn_embed_dim = decoder_ffn_embed_dim
        self.decoder_layers = decoder_layers
        # normalization related
        self.decoder_normalize_before = decoder_normalize_before
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.drop_path_rate = drop_path_rate
        self.activation_dropout = activation_dropout
        self.no_scale_embedding = no_scale_embedding
        self.layernorm_embedding = layernorm_embedding
        self.deepnorm = deepnorm
        self.subln = subln
        self.layernorm_eps = layernorm_eps
        # Blockwise
        self.recurrent_chunk_size = recurrent_chunk_size
        self.forward_impl = forward_impl

        if self.deepnorm:
            self.decoder_normalize_before = False
            self.subln = False
        if self.subln:
            self.decoder_normalize_before = True
            self.deepnorm = False

        super().__init__(is_decoder=is_decoder,
                         pad_token_id=pad_token_id,
                         eos_token_id=eos_token_id,
                         use_cache=use_cache,
                         **kwargs)
