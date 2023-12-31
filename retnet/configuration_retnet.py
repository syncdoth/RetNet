from dataclasses import dataclass
import json

from transformers.configuration_utils import PretrainedConfig


def load_config_from_json(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
        config = RetNetConfig.from_dict(config)
    return config


@dataclass
class RetNetConfig(PretrainedConfig):
    model_type = "retnet"
    initializer_factor: float = 2**-2.5
    initializer_range: float = 0.02
    lm_head_initializer_range: float = 4096**-0.5
    pad_token_id: int = None
    bos_token_id: int = 1
    eos_token_id: int = 2
    use_cache: bool = True
    forward_mode: str = "parallel"
    activation_fn: str = "swish"
    dropout: float = 0.0
    activation_dropout: float = 0.0
    drop_path_rate: float = 0.0
    decoder_embed_dim: int = 4096
    decoder_value_embed_dim: int = 6912
    decoder_ffn_embed_dim: int = 6912
    decoder_layers: int = 32
    decoder_retention_heads: int = 16
    decoder_normalize_before: bool = True
    rms_norm_embedding: bool = True
    no_scale_embedding: bool = False
    recurrent_chunk_size: int = 512
    use_lm_decay: bool = False
    z_loss_coeff: float = 0.0
    deepnorm: bool = False
    rms_norm_eps: float = 1e-6
    groupnorm_eps: float = 1e-6
    tie_word_embeddings: bool = False

    attribute_map = {
        "hidden_size": "decoder_embed_dim",
        "intermediate_size": "decoder_ffn_embed_dim",
        "num_attention_heads": "decoder_retention_heads",
        "num_hidden_layers": "decoder_layers",
    }

    def __init__(
        self,
        vocab_size: int = 32000,
        initializer_factor: float = 2**-2.5,
        initializer_range: float = 0.02,
        lm_head_initializer_range: float = 4096**-0.5,
        pad_token_id: int = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        use_cache: bool = True,
        forward_mode: str = "parallel",
        activation_fn: str = "swish",
        dropout: float = 0.0,
        activation_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        decoder_embed_dim: int = 4096,
        decoder_value_embed_dim: int = 6912,
        decoder_ffn_embed_dim: int = 6912,
        decoder_layers: int = 32,
        decoder_retention_heads: int = 16,
        decoder_normalize_before: bool = True,
        rms_norm_embedding: bool = True,
        no_scale_embedding: bool = False,
        recurrent_chunk_size: int = 512,
        use_lm_decay: bool = False,
        z_loss_coeff: float = 0.0,
        deepnorm: bool = False,
        rms_norm_eps: float = 1e-6,
        groupnorm_eps: float = 1e-6,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.lm_head_initializer_range = lm_head_initializer_range
        # retentive network related
        self.use_lm_decay = use_lm_decay
        self.recurrent_chunk_size = recurrent_chunk_size
        self.forward_mode = forward_mode
        # size related
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_value_embed_dim = decoder_value_embed_dim
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
        self.rms_norm_embedding = rms_norm_embedding
        self.deepnorm = deepnorm
        self.rms_norm_eps = rms_norm_eps
        self.groupnorm_eps = groupnorm_eps
        self.z_loss_coeff = z_loss_coeff

        if self.deepnorm:
            self.decoder_normalize_before = False

        super().__init__(bos_token_id=bos_token_id,
                         pad_token_id=pad_token_id,
                         eos_token_id=eos_token_id,
                         use_cache=use_cache,
                         tie_word_embeddings=tie_word_embeddings,
                         **kwargs)

    def override(self, args):
        for hp in self.__dict__.keys():
            if getattr(args, hp, None) is not None:
                self.__dict__[hp] = getattr(args, hp, None)
