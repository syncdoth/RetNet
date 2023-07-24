import yaml

from transformers.configuration_utils import PretrainedConfig


def load_config_from_yaml(config_file):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = RetNetConfig.from_dict(config)
    return config


class RetNetConfig(PretrainedConfig):
    model_type = "retnet"

    def __init__(self,
                 vocab_size: int = 50257,
                 hidden_size: int = 2048,
                 num_layers: int = 24,
                 num_heads: int = 8,
                 qk_dim: int = 2048,
                 v_dim: int = 4096,
                 ffn_proj_size: int = 4096,
                 use_bias_in_msr: bool = False,
                 use_bias_in_mlp: bool = True,
                 use_bias_in_msr_out: bool = False,
                 use_default_gamma: bool = False,
                 initializer_range: float = 0.02,
                 is_decoder: bool = True,
                 pad_token_id: int = 50256,
                 eos_token_id: int = 50256,
                 output_retentions: bool = False,
                 use_cache: bool = True,
                 forward_impl: str = 'parallel',
                 chunk_size: int = 512,
                 **kwargs):

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.ffn_proj_size = ffn_proj_size
        self.use_bias_in_msr = use_bias_in_msr
        self.use_bias_in_mlp = use_bias_in_mlp
        self.use_bias_in_msr_out = use_bias_in_msr_out
        self.use_default_gamma = use_default_gamma
        self.initializer_range = initializer_range
        self.output_retentions = output_retentions
        self.forward_impl = forward_impl
        self.chunk_size = chunk_size

        super().__init__(is_decoder=is_decoder,
                         pad_token_id=pad_token_id,
                         eos_token_id=eos_token_id,
                         use_cache=use_cache,
                         **kwargs)
