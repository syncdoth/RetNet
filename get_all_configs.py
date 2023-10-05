from argparse import Namespace
import json

import fire

from retnet.configuration_retnet import RetNetConfig


# retnet_base
def retnet_base_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, "no_tie_adaptive_proj"):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if hasattr(args, "decoder_final_norm"):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = getattr(args, "dropout", 0.0)  # NOTE: might want to set it 0.1

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_value_embed_dim = getattr(args, "decoder_value_embed_dim", 864)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 864)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_retention_heads = getattr(args, "decoder_retention_heads", 2)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.adaptive_softmax_factor = getattr(args, "adaptive_softmax_factor", 4)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.activation_fn = getattr(args, "activation_fn", "swish")

    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)

    args.base_layers = getattr(args, "base_layers", 0)
    args.base_sublayers = getattr(args, "base_sublayers", 1)
    args.base_shuffle = getattr(args, "base_shuffle", False)

    args.add_bos_token = getattr(args, "add_bos_token", False)
    args.no_token_positional_embeddings = getattr(args, "no_token_positional_embeddings", False)
    args.share_decoder_input_output_embed = getattr(args, "share_decoder_input_output_embed", False)
    args.character_embeddings = getattr(args, "character_embeddings", False)

    args.decoder_output_dim = getattr(args, "decoder_output_dim", args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.chunkwise_recurrent = getattr(args, "chunkwise_recurrent", False)
    args.recurrent_chunk_size = getattr(args, "recurrent_chunk_size", 512)

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", False)

    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.adaptive_input_factor = getattr(args, "adaptive_input_factor", 4)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", None)

    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True


def retnet_medium(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_value_embed_dim = getattr(args, "decoder_value_embed_dim", 1728)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1728)
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_retention_heads = getattr(args, "decoder_retention_heads", 4)
    retnet_base_architecture(args)


def retnet_xl(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 2048)
    args.decoder_value_embed_dim = getattr(args, "decoder_value_embed_dim", 3456)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3456)
    args.decoder_retention_heads = getattr(args, "decoder_retention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    retnet_base_architecture(args)


def retnet_3b(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 2560)
    args.decoder_value_embed_dim = getattr(args, "decoder_value_embed_dim", 4280)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4280)
    args.decoder_retention_heads = getattr(args, "decoder_retention_heads", 10)
    args.decoder_layers = getattr(args, "decoder_layers", 32)
    retnet_base_architecture(args)


def retnet_7b(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 4096)
    args.decoder_value_embed_dim = getattr(args, "decoder_value_embed_dim", 6912)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 6912)
    args.decoder_retention_heads = getattr(args, "decoder_retention_heads", 16)
    args.decoder_layers = getattr(args, "decoder_layers", 32)
    retnet_base_architecture(args)


def retnet_13b(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 5120)
    args.decoder_value_embed_dim = getattr(args, "decoder_value_embed_dim", 8560)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 8560)
    args.decoder_retention_heads = getattr(args, "decoder_retention_heads", 20)
    args.decoder_layers = getattr(args, "decoder_layers", 40)
    retnet_base_architecture(args)


def retnet_65b(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 8192)
    args.decoder_value_embed_dim = getattr(args, "decoder_value_embed_dim", 13824)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 13824)
    args.decoder_retention_heads = getattr(args, "decoder_retention_heads", 32)
    args.decoder_layers = getattr(args, "decoder_layers", 64)
    retnet_base_architecture(args)


def get_config(size, args=None):
    size_map = {
        'base': retnet_base_architecture,
        'medium': retnet_medium,
        'xl': retnet_xl,
        '3b': retnet_3b,
        '7b': retnet_7b,
        '13b': retnet_13b,
        '65b': retnet_65b,
    }
    if args is None:
        args = Namespace()
    size_map[size](args)
    config = RetNetConfig()
    config.override(args)
    return config


def main(**kwargs):
    for size in ['base', 'medium', 'xl', '3b', '7b', '13b', '65b']:
        args = Namespace(**kwargs) if kwargs else None
        config = get_config(size, args=args)
        config.save_pretrained('configs/retnet-' + size)


if __name__ == "__main__":
    fire.Fire(main)
