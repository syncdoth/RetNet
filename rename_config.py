import json

import fire


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def rename_config(config):
    config["pad_token_id"] = None
    config["bos_token_id"] = 1
    config["eos_token_id"] = 2
    config["vocab_size"] = 32000
    config["groupnorm_eps"] = 1e-06
    config["initializer_factor"] = 2**-2.5
    config["lm_head_initializer_range"] = config["decoder_embed_dim"]**-0.5

    config["forward_mode"] = config.pop("forward_impl")
    config["rms_norm_embedding"] = config.pop("layernorm_embedding")
    config["rms_norm_eps"] = config.pop("layernorm_eps")

    sorted_config = {k: config[k] for k in sorted(config)}
    return sorted_config


def main(config_path):
    # for size in ["3b", "7b", "13b", "65b", "300m", "base", "medium", "xl"]:
    #     config_path = f"configs/retnet-{size}/config.json"
    save_json(config_path, rename_config(load_json(config_path)))


if __name__ == "__main__":
    fire.Fire(main)
