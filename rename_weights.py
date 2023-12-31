import fire
import torch


def rename_state_dict(state_dict):
    renamed_state_dict = {}
    for name in state_dict:
        weight = state_dict[name]
        name = name.replace("layernorm", "rms_norm")
        name = name.replace("layer_norm", "rms_norm")
        name = name.replace("retnet_rel_pos.", "rel_pos.")
        renamed_state_dict[name] = weight
    return renamed_state_dict


def main(ckpt_path):
    state_dict = torch.load(ckpt_path)
    renamed_state_dict = rename_state_dict(state_dict)

    # save shards
    torch.save(renamed_state_dict, ckpt_path)


if __name__ == "__main__":
    fire.Fire(main)
