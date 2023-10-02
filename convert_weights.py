"""
IMPORTANT NOTE: This works for the weights created from official torchscale/
code, at the commit 0b1f113985a0339bc322b0c7df91be0f745cb311 (July 24th, 2023).
The updated code on Sep 29, 2023 has different weights (such as RMSNorm instead of
LayerNorm, etc.), so this script will not work for the updated code.
"""

import os

import fire
import torch


def main(path_to_torchscale_weights, path_to_save=None):
    if path_to_save is None:
        weight_name, ext = os.path.splitext(path_to_torchscale_weights)
        path_to_save = weight_name + 'hf' + ext

    state_dict = torch.load(path_to_torchscale_weights, map_location='cpu')

    new_state_dict = {}
    for k, v in state_dict.items():
        if 'moe_layer' in k:  # ignore moe_layer
            continue
        if 'output_projection' in k:
            k = k.replace('output_projection', 'lm_head')  # output_projection -> lm_head
        else:
            k = 'model.' + k  # add model. prefix
        new_state_dict[k] = v

    os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
    torch.save(new_state_dict, path_to_save)


if __name__ == '__main__':
    fire.Fire(main)
