import argparse
from collections import OrderedDict
import torch


def convert(pth, outfile):
    # Convert from CUDA --> CPU
    checkpoint = torch.load(pth, map_location=lambda storage, loc: storage)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    # The state_dict saved in this checkpoint has keys formatted as:
    # 'module.fc.weight' (or something of this form). We don't want
    # the 'module' part.
    new_state_dict = {}
    for name in state_dict:
        splits = name.split('.')

        if splits[0] == "linear":
            new_name = '.'.join(["fc"] + splits[1:])
            new_state_dict[new_name] = state_dict[name]
        elif len(splits) > 2 and splits[2] == "shortcut":
            new_name = '.'.join(splits[:2] + ["downsample"] + splits[3:])
            new_state_dict[new_name] = state_dict[name]
        else:
            new_state_dict[name] = state_dict[name]

    new_state_dict = OrderedDict(new_state_dict)
    torch.save(new_state_dict, outfile)


def strip(pth, outfile):
    # Convert from CUDA --> CPU
    checkpoint = torch.load(pth, map_location=lambda storage, loc: storage)

    if "state_dict" in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # The state_dict saved in this checkpoint has keys formatted as:
    # 'module.fc.weight' (or something of this form). We don't want
    # the 'module' part.
    new_state_dict = {}
    for name in state_dict:
        splits = name.split('.')
        new_name = '.'.join(splits[1:])
        new_state_dict[new_name] = state_dict[name]
        print(name, new_name)

    new_state_dict = OrderedDict(new_state_dict)
    torch.save(new_state_dict, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to pytorch checkpoint")
    parser.add_argument("outfile", type=str,
                        help="Path to place converted state_dict")
    parser.add_argument("mode", type=str, help="One of {convert, strip}")
    args = parser.parse_args()

    assert args.mode in ["convert", "strip"]

    if args.mode == "convert":
        convert(args.path, args.outfile)
    else:
        strip(args.path, args.outfile)
