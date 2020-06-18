""" Utility functions """
from functools import reduce
import os
from shutil import copyfile
import torch
import torch.nn as nn


def construct(obj_map, extra_kwargs={}):
    """
    Constructs an object of class and with parameters specified in
    ``obj_map``, along with any additional arguments passed in through
    ``extra_kwargs``.
    """
    classname = obj_map["class"]
    if "args" in obj_map:
        kwargs = obj_map["args"]
    else:
        kwargs = {}
    kwargs.update(extra_kwargs)
    c = get_from_module(classname)
    return c(**kwargs)


def get_flattened_dim(in_dim):
    """
    Parameters
    ----------
        in_dim: int or list
            List of sizes of input as (batch, num_channels, height, width).
    Returns
    -------
        int
            Flattened version of in_dm (i.e., num_channels * height * width).
    """
    if isinstance(in_dim, int):
        return in_dim
    else:
        return reduce((lambda x, y: x * y), in_dim[1:])


def get_from_module(attrname):
    """
    Returns the Python class/method of the specified |attrname|.
    Typical usage pattern:
        m = get_class("this.module.MyClass")
        my_class = m(**kwargs)
    """
    parts = attrname.split('.')
    module = '.'.join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def init_weights(mod):
    """
    Initializes parameters for PyTorch module ``mod``. This should only be
    called when ``mod`` has been newly insantiated has not yet been trained.
    """
    if len(list(mod.modules())) == 0:
        return
    for m in mod.modules():
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(
                m.weight, gain=torch.nn.init.calculate_gain('relu'))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(
                m.weight, gain=torch.nn.init.calculate_gain('relu'))
            if m.bias is not None:
                m.bias.data.zero_()


def load_state(filename):
    """
    Loads the PyTorch files saved at ``filename``, converting them to be
    able to run with CPU or GPU, depending on the availability on the machine.
    """
    if torch.cuda.is_available():
        return torch.load(filename)
    else:
        return torch.load(filename, map_location=lambda storage, loc: storage)


def save_checkpoint(state_dict, save_dir, filename, is_best):
    """
    Serializes and saves dictionary ``state_dict`` to ``save_dir`` with name
    ``filename``. If parameter ``is_best`` is set to ``True``, then this
    dictionary is also saved under ``save_dir`` as "best.pth".
    """
    save_file = os.path.join(save_dir, filename)
    torch.save(state_dict, save_file)
    if is_best:
        copyfile(save_file, os.path.join(save_dir, 'best.pth'))


def try_cuda(x):
    """
    Sends PyTorch tensor or Variable ``x`` to GPU, if available.
    """
    if torch.cuda.is_available():
        return x.cuda()
    return x


def write_vals(outfile, vals, names):
    """
    Writes each value in ``vals[i]`` to a file with name formatted as
    ``outfile.format(names[i])``.
    """
    def write_value(val, outfilename):
        with open(outfilename, 'a') as outfile:
            outfile.write("{}\n".format(val))

    for v, n in zip(vals, names):
        write_value(v, outfile.format(n))


def write_vals_dict(outfile, val_dict):
    """
    Writes each value in ``val_dict`` to a file with name formatted as
    ``outfile.format(val_dict.keys()[i])``.
    """
    def write_value(val, outfilename):
        with open(outfilename, 'a') as outfile:
            outfile.write("{}\n".format(val))

    for k, v in val_dict.items():
        write_value(v, outfile.format(k))
