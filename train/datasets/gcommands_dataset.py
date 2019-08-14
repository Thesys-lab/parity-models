# Derived from https://github.com/adiyoss/GCommandsPytorch

import librosa
import numpy as np
import os
import torch
import torch.utils.data as data

import datasets.code_dataset as code_dataset

AUDIO_EXTENSIONS = [
    '.wav', '.WAV',
]


def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(
        dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    spects = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_audio_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    spects.append(item)
    return spects


class GCommandLoader(data.Dataset):
    """A dataset loader where the wavs are arranged in this way:
        root/one/xxx.wav
        root/one/xxy.wav
        root/one/xxz.wav
        root/head/123.wav
        root/head/nsdf3.wav
        root/head/asd932_.wav
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        window_size: window size for the stft, default value is .02
        window_stride: window stride for the stft, default value is .01
        window_type: typye of window to extract the stft, default value is 'hamming'
        normalize: boolean, whether or not to normalize the spect to have zero mean and one std
        max_len: the maximum length of frames to use
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        spects (list): List of (spects path, class_index) tuples
        STFT parameter: window_size, window_stride, window_type, normalize
    """

    def __init__(self, root):
        classes, class_to_idx = find_classes(root)
        spects = make_dataset(root, class_to_idx)
        self.root = root
        self.spects = spects
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.window_size = 0.02
        self.window_stride = 0.01
        self.window_type = "hamming"
        self.normalize = True
        self.max_len = 101

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (spect, target) where target is class_index of the target class.
        """
        path, target = self.spects[index]
        spect = self.loader(path)
        return spect, target

    def __len__(self):
        return len(self.spects)

    def loader(self, path):
        y, sr = librosa.load(path, sr=None)
        # n_fft = 4096
        n_fft = int(sr * self.window_size)
        win_length = n_fft
        hop_length = int(sr * self.window_stride)

        # STFT
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=self.window_type)
        spect, phase = librosa.magphase(D)

        # S = log(S+1)
        spect = np.log1p(spect)

        # Make all spects with the same dims
        if spect.shape[1] < self.max_len:
            pad = np.zeros((spect.shape[0], self.max_len - spect.shape[1]))
            spect = np.hstack((spect, pad))
        elif spect.shape[1] > self.max_len:
            spect = spect[:, :self.max_len]
        spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
        spect = torch.FloatTensor(spect)

        # z-score normalization
        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            if std != 0:
                spect.add_(-mean)
                spect.div_(std)

        return spect


class GCommandsCodeDataset(code_dataset.CodeDataset):
    def __init__(self, name, base_model, ec_k):
        # `name` will be one of {train, val, test}, but the GCommands dataset is
        # organized as {train, valid, test}.
        if "val" in name:
            root = "valid"
        else:
            root = name

        root = "data/gcommands/{}".format(root)
        base_dataset = GCommandLoader(root)
        super().__init__(name=name, base_model=base_model, ec_k=ec_k,
                         base_dataset=base_dataset,
                         num_classes=30,
                         put_gpu=False)
