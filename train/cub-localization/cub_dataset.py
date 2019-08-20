import numpy as np
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CUBDataset(Dataset):
    def __init__(self, im_ids, transform=None):
        with open('src/data/images.txt') as f:
            id_to_path = dict([l.split(' ', 1) for l in f.read().splitlines()])
        with open('src/data/bounding_boxes.txt') as f:
            id_to_box = dict()
            for line in f.read().splitlines():
                im_id, *box = line.split(' ')
                id_to_box[im_id] = list(map(float, box))
        with open('src/data/base_bounding-boxes.txt') as f:
            base_id_to_box = dict()
            for line in f.read().splitlines():
                im_id, *box = line.split(' ')
                base_id_to_box[im_id] = list(map(float, box))

        self.imgs = [(os.path.join('src/data/images', id_to_path[i]), id_to_box[i], base_id_to_box[i])
                     for i in im_ids]
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Scale((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __getitem__(self, index):
        path, box, base_box = self.imgs[index]
        im = Image.open(path).convert('RGB')
        im_size = np.array(im.size, dtype='float32')
        box = np.array(box, dtype='float32')
        base_box = np.array(base_box, dtype='float32')

        im = self.transform(im)

        return im, box, base_box, im_size

    def __len__(self):
        return len(self.imgs)
