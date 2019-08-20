import torch
import torchvision
import numpy as np

from box_util import *
from cub_dataset import *


batch_size = 128

with open('src/data/images.txt') as f:
    len_lines = len(f.read().splitlines())

train_id = [str(i) for i in range(1, len_lines+1)]

splits = {'train': train_id}
datasets = {split: CUBDataset(splits[split]) for split in ('train',)}

# prepare data
dataloaders = {split: torch.utils.data.DataLoader(
                datasets[split], batch_size=batch_size,shuffle=False,
                num_workers=2, pin_memory=True) for split in ('train',)}

box_sum = 0
base_sum = 0
num_so_far = 0
for ims, boxes, base_boxes, im_sizes in dataloaders['train']:
    boxes = crop_boxes(boxes, im_sizes)
    boxes = box_transform(boxes, im_sizes)
    base_boxes = crop_boxes(base_boxes, im_sizes)
    base_boxes = box_transform(base_boxes, im_sizes)
    num_so_far += boxes.size(0)

    base_sum += boxes.sum(dim=0)
    box_sum += base_boxes.sum(dim=0)

print("base:", base_sum / num_so_far)
print("box:", box_sum / num_so_far)
