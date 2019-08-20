import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, models

from box_util import *
from cub_dataset import *


with open('src/data/images.txt') as f:
    num_lines = len(f.read().splitlines())

train_id = [str(i) for i in range(1, num_lines+1)]

splits = {'train': train_id}
datasets = {split: CUBDataset(splits[split]) for split in ('train',)}

# prepare data
dataloaders = {split: torch.utils.data.DataLoader(
                datasets[split], batch_size=32,shuffle=False,
                num_workers=2, pin_memory=True) for split in ('train',)}

# construct model
model = models.resnet18(pretrained=False)
fc_in_size = model.fc.in_features
model.fc = nn.Linear(fc_in_size, 4)
model.load_state_dict(torch.load('best_model_state.path.tar'))
model = model.cuda()
criterion = nn.SmoothL1Loss().cuda()

best_model_state = model.state_dict()
best_epoch = -1
best_acc = 0.0

epoch_loss = {'train': [], 'test': []}
epoch_acc = {'train': [], 'test': []}
epochs = 20
bboxes = []
for phase in ('train',):
    model.train(False)

    with torch.no_grad():
        for ims, boxes, im_sizes in dataloaders[phase]:
            boxes = crop_boxes(boxes, im_sizes)
            boxes = box_transform(boxes, im_sizes)

            inputs = Variable(ims.cuda())
            targets = Variable(boxes.cuda())

            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            preds = box_transform_inv(outputs.data.cpu().clone(), im_sizes)
            preds = crop_boxes(preds, im_sizes)
            for bbox in preds:
                bboxes.append(bbox.numpy())

with open("base_bounding-boxes.txt", 'w') as outfile:
    for i, bbox in enumerate(bboxes):
        outfile.write("{} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(
            i+1, bbox[0], bbox[1], bbox[2], bbox[3]))
