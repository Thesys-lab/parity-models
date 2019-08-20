import os
import random
import time

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models

from box_util import *
from cub_dataset import *


train_id, test_id = split(0.2)

splits = {'train': train_id, 'test': test_id}
datasets = {split: CUBDataset(splits[split]) for split in ('train', 'test')}

# prepare data
dataloaders = {split: torch.utils.data.DataLoader(
                datasets[split], batch_size=32,shuffle=(split=='train'),
                num_workers=2, pin_memory=True) for split in ('train', 'test')}

# construct model
model = models.resnet18(pretrained=False)
fc_in_size = model.fc.in_features
model.fc = nn.Linear(fc_in_size, 4)
criterion = nn.SmoothL1Loss()
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_model_state = model.state_dict()
best_epoch = -1
best_acc = 0.0

epoch_loss = {'train': [], 'test': []}
epoch_acc = {'train': [], 'test': []}
epochs = 20
print(4)
for epoch in range(160):
    accs = AverageMeter()
    losses = AverageMeter()
    for phase in ('train', 'test'):
        if phase == 'train':
            model.train(True)
        else:
            model.train(False)

        end = time.time()
        for ims, boxes, _, im_sizes in dataloaders[phase]:
            boxes = crop_boxes(boxes, im_sizes)
            boxes = box_transform(boxes, im_sizes)

            inputs = Variable(ims)
            targets = Variable(boxes)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()

            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc, _, _, _ = compute_acc(outputs.data.cpu(), targets.data.cpu(), None, im_sizes, epoch=epoch)

            nsample = inputs.size(0)
            accs.update(acc, nsample)
            losses.update(loss.item(), nsample)

            if phase == 'train':
                loss.backward()
                optimizer.step()

        if phase == 'test' and accs.avg > best_acc:
            best_acc = accs.avg
            best_epoch = epoch
            best_model_state = model.state_dict()

        elapsed_time = time.time() - end
        print('[{}]\tEpoch: {}/{}\tLoss: {:.4f}\tAcc: {:.4%}\tTime: {:.3f}'.format(
            phase, epoch+1, epochs, losses.avg, accs.avg, elapsed_time))
        epoch_loss[phase].append(losses.avg)
        epoch_acc[phase].append(accs.avg)

    print('[Info] best test acc: {:.2%} at {}th epoch'.format(best_acc, best_epoch))
    torch.save(best_model_state, 'best_model_state.path.tar')
