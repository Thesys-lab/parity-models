import os
from shutil import copyfile
import time
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

from box_util import *
from cub_dataset import *


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


ec_k = 2
epochs = 1000
outdir = "coded-k2"
batch_size = 128


if not os.path.isdir(outdir):
    os.makedirs(outdir)

train_id, test_id = split(0.2)
splits = {'train': train_id, 'test': test_id}
datasets = {split: CUBDataset(splits[split]) for split in ('train', 'test')}
dataloaders = {split: torch.utils.data.DataLoader(
                datasets[split], batch_size=batch_size,shuffle=(split=='train'),
                num_workers=2, pin_memory=True) for split in ('train', 'test')}

model = models.resnet18(pretrained=False)
fc_in_size = model.fc.in_features
model.fc = nn.Linear(fc_in_size, 4)
init_weights(model)

# Initialize final bias to be equal to the the sum of ec_k
# average bounding boxes from the training dataset.
model.fc.bias.data = torch.Tensor([-0.5591, -0.5680,  1.1064,  1.2060]) * ec_k

criterion = nn.MSELoss()
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-05)

best_model_state = model.state_dict()
best_epoch = -1
best_acc = 0.0

epoch_loss = {'train': [], 'test': []}
epoch_acc = {'train': [], 'test': []}
epoch_recon_acc = {'train': [], 'test': []}
epoch_iou = {'train': [], 'test': []}
epoch_recon_iou = {'train': [], 'test': []}

for epoch in range(epochs):
    accs = AverageMeter()
    recon_accs = AverageMeter()
    ious = AverageMeter()
    recon_ious = AverageMeter()
    losses = AverageMeter()
    for phase in ('train', 'test'):
        if phase == 'train':
            model.train(True)
        else:
            model.train(False)

        end = time.time()
        pbar = tqdm(dataloaders[phase],
                    ascii=True,
                    desc="Epoch {}. {}".format(epoch, phase))

        for ims, boxes, base_boxes, im_sizes in pbar:
            if ims.size(0) % ec_k != 0:
                # Chop off any batches that aren't a multiple of `ec_k`.
                if ims.size(0) > ec_k:
                    num_full = (ims.size(0) // ec_k) * ec_k
                    ims = ims[:num_full]
                    boxes = boxes[:num_full]
                    base_boxes = base_boxes[:num_full]
                    im_sizes = im_sizes[:num_full]
                else:
                    # Not enough samples in batch to fill up a full
                    # coding group.
                    continue

            boxes = crop_boxes(boxes, im_sizes)
            boxes = box_transform(boxes, im_sizes)
            base_boxes = crop_boxes(base_boxes, im_sizes)
            base_boxes = box_transform(base_boxes, im_sizes)

            inputs = ims
            targets = boxes
            base_targets = base_boxes

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()
                base_targets = base_targets.cuda()

            # Encode inputs
            reshaped_ims = inputs.view(-1, ec_k, *list(inputs.size()[1:]))
            encoded_ims = torch.sum(reshaped_ims, dim=1)

            optimizer.zero_grad()

            # forward
            outputs = model(encoded_ims)

            # Sum together base_targets
            summed_base = torch.sum(base_targets.view(-1, ec_k, *list(base_targets.size()[1:])), dim=1)
            loss = criterion(outputs, summed_base)

            # Decode to get outputs
            repeated_outs = outputs.repeat(1, ec_k).view(-1, *list(base_targets.size()[1:]))
            grouped_base_targets = base_targets.view(-1, ec_k, 4)
            to_sub = grouped_base_targets.flip(1).view(-1, 4)
            decoded = repeated_outs - to_sub

            acc, recon_acc, iou, recon_iou = compute_acc(decoded.data.cpu(),
                                                         targets.data.cpu(),
                                                         base_targets.data.cpu(),
                                                         im_sizes,
                                                         epoch=epoch)

            nsample = inputs.size(0)
            accs.update(acc, nsample)
            recon_accs.update(recon_acc, nsample)
            ious.update(iou, nsample)
            recon_ious.update(recon_iou, nsample)
            losses.update(loss.item(), nsample)

            pbar.set_description(
                    "Epoch {}. {}. IoU={:.4f}".format(
                    epoch, phase, ious.med()))

            if phase == 'train':
                loss.backward()
                optimizer.step()

        if phase == 'test':
            save_dict = {
                "acc": accs.avg,
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": optimizer.state_dict()
            }
            save_file = os.path.join(outdir, "current.pth")
            torch.save(save_dict, save_file)
            if accs.avg > best_acc:
                best_acc = accs.avg
                best_epoch = epoch
                copyfile(save_file, os.path.join(outdir, "best.pth"))

        elapsed_time = time.time() - end
        print('[{}]\tEpoch: {}/{}\tLoss: {:.4f}\tAcc: {:.4%}\tRecon. Acc: {:.4%}\tIoU: {:.4%}\tRecon. IoU: {:.4%}\tTime: {:.3f}'.format(
            phase, epoch+1, epochs, losses.avg, accs.avg, recon_accs.avg, ious.med(), recon_ious.med(), elapsed_time))
        epoch_loss[phase].append(losses.avg)
        epoch_acc[phase].append(accs.avg)
        epoch_recon_acc[phase].append(recon_accs.avg)
        epoch_iou[phase].append(ious.med())
        epoch_recon_iou[phase].append(recon_ious.med())


    print('[Info] best test acc: {:.2%} at {}th epoch'.format(best_acc, best_epoch))
