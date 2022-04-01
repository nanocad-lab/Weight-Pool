import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch import utils
from typing import Type, Any, Callable, Union, List, Optional
import time
import importlib
import json
from collections import OrderedDict
import logging
import argparse
import numpy as np
import random
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.optim
import torch.backends.cudnn as cudnn
import torchvision
from torchvision.transforms import Compose

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", help="specify the root directory, default is the 'accuracy_codes folder'",
                    default = ".." )
parser.add_argument("--epochs", help="number of epochs", type = int,
                    default = 300 )
args = parser.parse_args()
rootdir = args.root_dir
n_epoch = args.epochs

#check and create the weight and cluster center folder if not existed
if not os.path.isdir(rootdir):
    print("Error! Specified root directory does not exist!")
    quit()
weightfolder = os.path.join(rootdir,"pretrained_weights")
ccfolder = os.path.join(rootdir,"cluster_centers")
if not os.path.isdir(weightfolder):
    print("weight folder does not exist, the program will create the folder")
    os.mkdir(weightfolder)
if not os.path.isdir(ccfolder):
    print("cluster center folder does not exist, the program will create the folder")
    os.mkdir(ccfolder)

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 1),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=100):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)
        self.pool = nn.MaxPool2d(2)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool(out)
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

model = MobileNetV2().cuda()

print_freq = 1000
def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter() 

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)
        input = input.view(-1, 1, 28, 28)
        input /= 255.0

        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output  .float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        #print('new batch')
        input = input.view(-1, 1, 28, 28)
        input /= 255.0
        input = input.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, init_lr, freq):
    """Sets the learning rate to the initial LR decayed by 2 every n epochs"""
    lr = init_lr * (0.5 ** (epoch // freq))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0)
      res.append(correct_k.mul_(100.0 / batch_size))
  return res

def lr_schedule(optimizer, epoch):
    initial_learning_rate = 0.001
    decay_per_epoch = 0.99
    lrate = initial_learning_rate * (decay_per_epoch ** epoch)
    print('Learning rate = %f'%lrate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lrate
criterion = nn.CrossEntropyLoss(size_average=True).cuda()

def load_dataset(root, mtype):
    num_classes = 0
    qd_classname_path = "../QuickDraw-pytorch/DataUtils/class_names.txt"
    with open(qd_classname_path, "r") as f:
        for line in f:
            num_classes = num_classes+1

    # load data from cache
    if os.path.exists(os.path.join(root, mtype+'.npz')):
        print("*"*50)
        print("Loading "+mtype+" dataset...")
        print("*"*50)
        print("Classes number of "+mtype+" dataset: "+str(num_classes))
        print("*"*50)
        data_cache = np.load(os.path.join(root, mtype+'.npz'))
        return data_cache["data"].astype('float32'), \
            data_cache["target"].astype('int64'), num_classes

    else:
        raise FileNotFoundError("%s doesn't exist!" %
                                os.path.join(root, mtype+'.npz'))
        

class QD_Dataset(data.Dataset):
    def __init__(self, mtype, root='Dataset'):
        """
        args:
        - mytpe: str, specify the type of the dataset, i.e, 'train' or 'test'
        - root: str, specify the root of the dataset directory
        """

        self.data, self.target, self.num_classes = load_dataset(root, mtype)
        self.data = torch.from_numpy(self.data)
        self.target = torch.from_numpy(self.target)
        print("Dataset "+mtype+" loading done.")
        print("*"*50+"\n")

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)

    def get_number_classes(self):
        return self.num_classes

train_transform = Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    #transforms.Normalize([0, 0, 0], [1, 1, 1])
])

test_transform = Compose([
    transforms.ToTensor(),
    #transforms.Normalize([0, 0, 0], [1, 1, 1])
])

batch_size = 128
workers = 16

data_root = '../QuickDraw-pytorch/Dataset'

train_data = QD_Dataset(mtype="train", root=data_root)
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True, drop_last = True)

test_data = QD_Dataset(mtype="test", root=data_root)
val_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=True, drop_last = True)

num_classes = train_data.get_number_classes()

init_lr = 0.001
decay_val = 0.2 #how much weight decay is applied when activated
#optimizer = torch.optim.Adam(model.parameters(), lr =  init_lr)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)
best_prec1 = 0
total_time = 0
PATH = os.path.join(weightfolder,'mobilenetv2_qd.pth')

for epoch in range(n_epoch):
    starttime = time.time()
    #adjust_learning_rate(optimizer, epoch, init_lr, 20)
    #lr_schedule(optimizer, epoch)

    # train for one epoch
    train(train_loader, model, criterion, optimizer, epoch)

    # evaluate on validation set
    prec1 = validate(val_loader, model, criterion)
    #result_list.append(prec1)

    # remember best prec@1 and save checkpoint
    if (prec1 > best_prec1):
        #Save the weight for the best accuracy 
        torch.save(model.state_dict(), PATH)
        print("best accuracy achieved, weight saved, accuracy is ", prec1)
    best_prec1 = max(prec1, best_prec1)
    endtime = time.time()
    elapsedtime = endtime - starttime
    total_time = total_time + elapsedtime
    total_time_h = total_time/3600
    print("epoch time is: ", elapsedtime, "s, current best accuracy is ",best_prec1)
    scheduler.step()


from kmeans_pytorch import kmeans
'''
version for you channel-wise pool
'''
combine_size = 8
state_dict=torch.load(PATH)
for name, param in state_dict.items():
    if "conv" in name and param.shape[1] != 1 and name != 'conv1.weight':
        print(name, param.shape)
        #extract the actual weight tensor
        param = param.permute(0,2,3,1)
        #extract the actual weight tensor
        orishape = param.shape
        #print(orishape)
        param = param.reshape(int(orishape[0]*orishape[1]*orishape[2]*orishape[3]/combine_size),combine_size)
        if 'layer' in name and 'conv' in name:
            if name == "layers.0.conv1.weight":
                stacked_weight = param
            else:
                stacked_weight = torch.cat((stacked_weight,param),0)
print(stacked_weight.shape)
stacked_weight = stacked_weight.cpu().detach().numpy()

for n_cluster in [32,64,128]:
    data_size, dims, num_clusters = stacked_weight.shape[0], 2, n_cluster
    x = torch.from_numpy(stacked_weight)
    if n_cluster == 128:
        tol = 1e-3
    else:
        tol = 1e-4

    # kmeans 
    cluster_ids_x, cluster_centers_cos = kmeans(
        X=x, num_clusters=num_clusters, distance='cosine', device=torch.device('cuda:0'), tol = tol
    )
    ccname = "mobilenetv2_qd_clustercenter_zdim" + str(n_cluster) + ".npy"
    output_path = os.path.join(ccfolder,ccname)
    np.save(output_path, cluster_centers_cos)

print(best_prec1)