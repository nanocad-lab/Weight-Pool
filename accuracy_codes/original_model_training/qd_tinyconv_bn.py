import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple
import torch.nn.functional as F
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", help="specify the root directory, default is the 'accuracy_codes folder'",
                    default = ".." )
parser.add_argument("--epochs", help="number of epochs", type = int,
                    default = 200 )
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

class CONV_tiny_quant(nn.Module):
    def __init__(self, uniform=False):
        super(CONV_tiny_quant, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=4, bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2, bias=False)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=False)
        #self.fc1 = Linear_q(4*4*64, 10, bias=False)
        self.fc1 = nn.Linear(4*4*64, 100, bias=False)

        self.tanh = nn.Hardtanh()
        self.pool = nn.AvgPool2d(2)
        self.bn1 = nn.BatchNorm2d(32, affine=True)
        self.bn2 = nn.BatchNorm2d(32, affine=True)
        self.bn3 = nn.BatchNorm2d(64, affine=True)
        self.bn4 = nn.BatchNorm1d(100, affine=True)
        

        self.err_used = [3,3,3,3]
        if uniform:
            self.conv1.weight_org.uniform_(-1,1)
            self.conv2.weight_org.uniform_(-1,1)
            self.conv3.weight_org.uniform_(-1,1)
            self.fc1.weight_org.uniform_(-1,1)
            self.conv1.weight.data.uniform_(-1,1)
            self.conv2.weight.data.uniform_(-1,1)
            self.conv3.weight.data.uniform_(-1,1)
            self.fc1.weight.data.uniform_(-1,1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.tanh(x)
        x = self.bn1(x)
#         x = self.tanh(self.pool(x))

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.tanh(x)
        x = self.bn2(x)
#         x = self.tanh(self.pool(x))

        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.tanh(x)
        x = self.bn3(x)
#         x = self.tanh(self.pool(x))

        x = x.view(-1, 4*4*64)
        x = self.fc1(x)
        x = self.bn4(x)
        return x

model = CONV_tiny_quant()
model.cuda()

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

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

batch_size = 128
workers = 2

data_root = '../QuickDraw-pytorch/Dataset'

train_data = QD_Dataset(mtype="train", root=data_root)
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True, drop_last = True)

test_data = QD_Dataset(mtype="test", root=data_root)
val_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=True, drop_last = True)

num_classes = train_data.get_number_classes()
'''
train_data = pyvww.pytorch.VisualWakeWordsClassification(root="/content/coco/all2014", annFile="/content/vww/annotations/instances_train.json")
val_data = pyvww.pytorch.VisualWakeWordsClassification(root="/content/coco/all2014", annFile="/content/vww/annotations/instances_val.json")


train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size, shuffle=True,
    num_workers=workers, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=batch_size, shuffle=False,
    num_workers=workers, pin_memory=True)
'''
#load the quick draw dataset

# define loss function (criterion) and pptimizer
criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()
'''
if args.half:
    model.half()
    criterion.half()
'''

#optimizer = torch.optim.SGD(model.parameters(), args.lr,momentum=args.momentum, weight_decay=args.weight_decay)

print(num_classes)

PATH = os.path.join(weightfolder,'tinyconv_qd.pth')
best_prec1 = 0

#optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01,momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)
for epoch in range(n_epoch):
    #adjust_learning_rate(optimizer, epoch, 0.001, 5)

    # train for one epoch
    train(train_loader, model, criterion, optimizer, epoch)

    # evaluate on validation set
    prec1 = validate(val_loader, model, criterion)

    # remember best prec@1 and save checkpoint
    is_best = prec1 > best_prec1
    if(is_best):
        torch.save(model.state_dict(), PATH)
        print("best accuracy achieved, weight saved, accuracy is ", prec1)
    best_prec1 = max(prec1, best_prec1)
    scheduler.step()


from kmeans_pytorch import kmeans
import torch.utils.model_zoo
from torch import utils

#cluster generation for z dimension
#version for you channel-wise pool
combine_size = 8

state_dict=torch.load(PATH)
#state_dict = utils.model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth',progress=True)
for name, param in state_dict.items():
    if "conv" in name and len(param.shape) == 4:
        print(name, param.shape)
        #extract the actual weight tensor
        param = param.permute(0,2,3,1)
        #extract the actual weight tensor
        orishape = param.shape
        print(orishape)
        param = param.reshape(int(orishape[0]*orishape[1]*orishape[2]*orishape[3]/combine_size),combine_size)
        if "conv1" not in name:
            if name == "conv2.weight":#for tinyconv
                stacked_weight = param
            else:
                stacked_weight = torch.cat((stacked_weight,param),0)
print(stacked_weight.shape)
stacked_weight = stacked_weight.cpu().detach().numpy()

for n_cluster in [32,64,128]:
    data_size, dims, num_clusters = stacked_weight.shape[0], 2, n_cluster
    x = torch.from_numpy(stacked_weight)
    if n_cluster == 128:
        tol = 1e-4
    else:
        tol = 1e-4

    # kmeans 
    cluster_ids_x, cluster_centers_cos = kmeans(
        X=x, num_clusters=num_clusters, distance='cosine', device=torch.device('cuda:0'), tol = tol
    )
    ccname = "tinyconv_qd_clustercenter_zdim" + str(n_cluster) + ".npy"
    output_path = os.path.join(ccfolder,ccname)
    np.save(output_path, cluster_centers_cos)