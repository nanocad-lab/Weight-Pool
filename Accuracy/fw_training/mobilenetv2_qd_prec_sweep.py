import torch
import os
from torch import Tensor
import torch.nn as nn
#from .utils import load_state_dict_from_url
import torch.utils.model_zoo
from torch import utils
import argparse
import shutil
import time
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from typing import Type, Any, Callable, Union, List, Optional
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
import torchvision
from torchvision.transforms import Compose

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", help="specify the root directory, default is the 'accuracy_codes folder'",
                    default = ".." )
parser.add_argument("--epochs", help="number of epochs", type = int,
                    default = 50 )
args = parser.parse_args()
rootdir = args.root_dir
n_epoch = args.epochs

#check and create the weight and cluster center folder if not existed
if not os.path.isdir(rootdir):
    print("Error! Specified root directory does not exist!")
    quit()
weightfolder = os.path.join(rootdir,"weight_pool_weights")
ccfolder = os.path.join(rootdir,"cluster_centers")
if not os.path.isdir(weightfolder):
    print("weight folder does not exist!")
    quit()
if not os.path.isdir(ccfolder):
    print("cluster center folder does not exist!")
    quit()


accuracy_list = []
layer_max_list = []

max_val_list = [0.7522924542427063, 0.6374761462211609, 0.7522924542427063, 0.590674102306366, 1.5894120931625366, 0.590674102306366, 1.229587435722351, 1.5620826482772827, 1.636355996131897, 1.4107310771942139, 1.6372816562652588, 0.7559004426002502, 1.615250587463379, 0.84708571434021, 1.4444609880447388, 1.183425784111023, 0.9282110929489136, 0.3410528600215912, 0.9204574823379517, 0.6738632321357727, 0.9240310788154602, 0.4578166604042053, 0.9292969107627869, 0.5933653712272644, 0.9292969107627869, 1.0722099542617798, 0.5239707827568054, 1.1697547435760498, 0.598934531211853, 1.3050305843353271, 1.188023567199707, 0.568064272403717, 0.6141440868377686, 0.8530181646347046, 0.7007849216461182, 1.170894980430603, 1.1950786113739014, 1.170894980430603, 0.6559980511665344]
power_of_2s = np.array([0.125,0.5,1,2,4,8])
max_val_round = []
max_val_ceil = []
max_val_floor = []
max_val_floor_s = []
max_val_round_s = []
max_val_floor_ss = []
for value in max_val_list:
    diff = np.abs(value-power_of_2s)
    diff_ceil = power_of_2s-value
    diff_floor = value-power_of_2s
    diff_ceil[diff_ceil < 0] = 1000
    diff_floor[diff_floor < 0] = 1000
    closest = power_of_2s[np.argmin(diff)]
    closest_ceil = power_of_2s[np.argmin(diff_ceil)]
    closest_floor = power_of_2s[np.argmin(diff_floor)]
    max_val_round.append(closest)
    max_val_ceil.append(closest_ceil)
    max_val_floor.append(closest_floor)
    max_val_floor_s.append(closest_floor/2)
    max_val_round_s.append(closest/2)
    max_val_floor_ss.append(closest_floor/4)

act_config_all = [max_val_round, max_val_round, max_val_round, max_val_round, max_val_round_s,max_val_round_s]
result_holder = []
best_setup_holder = []

ccname = "mobilenetv2_qd_clustercenter_zdim64.npy"
cluster_path = os.path.join(ccfolder,ccname)
clustercenter = np.load(cluster_path)
clustercenter = torch.from_numpy(clustercenter)

for idx, act_prec in enumerate([8,7,6,5,4,3]):  
    temp_best = 0
    temp_setup = None

    class _ConvNd(Module):

        __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
                        'padding_mode', 'output_padding', 'in_channels',
                        'out_channels', 'kernel_size']   

        def __init__(self, in_channels, out_channels, kernel_size, batch_size, stride,
                    padding, dilation, transposed, output_padding,
                    groups, bias, padding_mode, layer_idx):
            super(_ConvNd, self).__init__()
            if in_channels % groups != 0:
                raise ValueError('in_channels must be divisible by groups')
            if out_channels % groups != 0:
                raise ValueError('out_channels must be divisible by groups')
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.batch_size = batch_size
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.transposed = transposed
            self.output_padding = output_padding
            self.groups = groups
            self.padding_mode = padding_mode
            self.layer_idx = layer_idx
            if transposed:
                self.weight = Parameter(torch.Tensor(
                    in_channels, out_channels // groups, *kernel_size))
            else:
                self.weight = Parameter(torch.Tensor(
                    out_channels, in_channels // groups, *kernel_size))
            if bias:
                self.bias = Parameter(torch.Tensor(out_channels))
            else:
                self.register_parameter('bias', None)
            self.reset_parameters()

        def reset_parameters(self):
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            #init.xavier_uniform_(self.weight)
            if self.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

        def extra_repr(self):
            s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
                ', stride={stride}')
            if self.padding != (0,) * len(self.padding):
                s += ', padding={padding}'
            if self.dilation != (1,) * len(self.dilation):
                s += ', dilation={dilation}'
            if self.output_padding != (0,) * len(self.output_padding):
                s += ', output_padding={output_padding}'
            if self.groups != 1:
                s += ', groups={groups}'
            if self.bias is None:
                s += ', bias=False'
            return s.format(**self.__dict__)

        def __setstate__(self, state):
            super(_ConvNd, self).__setstate__(state)
            if not hasattr(self, 'padding_mode'):
                self.padding_mode = 'zeros'


    def quantization_formal(input, n = 8, rangeq = 1):
        #updated version so that the minimal interval are integer power of two, so it fits better with stream gen function
        #actual bitwidth is 1 larger than the input bitwidth due to extened range over negative parts
        maxpower = int(math.log(rangeq,2))
        minpower = maxpower-(n)
        max = rangeq-2**minpower
        intv = (rangeq-2**minpower)/(2**n-1)
        qunt = torch.round(torch.mul(input,(1/intv))) 
        #the above line divide the whole tensor by the smallest interval (1/2**n-1), which is same as multiply with 2**n-1, then take the floor and finally multiply the whole tensor with the smallest interval
        out = torch.mul(qunt,intv)
        out = torch.clamp(out, min=-rangeq, max=max) #make sure the quantized version lies in the interval 0-1, if it's bigger than one just clamp it at one
        return(out)

    def find_optimal_l2(input, kernel_pool): #tested
        #input shape is (K*C,9), kernel_pool shape is (N,9)
        #vectorized method will be quicker, but will be more memory consuming (using float for a 512*512*3*3 layer with pool size 100 leads to ~900MB*3 memory)
        kernel_pool = kernel_pool.unsqueeze(0).repeat(input.shape[0],1,1)#shape will be (K*C,N,9)
        input = input.unsqueeze(1)#shape will be (K*C,1,9)
        result_stacked = kernel_pool - input #shape will be (K*C,N,9)
        mse = torch.sum(result_stacked**2, 2, keepdim = True)/result_stacked.shape[2]
        minidx = torch.argmin(mse, dim = 1 , keepdim = True)#find the minimal value for dimension 1 (N), shape should be (K*C,1,1)
        minidx_pe = minidx.repeat(1, 1, input.shape[2])
        result = kernel_pool.gather(1, minidx_pe).squeeze(1)#output should have size (K*C,9)
        return result

    def find_optimal_cossim(input, kernel_pool): #tested
        #input shape is (K*C,9), kernel_pool shape is (N,9)
        #vectorized method will be quicker, but will be more memory consuming (using float for a 512*512*3*3 layer with pool size 100 leads to ~900MB*3 memory)
        kernel_pool = kernel_pool.unsqueeze(0).repeat(input.shape[0],1,1)#shape will be (K*C,N,9)
        input = input.unsqueeze(1)#shape will be (K*C,1,9)
        input_repeat = input.repeat(1,kernel_pool.shape[1],1)#shape will be (K*C,N,9)
        result_stacked = F.cosine_similarity(input_repeat,kernel_pool,dim=2).unsqueeze(2) #shape should be (K*C,N,1)
        minidx = torch.argmax(result_stacked, dim = 1 , keepdim = True)#find the max value for dimension 1 (N), shape should be (K*C,1,1)
        minidx_pe = minidx.repeat(1, 1, input.shape[2])
        result = kernel_pool.gather(1, minidx_pe).squeeze(1)#output should have size (K*C,9)
        return result

    def find_optimal_mixednorm(input, kernel_pool): #tested
        coeff = 1 #coefficient for suqared signed error and l2 norm
        #input shape is (K*C,9), kernel_pool shape is (N,9)
        #vectorized method will be quicker, but will be more memory consuming (using float for a 512*512*3*3 layer with pool size 100 leads to ~900MB*3 memory)
        kernel_pool = kernel_pool.unsqueeze(0).repeat(input.shape[0],1,1)#shape will be (K*C,N,9)
        input = input.unsqueeze(1)#shape will be (K*C,1,9)
        result_stacked = kernel_pool - input #shape will be (K*C,N,9)
        mse = torch.sum(result_stacked**2, 2, keepdim = True)
        signederr = torch.abs(torch.sum(result_stacked,2, keepdim=True))**2
        meanval_pe = (signederr*coeff + mse)/result_stacked.shape[2]
        minidx = torch.argmin(meanval_pe, dim = 1 , keepdim = True)#find the minimal value for dimension 1 (N), shape should be (K*C,1,1)
        minidx_pe = minidx.repeat(1, 1, input.shape[2])
        result = kernel_pool.gather(1, minidx_pe).squeeze(1)#output should have size (K*C,9)
        return result
        

    def select_kernel(input, kernel_pool):
        #kernel_pool should have size of (N,9) for VGG network
        #input should have size (K,C,X,Y)/(K,C,3,3)
        input_reshaped = torch.reshape(input, (input.shape[0]*input.shape[1],input.shape[2]*input.shape[3]))
        output = find_optimal_cossim(input_reshaped, kernel_pool).reshape(input.shape[0],input.shape[1],input.shape[2],input.shape[3])
        return output

    def select_kernel_channelwise(input, kernel_pool):
        #kernel_pool should have size of (N,9) for VGG network
        #input should have size (K,X,Y,C)/(K,3,3,C)
        orishape = input.shape
        input_reshaped = torch.reshape(input, (int(input.shape[0]*input.shape[1]*input.shape[2]*input.shape[3]/8), 8))
        output = find_optimal_cossim(input_reshaped, kernel_pool).reshape(orishape)
        return output

    def kernelpool_gen_random(kernel_size,pool_size):
        #kernelpool = torch.rand((pool_size,kernel_size))-torch.rand((pool_size,kernel_size))#make range -1 to 1
        kernelpool = torch.normal(0, 0.5, size=(pool_size,kernel_size))
        return kernelpool

    #kernelpool = kernelpool_gen_random(9,50).cuda() #generate 50 random kernels
    kernelpool = clustercenter.cuda()#load kernel pool using cluster centers from K means clustering

    coeff_groupsize = 1 #number of kernels to share a same coeff

    class Conv2d_q(_ConvNd):    
        def __init__(self, in_channels, out_channels, kernel_size, batch_size = 32,stride=1,
                    padding=0, dilation=1, groups=1,
                    bias=True, padding_mode='zeros', layer_idx = 0):
            batch_size = 32
            kernel_size_ori = kernel_size
            kernel_size = _pair(kernel_size)
            stride = _pair(stride)
            padding = _pair(padding)
            dilation = _pair(dilation)
            super(Conv2d_q, self).__init__(
                in_channels, out_channels, kernel_size, batch_size, stride, padding, dilation,
                False, _pair(0), groups, bias, padding_mode, layer_idx)
            self.coeff = nn.Parameter(torch.ones(int(in_channels*out_channels/coeff_groupsize)))
            #self.coeff = nn.Parameter(torch.ones(int(in_channels*out_channels*kernel_size_ori*kernel_size_ori/(8*coeff_groupsize))))
            #self.sharedkernelcoeff = nn.Parameter(torch.ones(in_channels))
            #self.layercoeff = nn.Parameter(torch.ones(1))
            

        def _conv_forward(self, input, weight, coeff):
            act_config = act_config_all[idx]
            act_max = act_config[self.layer_idx]
            input = quantization_formal(input,act_prec-1, act_max)
            # for coefficients for each kernel
            weight = weight.permute(0,2,3,1)
            permuteshape = weight.shape
            #weight.data = select_kernel(weight.data,kernelpool_layer)
            weight.data = select_kernel_channelwise(weight.data,kernelpool)
            weight = weight.permute(0,3,1,2)
            
            
            #print("------------------------------------------")
            if self.padding_mode != 'zeros':
                return (F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                                weight, self.bias, self.stride,
                                _pair(0), self.dilation, self.groups))
            return (F.conv2d(input, weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups))

        def forward(self, input):
            return self._conv_forward(input, self.weight, self.coeff)

    class Block(nn.Module):
        '''expand + depthwise + pointwise'''
        def __init__(self, in_planes, out_planes, expansion, stride, layer_idx = 0):
            super(Block, self).__init__()
            self.stride = stride

            planes = expansion * in_planes
            self.conv1 = Conv2d_q(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False, layer_idx = layer_idx)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = Conv2d_q(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False, layer_idx = layer_idx+1)
            self.bn3 = nn.BatchNorm2d(out_planes)

            self.shortcut = nn.Sequential()
            if stride == 1 and in_planes != out_planes:
                self.shortcut = nn.Sequential(
                    Conv2d_q(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False, layer_idx = layer_idx+2),
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
              (6,  32, 3, 2),
              (6,  64, 4, 2),
              (6,  96, 3, 1),
              (6, 160, 3, 2),
              (6, 320, 1, 1)]

        def __init__(self, num_classes=100):
            super(MobileNetV2, self).__init__()
            layer_idx = 0
            # NOTE: change conv1 stride 2 -> 1 for CIFAR10
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(32)
            self.layers = self._make_layers(in_planes=32, layer_idx = layer_idx)
            self.conv2 = Conv2d_q(320, 1280, kernel_size=1, stride=1, padding=0, bias=False, layer_idx = 38)
            self.bn2 = nn.BatchNorm2d(1280)
            self.linear = nn.Linear(1280, num_classes)

        def _make_layers(self, in_planes, layer_idx):
            layers = []
            for expansion, out_planes, num_blocks, stride in self.cfg:
                strides = [stride] + [1]*(num_blocks-1)
                for stride in strides:
                    layers.append(Block(in_planes, out_planes, expansion, stride, layer_idx = layer_idx))
                    if stride == 1 and in_planes != out_planes:
                        layer_idx = layer_idx + 3
                    else:
                        layer_idx = layer_idx + 2
                    in_planes = out_planes
            return nn.Sequential(*layers)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
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


    def adjust_learning_rate(optimizer, epoch, lr_init, n):
        """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
        lr = lr_init * (0.5 ** (epoch // n))
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
    workers = 4

    data_root = '../QuickDraw-pytorch/Dataset'

    train_data = QD_Dataset(mtype="train", root=data_root)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last = True)

    test_data = QD_Dataset(mtype="test", root=data_root)
    val_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True, drop_last = True)

    num_classes = train_data.get_number_classes()

    PATH = os.path.join(weightfolder,'mobilenetv2_qd_zdim64.pth')
    state_dict=torch.load(PATH)
    model.load_state_dict(state_dict)

    init_lr = 0.001
    decay_val = 0.2 #how much weight decay is applied when activated
    #optimizer = torch.optim.Adam(model.parameters(), lr =  init_lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002,
                    momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)
    starttime = time.time()
    inferenceacc = validate(val_loader, model, criterion)
    print("Direct inference accuracy is ",inferenceacc)
    
    best_prec1 = inferenceacc
    if act_prec in [5,4,3]:
        print("Retraining starts!")
        weight_name = "tinyconv_qd_zdim64_retrained_"+str(act_prec)+"bit.pth"
        SAVEPATH = os.path.join(weightfolder,weight_name)     
        for epoch in range(n_epoch):
            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch)
            
            # evaluate on validation set
            prec1 = validate(val_loader, model, criterion)

            # remember best prec@1 and save checkpoint
            if (prec1 > best_prec1):
                #Save the weight for the best accuracy 
                torch.save(model.state_dict(), SAVEPATH)
                print("best accuracy achieved, weight saved, accuracy is ", prec1)
            best_prec1 = max(prec1, best_prec1)
            scheduler.step()
    result_holder.append(best_prec1)
    
print("the weight pool accuracy for specified bitwidth is ",result_holder)