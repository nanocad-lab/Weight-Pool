import torch
from torch import Tensor
import torch.nn as nn
#from .utils import load_state_dict_from_url
import torch.utils.model_zoo
from torch import utils
from typing import Type, Any, Callable, Union, List, Optional
import os
import time
import importlib
import shutil
import json
from collections import OrderedDict
import logging
import argparse
import numpy as np
import random
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.optim
import torch.backends.cudnn as cudnn
import math
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple
import torch.nn.functional as F
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



ccname = "tinyconv_qd_clustercenter_zdim64.npy"
cluster_path = os.path.join(ccfolder,ccname)
clustercenter = np.load(cluster_path)
clustercenter = torch.from_numpy(clustercenter)

print_freq = 1000

#list of activation max values, profiled by checking the actual activation maximum values, mid is assign to nearest power of 2 (has clipping effect), max is always ceil to the next power of 2 results
act_config_all = [[4,8]]
result_holder = []
setup_holder = []

maxval = 1
for act_prec in [8]:
  act_prec = act_prec-1 #adjust to fit the quantization funciton which effectively adds one more bit
  for psumbw in [16,8,4]: 
    temp_best = 0
    best_setup = None
    for act_config in act_config_all:
      for maxval in [0.25,0.5]:
    
        class _ConvNd(Module):

            __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
                            'padding_mode', 'output_padding', 'in_channels',
                            'out_channels', 'kernel_size']   

            def __init__(self, in_channels, out_channels, kernel_size, batch_size, stride,
                        padding, dilation, transposed, output_padding,
                        groups, bias, padding_mode, layer_cnt):
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
                self.layer_cnt = layer_cnt
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

        def quantization_n_formal(input, n = 8, rangeq = 1):
            #updated version so that the minimal interval are integer power of two, so it fits better with stream gen functio
            #support both positive and negative input, assumes a sign bit that not count into total bits
            maxpower = int(math.log(rangeq,2))
            minpower = maxpower-(n)
            max = rangeq-2**minpower
            intv = (rangeq-2**minpower)/(2**n-1)
            qunt = torch.round(torch.mul(input,(1/intv)))  #***use ceil instead of floor for small value of n to make sure that not all weights are modified to zero in the beginning, achieves good accuracy for small n
            #the above line divide the whole tensor by the smallest interval (1/2**n-1), which is same as multiply with 2**n-1, then take the floor and finally multiply the whole tensor with the smallest interval
            out = torch.mul(qunt,intv)
            out = torch.clamp(out, min=-rangeq, max=max) #make sure the quantized version lies in the interval 0-1, if it's bigger than one just clamp it at one
            return(out)

        def bit_stream_gen(input,prec, max_val):
            #generate the 8-bit bit representation of input tensor, add an extra dimension to the end, of size n (inlcuidng sign bits)
            #also need a sign bit, but in this code the sign bit is not count into the total number of bits
            input = quantization_n_formal(input,prec,max_val)
            MSB_val = max_val/2
            offset = math.log(MSB_val,2)
            #Then 8 shift is [maxval/2,maxval/4, ...]
            e = 0.0000001 # a very small number to covert the ouptut of torch.sign to desired form
            #sign = torch.clamp(torch.sign(input+e), min = 0)#take the sign
            sign = torch.sign(input)
            input = torch.abs(input)
            result = sign.unsqueeze(-1)#add one dimension to the tensors
            for i in range(prec):
                tmp = input - 2**(-i+offset) + e #will be positive if this bit is 1
                bit = torch.clamp(torch.sign(tmp), min = 0).unsqueeze(-1) #0/1 bit 
                input[tmp>0] = input[tmp>0] - 2**(-i+offset) #substract the value for those bit = 1
                result = torch.cat((result, bit),-1)#cat the bit result to result tensor
            return result #result should have a dimension of n contains a sign bit and n-1 actual bit


        def conv2d_lut_zdim_stride(input, weight, bias, max_val, act_prec, stride = 1):
            #version for z dimension weight sharing, modified from the code for xy dimension.
            #input should have 5 dimensions, the last dimension is the bit dimension, weight is still normal weight, doesn't need to be bit-serialized
            #inpud shape: [B,C,X,Y,bit], weight shape: [F,C,x,y]
            #stride supported, and group convolution is not currently supported
            #max-val is the maximum value (positive) that representable by the input, act_prec is the activation's bit width
            #generate the scale vector for summing the bits , -1 because sign bit is excluded
            group_size = 8 #number of channels in a group that stored in the lookup table
            #scale_vec = torch.zeros(act_prec-1)
            scale_vec =  torch.cuda.FloatTensor(act_prec).fill_(0)#(act_prec-1 for signed version)
            for i in range(act_prec):#use (act_prec-1 for signed version)
                if i == 0:
                    scale_vec[i] = 1
                else:
                    scale_vec[i] = scale_vec[i-1]*2
            scale_vec = torch.flip(scale_vec,[0])
            scale_vec = scale_vec/(torch.max(scale_vec)/(max_val/2))
            #print(scale_vec)
            #reshape the xy dimension into a single dimension, and then permute it to the last dimension
            #All bits, channels, filters and batches can be processed in parallel, and take the sum over
            #channels and bits and xy dimension accordingly
            #IN this scheme need to move the filter over image xy diemsnion, so that need this loop and corresponding index computation
            #need to think how to compute the indexes in this case (consider padding and need 9 indexes in this same time)
            input_shape = input.shape
            weight_shape = weight.shape
            k_size = weight.shape[2]
            k_offset = 1#offset of filter window 
            #define the output, size should be [B,F,X,Y] 
            #output = torch.zeros(input_shape[0], weig ht_shape[0], input_shape[2]-(k_size-k_offset), input_shape[3]-(k_size-k_offset))
            #output = torch.zeros(input_shape[0], weight_shape[0], int((input_shape[2]-(k_size-k_offset))/stride), int((input_shape[3]-(k_size-k_offset))/stride))
            output = torch.cuda.FloatTensor(input_shape[0], weight_shape[0], int((input_shape[2]-(k_size-k_offset))/stride), int((input_shape[3]-(k_size-k_offset))/stride)).fill_(0)
            #input = input.reshape(input_shape[0], input_shape[1], input_shape[2]*input_shape[3, input_shape[4]]).permute(0,1,3,2)
            #weight = weight.reshape(weight_shape[0], weight_shape[1], weight_shape[2]*weight_shape[3],weight_shape[4]).permute(0,1,3,2)
            sign = input[...,0].unsqueeze(-1)#extract the sign of input, shape is [B,C,X,Y,1]
            #sign bit in fact not needed here as layer will receive positive inputs, kept the sign impelmemtation for genality purpose. sign bit is considered as explicit, do not take into account the total bit-width
            input = input[...,1:] * sign.repeat(1,1,1,1,act_prec) #multiply the sign to the bit stream for later processing (so the addition results are correct)
            input = input.permute(0,4,1,2,3)
            #Now the input shape should be [B,bit,C,X,Y], weight shape should be [F,C,x,y]
            #Then rehape input and weight for the channel group
            input = input.reshape(input.shape[0],input.shape[1],int(input.shape[2]/group_size),group_size,input.shape[3],input.shape[4])
            weight = weight.reshape(weight.shape[0],int(weight.shape[1]/group_size),group_size,weight.shape[2],weight.shape[3])
            #Now input shape should be [B,bit,Cg,8,X,Y], weight shape should be [F,Cg,8,x,y]
            #permute both input and weights for later processing to put group into last dimension
            input = input.permute(0,2,4,5,1,3)
            weight = weight.permute(0,1,3,4,2)
            #now input shape is [B,Cg,X,Y,bit,8], weight shape is [F,Cg,x,y,8]
            #add one dimension after batch dimension to input for broadcasting operation
            #Also add a dimension to weight tensor after second dimension, for the bit operation
            input = input.unsqueeze(1)
            weight = weight.unsqueeze(4)
            #Now input shape should be [B,1,Cg,X,Y,bit,8], weight shape should be [F,Cg,x,y,1,8]
            #Then loop through input plane to perform convolution, need to generate the required index for the entire window
            #Can be implemented uisng slicing operation to get the desired input
            for win_x in range(0,input_shape[2]-(k_size-k_offset), stride):
                for win_y in range(0, input_shape[3]-(k_size-k_offset), stride):
                    #get the conv window (input patch), shape should be [B,1,C,bit,x,y]
                    input_patch = input[...,win_x:win_x+k_size,win_y:win_y+k_size,:,:]
                    raw_output = input_patch * weight #now the input bits are already signed, no need to worry about sign here
                    #raw output shape should be [B,F,Cg,x,y,bit,8]
                    #Then need to sum over channel group 
                    psum_z = torch.sum(raw_output, (-1))
                    #the result should have shape [B,F,Cg,x,y,bit]
                    #Now should apply the quantization to the reuslt corresponding to desired LUT precision
                    #print(torch.mean(torch.abs(psum_z)), torch.max(psum_z))
                    psum_z.data = quantization_n(psum_z.data,psumbw,maxval)
                    #Now should sum over bits, need to scale each bit with corresponding power of 2
                    psum_z = psum_z*scale_vec
                    psum_bit = torch.sum(psum_z, -1)
                    #shape should be [B,F,Cg,x,y]
                    #Then sum over channel and xy dimension
                    psum = torch.sum(psum_bit, (-1,-2,-3))
                    output[...,int(win_x/stride),int(win_y/stride)] = psum
            #print("------------------------------------------")
            #Bias computation
            if bias != None:  
                print(bias)          
                bias = bias.reshape(1,bias.shape[0],1,1)
                output = output + bias
            return output           


        def quantization_n(input, n = 8, rangeq = 1):
            #rangeq = 0.5
            intv = (rangeq*2)/(2**n-1)
            qunt = torch.round(torch.mul(input,(1/intv)))  #***use ceil instead of floor for small value of n to make sure that not all weights are modified to zero in the beginning, achieves good accuracy for small n
            #qunt = torch.floor(torch.mul(input,2**n-1))
            #the above line divide the whole tensor by the smallest interval (1/2**n-1), which is same as multiply with 2**n-1, then take the floor and finally multiply the whole tensor with the smallest interval
            out = torch.mul(qunt,intv)
            out = torch.clamp(out, min=-rangeq, max=rangeq) #make sure the quantized version lies in the interval 0-1, if it's bigger than one just clamp it at one
            return(out)

        def quantization_pos(input, n = 8, rangeq = 1):
            #rangeq = 0.5
            intv = (rangeq)/(2**n-1)
            qunt = torch.round(torch.mul(input,(1/intv)))  #***use ceil instead of floor for small value of n to make sure that not all weights are modified to zero in the beginning, achieves good accuracy for small n
            #qunt = torch.floor(torch.mul(input,2**n-1))
            #the above line divide the whole tensor by the smallest interval (1/2**n-1), which is same as multiply with 2**n-1, then take the floor and finally multiply the whole tensor with the smallest interval
            out = torch.mul(qunt,intv)
            out = torch.clamp(out, min=0, max=rangeq) #make sure the quantized version lies in the interval 0-1, if it's bigger than one just clamp it at one
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

        def select_kernel_channelwise(input, kernel_pool):
            #kernel_pool should have size of (N,9) for VGG network
            #input should have size (K,X,Y,C)/(K,3,3,C)
            orishape = input.shape
            input_reshaped = torch.reshape(input, (int(input.shape[0]*input.shape[1]*input.shape[2]*input.shape[3]/8), 8))
            output = find_optimal_cossim(input_reshaped, kernel_pool).reshape(orishape)
            return output

        def select_kernel(input, kernel_pool):
            #kernel_pool should have size of (N,9) for VGG network
            #input should have size (K,C,X,Y)/(K,C,3,3)
            input_reshaped = torch.reshape(input, (input.shape[0]*input.shape[1],input.shape[2]*input.shape[3]))
            output = find_optimal_cossim(input_reshaped, kernel_pool).reshape(input.shape[0],input.shape[1],input.shape[2],input.shape[3])
            return output

        def kernelpool_gen_random(kernel_size,pool_size):
            #kernelpool = torch.rand((pool_size,kernel_size))-torch.rand((pool_size,kernel_size))#make range -1 to 1
            kernelpool = torch.normal(0, 0.5, size=(pool_size,kernel_size))
            return kernelpool

        kernelpool = clustercenter.cuda()#load kernel pool using cluster centers from K means clustering
        coeff_groupsize = 1 #number of kernels to share a same coeff

        class Conv2d_q(_ConvNd):    
            def __init__(self, in_channels, out_channels, kernel_size, batch_size = 32,stride=1,
                        padding=0, dilation=1, groups=1,
                        bias=True, padding_mode='zeros', layer_cnt = 0):
                batch_size = 32
                kernel_size_ori = kernel_size
                kernel_size = _pair(kernel_size)
                stride = _pair(stride)
                padding = _pair(padding)
                dilation = _pair(dilation)
                super(Conv2d_q, self).__init__(
                    in_channels, out_channels, kernel_size, batch_size, stride, padding, dilation,
                    False, _pair(0), groups, bias, padding_mode, layer_cnt)
                self.coeff = nn.Parameter(torch.ones(int(in_channels*out_channels/coeff_groupsize)))
                #self.coeff = nn.Parameter(torch.ones(int(in_channels*out_channels*kernel_size_ori*kernel_size_ori/(8*coeff_groupsize))))
                #self.sharedkernelcoeff = nn.Parameter(torch.ones(in_channels))
                #self.layercoeff = nn.Parameter(torch.ones(1))
                

            def _conv_forward(self, input, weight, coeff):
                #input.data = quantization_n(input.data,4,2)
                #if(self.layer_idx != 0):
                #weight.data = select_kernel(weight.data,kernelpool)
                #weight = select_kernel(weight.data, model.filterpool_trainable)
                #weight = weight * coeff
                act_max = act_config[self.layer_cnt]
                #print(act_max,torch.min(input))

                #input.data = quantization_pos(input.data,act_prec,act_max)

                #for coefficients for each kernel
                weight = weight.permute(0,2,3,1)
                permuteshape = weight.shape
                #weight.data = select_kernel(weight.data,kernelpool_layer)
                weight.data = select_kernel_channelwise(weight.data,kernelpool)
                '''
                #coefficients
                weight = torch.reshape(weight, (int(weight.shape[0]*weight.shape[1]*weight.shape[2]*weight.shape[3]/8), 8))
                coeff_repeat = coeff.unsqueeze(1)
                coeff_repeat = coeff_repeat.repeat(1,8*coeff_groupsize)
                coeff_repeat = coeff_repeat.reshape(weight.shape)
                #print(coeff_repeat[-1])
                weight = weight * coeff_repeat
                weight = weight.reshape(permuteshape)
                '''
                weight = weight.permute(0,3,1,2)
                
                pad_width = math.floor(weight.shape[2]/2)
                input = F.pad(input, (pad_width,pad_width,pad_width,pad_width))
                input = bit_stream_gen(input, act_prec, act_max)
                result = conv2d_lut_zdim_stride(input, weight, self.bias, act_max, act_prec, self.stride[0])
                return result
                

            def forward(self, input):
                return self._conv_forward(input, self.weight, self.coeff)




        class CONV_tiny_quant(nn.Module):
            def __init__(self, uniform=False):
                super(CONV_tiny_quant, self).__init__()

                self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=4, bias=False)
                self.conv2 = Conv2d_q(32, 32, kernel_size=5, padding=2, bias=False,layer_cnt=0)
                self.conv3 = Conv2d_q(32, 64, kernel_size=5, padding=2, bias=False,layer_cnt=1)
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

                x = x.reshape(-1, 4*4*64)
                x = self.fc1(x)
                x = self.bn4(x)
                return x

        model = CONV_tiny_quant()
        model.cuda()

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


        def adjust_learning_rate(optimizer, epoch, lr_init, n, decay_val):
            """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
            lr = lr_init * (decay_val ** (epoch // n))
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


        PATH = os.path.join(weightfolder,'tinyconv_qd_zdim64.pth')
        state_dict=torch.load(PATH)
        model.load_state_dict(state_dict)

        starttime = time.time()
        inferenceacc = validate(val_loader, model, criterion)
        endtime = time.time()
        elapsedtime = endtime - starttime
        setup = [act_prec,psumbw,act_config,maxval]
        print("Activation bit width is ", act_prec, "activation configuration is ", act_config, ", max psum value is ", maxval, ", psum bitwidth is ", psumbw)
        print("inference accuracy is ", inferenceacc, " original accuacy is 82.18 for tinyconv zdim")
        print("inference time is: ", elapsedtime)
        #result_holder.append(inferenceacc)      
        if inferenceacc > temp_best:
          temp_best = inferenceacc
          best_setup = setup
    result_holder.append(temp_best)
    setup_holder.append(best_setup)
print("the weight pool accuracy for specified lookup table precision is ",result_holder)



