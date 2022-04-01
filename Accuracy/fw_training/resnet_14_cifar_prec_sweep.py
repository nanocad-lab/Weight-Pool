import torch
import os
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
import torch.utils.data
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


ccname = "resnet14_cifar_clustercenter_zdim64.npy"
cluster_path = os.path.join(ccfolder,ccname)
clustercenter = np.load(cluster_path)
clustercenter = torch.from_numpy(clustercenter)

print_freq = 1000

#list of activation max values, profiled by checking the actual activation maximum values, mid is assign to nearest power of 2 (has clipping effect), max is always ceil to the next power of 2 results
act_maxval_layer_arr_max = [16,8,16,8,16,8,0,16,8,16,8,0,16,8]
act_maxval_layer_arr_mid = [16,8,16,8,16,8,0,8,4,8,8,0,8,4]
act_maxval_layer_arr_8 = [8,8,8,8,8,8,0,8,8,8,8,0,8,8]
act_maxval_layer_arr_4 = [4,4,4,4,4,4,0,4,4,4,4,0,4,4]
#act_config_all = [act_maxval_layer_arr_max, act_maxval_layer_arr_mid, act_maxval_layer_arr_8, act_maxval_layer_arr_4]
act_config_all = [act_maxval_layer_arr_4]
result_holder = []

maxval = 1
for act_prec in [8,7,6,5,4,3]: 
  temp_best = 0
  for act_config in act_config_all:
    for bw in [8]:
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

      def bit_stream_gen(input,prec, max_val):
          #generate the 8-bit bit representation of input tensor, add an extra dimension to the end, of size n (inlcuidng sign bits)
          #also need a sign bit
          MSB_val = max_val/2
          offset = math.log(MSB_val,2)
          #Then 8 shift is [maxval/2,maxval/4, ...]
          e = 0.0000001 # a very small number to covert the ouptut of torch.sign to desired form
          #sign = torch.clamp(torch.sign(input+e), min = 0)#take the sign
          sign = torch.sign(input)
          input = torch.abs(input)
          result = sign.unsqueeze(-1)#add one dimension to the tensors
          for i in range(prec-1):
              tmp = input - 2**(-i+offset) + e #will be positive if this bit is 1
              bit = torch.clamp(torch.sign(tmp), min = 0).unsqueeze(-1) #0/1 bit 
              input[tmp>0] = input[tmp>0] - 2**(-i+offset) #substract the value for those bit = 1
              result = torch.cat((result, bit),-1)#cat the bit result to result tensor
          return result #result should have a dimension of n contains a sign bit and n-1 actual bit

      def conv2d_lut_xydim(input, weight, bias, max_val, act_prec, stride):
          #TODO: add bias support
          #input  should have 5 dimensions, the last dimension is the bit dimension, weight is still normal weight, doesn't need to be bit-serialized
          #inpud shape: [B,C,X,Y,bit], weight shape: [F,C,x,y]
          #stride is assumed to be 1, and group convolution is not currently supported, bias not supported
          #max-val is the maximum value (positive) that representable by the input, act_prec is the activation's bit width
          #generate the scale vector for summing the bits , -1 because sign bit is excluded
          #scale_vec = torch.zeros(act_prec-1)
          scale_vec =  torch.cuda.FloatTensor(act_prec-1).fill_(0)
          for i in range(act_prec-1):
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
          k_offset = math.floor(k_size/2)#offset of filter window 
          #define the output, size should be [B,F,X,Y] 
          #output = torch.zeros(input_shape[0], weight_shape[0], input_shape[2]-(k_size-k_offset), input_shape[3]-(k_size-k_offset))
          output = torch.cuda.FloatTensor(input_shape[0], weight_shape[0], int((input_shape[2]-(k_size-k_offset))/stride), int((input_shape[3]-(k_size-k_offset))/stride)).fill_(0)
          #input = input.reshape(input_shape[0], input_shape[1], input_shape[2]*input_shape[3, input_shape[4]]).permute(0,1,3,2)
          #weight = weight.reshape(weight_shape[0], weight_shape[1], weight_shape[2]*weight_shape[3],weight_shape[4]).permute(0,1,3,2)
          sign = input[...,0].unsqueeze(-1)#extract the sign of input, shappe is [B,C,X,Y,1]
          input = input[...,1:] * sign.repeat(1,1,1,1,act_prec-1) #multiply the sign to the bit stream for later processing (so the addition results are correct)
          input = input.permute(0,1,4,2,3)
          #Now the input shape should be [B,C,bit,X,Y], weight shape should be [F,C,x,y]
          #add one dimension after batch dimension to input for broadcasting operation
          #Also add a dimension to weight tensor after second dimension, for the bit operation
          input = input.unsqueeze(1)
          weight = weight.unsqueeze(2)
          #Now input shape should be [B,1,C,bit,X,Y], weight shape should be [F,C,1,x,y]
          #Then loop through input plane to perform convolution, need to generate the required index for the entire window
          #Can be implemented uisng slicing operation to get the desired input
          for win_x in range(0,input_shape[2]-(k_size-k_offset), stride):
              for win_y in range(0,input_shape[3]-(k_size-k_offset), stride):
                  #get the conv window (input patch), shape should be [B,1,C,bit,x,y]
                  input_patch = input[...,win_x:win_x+k_size,win_y:win_y+k_size]
                  raw_output = input_patch * weight #now the input bits are already signed, no need to worry about sign here
                  #rawoutput shape shoudl be [B,F,C,bit,x,y]
                  #Then need to sum over x and y dimension of kernel
                  psum_xy = torch.sum(raw_output, (-1,-2))
                  #the result should have shape [B,F,C,bit]
                  #Now should apply the quantization to the reuslt corresponding to desired LUT precision
                  #print(torch.mean(torch.abs(psum_xy)), torch.max(psum_xy))
                  psum_xy.data = quantization_n(psum_xy.data,bw,maxval)
                  #Now should sum over bits, need to scale each bit with corresponding power of 2
                  psum_xy = psum_xy*scale_vec
                  psum_bit = torch.sum(psum_xy, -1)
                  #shape should be [B,F,C]
                  #Then sum over channels
                  psum = torch.sum(psum_bit, -1)
                  output[...,int(win_x/stride),int(win_y/stride)] = psum

          #Bias computation
          if bias != None:            
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

      def quantization_pos_formal(input, n = 8, rangeq = 1):
          #updated version so that the minimal interval are integer power of two, so it fits better with stream gen function and is more formal way to do quantization
          maxpower = int(math.log(rangeq,2))
          minpower = maxpower-(n)
          max = rangeq-2**minpower
          intv = (rangeq-2**minpower)/(2**n-1)
          qunt = torch.round(torch.mul(input,(1/intv)))  #***use ceil instead of floor for small value of n to make sure that not all weights are modified to zero in the beginning, achieves good accuracy for small n
          #the above line divide the whole tensor by the smallest interval (1/2**n-1), which is same as multiply with 2**n-1, then take the floor and finally multiply the whole tensor with the smallest interval
          out = torch.mul(qunt,intv)
          out = torch.clamp(out, min=0, max=max) #make sure the quantized version lies in the interval 0-1, if it's bigger than one just clamp it at one
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

              input = quantization_pos_formal(input,act_prec,act_max)

              # for coefficients for each kernel
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
              
              # for coefficients for each kernel
              '''
              weight_shape = weight.shape
              weight = weight.reshape(int(weight.shape[0]*weight.shape[1]/coeff_groupsize),weight.shape[2]*weight.shape[3]*coeff_groupsize)
              coeff_repeat = coeff.unsqueeze(1)
              coeff_repeat = coeff_repeat.repeat(1,weight_shape[2]*weight_shape[3]*coeff_groupsize)
              weight = weight * coeff_repeat
              weight = weight.reshape(weight_shape)
              '''
              
              #print("------------------------------------------")
              if self.padding_mode != 'zeros':
                  return (F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                                  weight, self.bias, self.stride,
                                  _pair(0), self.dilation, self.groups))
              return (F.conv2d(input, weight, self.bias, self.stride,
                              self.padding, self.dilation, self.groups))

          def forward(self, input):
              return self._conv_forward(input, self.weight, self.coeff)




      __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
                'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
                'wide_resnet50_2', 'wide_resnet101_2']


      model_urls = {
          'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
          'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
          'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
          'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
          'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
          'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
          'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
          'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
          'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
      }


      def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, layer_cnt = 0) -> nn.Conv2d:
          """3x3 convolution with padding"""
          #return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=dilation, groups=groups, bias=False, dilation=dilation)
          return Conv2d_q(in_planes, out_planes, kernel_size=3, stride=stride,padding=dilation, groups=groups, bias=False, dilation=dilation, layer_cnt = layer_cnt)


      def conv1x1(in_planes: int, out_planes: int, stride: int = 1, layer_cnt = 0) -> nn.Conv2d:
          """1x1 convolution"""
          return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
          #return Conv2d_q(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


      class BasicBlock(nn.Module):
          #expansion: int = 1
          expansion = 1

          def __init__(
              self,
              inplanes: int,
              planes: int,
              stride: int = 1,
              downsample: Optional[nn.Module] = None,
              groups: int = 1,
              base_width: int = 64,
              dilation: int = 1,
              layer_cnt = 0,
              norm_layer: Optional[Callable[..., nn.Module]] = None
          ) -> None:
              super(BasicBlock, self).__init__()
              if norm_layer is None:
                  norm_layer = nn.BatchNorm2d
              if dilation > 1:
                  raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
              # Both self.conv1 and self.downsample layers downsample the input when stride != 1
              self.conv1 = conv3x3(inplanes, planes, stride, layer_cnt = layer_cnt)
              layer_cnt = layer_cnt + 1
              self.bn1 = norm_layer(planes)
              self.relu = nn.ReLU(inplace=True)
              self.conv2 = conv3x3(planes, planes, layer_cnt = layer_cnt)
              #layer_cnt = layer_cnt + 1
              self.bn2 = norm_layer(planes)
              self.downsample = downsample
              self.stride = stride

          def forward(self, x: Tensor) -> Tensor:
              identity = x

              out = self.conv1(x)
              out = self.bn1(out)
              out = self.relu(out)

              out = self.conv2(out)
              out = self.bn2(out)

              if self.downsample is not None:
                  identity = self.downsample(x)

              out += identity
              out = self.relu(out)

              return out


      class Bottleneck(nn.Module):
          # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
          # while original implementation places the stride at the first 1x1 convolution(self.conv1)
          # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
          # This variant is also known as ResNet V1.5 and improves accuracy according to
          # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

          #expansion: int = 4
          expansion = 4

          def __init__(
              self,
              inplanes: int,
              planes: int,
              stride: int = 1,
              downsample: Optional[nn.Module] = None,
              groups: int = 1,
              base_width: int = 64,
              dilation: int = 1,
              layer_cnt = 0,
              norm_layer: Optional[Callable[..., nn.Module]] = None
          ) -> None:
              super(Bottleneck, self).__init__()
              if norm_layer is None:
                  norm_layer = nn.BatchNorm2d
              #layer_cnt = layer_cnt + 2
              width = int(planes * (base_width / 64.)) * groups
              # Both self.conv2 and self.downsample layers downsample the input when stride != 1
              self.conv1 = conv1x1(inplanes, width, layer_cnt = layer_cnt)
              #layer_cnt = layer_cnt + 1
              self.bn1 = norm_layer(width)
              self.conv2 = conv3x3(width, width, stride, groups, dilation, layer_cnt = layer_cnt)
              #layer_cnt = layer_cnt + 1
              self.bn2 = norm_layer(width)
              self.conv3 = conv1x1(width, planes * self.expansion, layer_cnt = layer_cnt)
              #layer_cnt = layer_cnt + 1
              self.bn3 = norm_layer(planes * self.expansion)
              self.relu = nn.ReLU(inplace=True)
              self.downsample = downsample
              self.stride = stride

          def forward(self, x: Tensor) -> Tensor:
              identity = x

              out = self.conv1(x)
              out = self.bn1(out)
              out = self.relu(out)

              out = self.conv2(out)
              out = self.bn2(out)
              out = self.relu(out)

              out = self.conv3(out)
              out = self.bn3(out)

              if self.downsample is not None:
                  identity = self.downsample(x)

              out += identity
              out = self.relu(out)

              return out


      class ResNet(nn.Module):

          def __init__(
              self,
              block: Type[Union[BasicBlock, Bottleneck]],
              layers: List[int],
              num_classes: int = 10,
              zero_init_residual: bool = False,
              groups: int = 1,
              width_per_group: int = 16,#64,
              replace_stride_with_dilation: Optional[List[bool]] = None,
              norm_layer: Optional[Callable[..., nn.Module]] = None
          ) -> None:
              super(ResNet, self).__init__()
              if norm_layer is None:
                  norm_layer = nn.BatchNorm2d
              self._norm_layer = norm_layer
              layer_cnt = 0
              self.inplanes = 64
              self.dilation = 1
              if replace_stride_with_dilation is None:
                  # each element in the tuple indicates if we should replace
                  # the 2x2 stride with a dilated convolution instead
                  replace_stride_with_dilation = [False, False, False]
              if len(replace_stride_with_dilation) != 3:
                  raise ValueError("replace_stride_with_dilation should be None "
                                      "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
              self.groups = groups
              self.base_width = width_per_group
              self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                                      bias=False)
              self.bn1 = norm_layer(self.inplanes)
              self.relu = nn.ReLU(inplace=True)
              self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
              self.layer1 = self._make_layer(block, 64, layers[0], layer_cnt = layer_cnt)
              layer_cnt = layer_cnt + 4
              self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                              dilate=replace_stride_with_dilation[0], layer_cnt = layer_cnt)
              layer_cnt = layer_cnt + 5
              self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                              dilate=replace_stride_with_dilation[1], layer_cnt = layer_cnt)
              #layer_cnt = layer_cnt + 5
              #self.layer4 = self._make_layer(block, 512, layers[3], stride=2,dilate=replace_stride_with_dilation[2], layer_cnt = layer_cnt)
              self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
              self.fc = nn.Linear(256 * block.expansion, num_classes)

              for m in self.modules():
                  if isinstance(m, Conv2d_q):
                      nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                  elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                      nn.init.constant_(m.weight, 1)
                      nn.init.constant_(m.bias, 0)

              # Zero-initialize the last BN in each residual branch,
              # so that the residual branch starts with zeros, and each residual block behaves like an identity.
              # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
              if zero_init_residual:
                  for m in self.modules():
                      if isinstance(m, Bottleneck):
                          nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                      elif isinstance(m, BasicBlock):
                          nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

          def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                          stride: int = 1, dilate: bool = False, layer_cnt = 0) -> nn.Sequential:
              norm_layer = self._norm_layer
              downsample = None
              previous_dilation = self.dilation
              if dilate:
                  self.dilation *= stride
                  stride = 1
              if stride != 1 or self.inplanes != planes * block.expansion:
                  downsample = nn.Sequential(
                      conv1x1(self.inplanes, planes * block.expansion, stride),
                      norm_layer(planes * block.expansion),
                  )

              layers = []
              layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                  self.base_width, previous_dilation, layer_cnt, norm_layer))
              self.inplanes = planes * block.expansion
              if stride != 1 or self.inplanes != planes * block.expansion:
                  layer_cnt = layer_cnt + 3
              else:
                  layer_cnt = layer_cnt + 2
              for _ in range(1, blocks):
                  layers.append(block(self.inplanes, planes, groups=self.groups,
                                      base_width=self.base_width, dilation=self.dilation,layer_cnt = layer_cnt,
                                      norm_layer=norm_layer))
                  layer_cnt = layer_cnt+2

              return nn.Sequential(*layers)

          def _forward_impl(self, x: Tensor) -> Tensor:
              # See note [TorchScript super()]
              x = self.conv1(x)
              x = self.bn1(x)
              x = self.relu(x)
              x = self.maxpool(x)

              x = self.layer1(x)
              x = self.layer2(x)
              x = self.layer3(x)
              #x = self.layer4(x)

              x = self.avgpool(x)
              x = torch.flatten(x, 1)
              x = self.fc(x)

              return x

          def forward(self, x: Tensor) -> Tensor:
              return self._forward_impl(x)


      def _resnet(
          arch: str,
          block: Type[Union[BasicBlock, Bottleneck]],
          layers: List[int],
          pretrained: bool,
          progress: bool,
          **kwargs: Any
      ) -> ResNet:
          model = ResNet(block, layers, **kwargs)
          if pretrained:
              #state_dict = load_state_dict_from_url(model_urls[arch],progress=progress)
              state_dict = utils.model_zoo.load_url(model_urls[arch],progress=progress)
              model.load_state_dict(state_dict)
          return model

      def resnet_cifar10(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
          r"""ResNet-18 model from
          `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
          Args:
              pretrained (bool): If True, returns a model pre-trained on ImageNet
              progress (bool): If True, displays a progress bar of the download to stderr
          """
          return _resnet('resnet18', BasicBlock, [2, 2, 2], pretrained, progress,
                      **kwargs)

      def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
          r"""ResNet-18 model from
          `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
          Args:
              pretrained (bool): If True, returns a model pre-trained on ImageNet
              progress (bool): If True, displays a progress bar of the download to stderr
          """
          return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                        **kwargs)


      def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
          r"""ResNet-34 model from
          `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
          Args:
              pretrained (bool): If True, returns a model pre-trained on ImageNet
              progress (bool): If True, displays a progress bar of the download to stderr
          """
          return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                        **kwargs)


      def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
          r"""ResNet-50 model from
          `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
          Args:
              pretrained (bool): If True, returns a model pre-trained on ImageNet
              progress (bool): If True, displays a progress bar of the download to stderr
          """
          return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                        **kwargs)


      def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
          r"""ResNet-101 model from
          `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
          Args:
              pretrained (bool): If True, returns a model pre-trained on ImageNet
              progress (bool): If True, displays a progress bar of the download to stderr
          """
          return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                        **kwargs)


      def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
          r"""ResNet-152 model from
          `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
          Args:
              pretrained (bool): If True, returns a model pre-trained on ImageNet
              progress (bool): If True, displays a progress bar of the download to stderr
          """
          return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                        **kwargs)


      def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
          r"""ResNeXt-50 32x4d model from
          `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
          Args:
              pretrained (bool): If True, returns a model pre-trained on ImageNet
              progress (bool): If True, displays a progress bar of the download to stderr
          """
          kwargs['groups'] = 32
          kwargs['width_per_group'] = 4
          return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                        pretrained, progress, **kwargs)


      def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
          r"""ResNeXt-101 32x8d model from
          `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
          Args:
              pretrained (bool): If True, returns a model pre-trained on ImageNet
              progress (bool): If True, displays a progress bar of the download to stderr
          """
          kwargs['groups'] = 32
          kwargs['width_per_group'] = 8
          return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                        pretrained, progress, **kwargs)


      def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
          r"""Wide ResNet-50-2 model from
          `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
          The model is the same as ResNet except for the bottleneck number of channels
          which is twice larger in every block. The number of channels in outer 1x1
          convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
          channels, and in Wide ResNet-50-2 has 2048-1024-2048.
          Args:
              pretrained (bool): If True, returns a model pre-trained on ImageNet
              progress (bool): If True, displays a progress bar of the download to stderr
          """
          kwargs['width_per_group'] = 64 * 2
          return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                        pretrained, progress, **kwargs)


      def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
          r"""Wide ResNet-101-2 model from
          `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
          The model is the same as ResNet except for the bottleneck number of channels
          which is twice larger in every block. The number of channels in outer 1x1
          convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
          channels, and in Wide ResNet-50-2 has 2048-1024-2048.
          Args:
              pretrained (bool): If True, returns a model pre-trained on ImageNet
              progress (bool): If True, displays a progress bar of the download to stderr
          """
          kwargs['width_per_group'] = 64 * 2
          return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                        pretrained, progress, **kwargs)


      model = resnet_cifar10(pretrained = False).cuda()
      #model = resnet50(pretrained = True).cuda()

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
              datatime = time.time()
              #print("data loading time is ", datatime - end)
              #print('new batch')
              input = input.cuda()
              target = target.cuda()
              #the transfer time to gpu is not slow
              starttime = time.time()
              # compute output
              with torch.no_grad():
                  output = model(input)
                  loss = criterion(output, target)
              endtime = time.time()
              #print("batch compute time is ", endtime - starttime)
              output = output.float()
              loss = loss.float()

              # measure accuracy and record loss
              prec1 = accuracy(output.data, target)[0]
              losses.update(loss.item(), input.size(0))
              top1.update(prec1.item(), input.size(0))

              # measure elapsed time
              batch_time.update(time.time() - end)
              #print("batch total time is ",time.time() - end)
              #print("-------------------------------------")
              end = time.time()
              
              if i % print_freq == 0:
                  print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))
              
          
          #print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
          
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
      criterion = nn.CrossEntropyLoss(size_average=True).cuda()


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

      batch_size = 128 # use a smaller batch size (32/64) when training on GTX 2080
      workers = 28

      trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=train_transform, target_transform=None, download=True)

      train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,drop_last=True,
                                                shuffle=True, pin_memory=False, num_workers=workers)

      testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=test_transform, target_transform=None, download=True)
      val_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,drop_last=True,
                                              shuffle=False, pin_memory=False, num_workers=workers)

      PATH = os.path.join(weightfolder,'resnet14_cifar_zdim64.pth')
      state_dict=torch.load(PATH)
      model.load_state_dict(state_dict)

      init_lr = 0.001
      decay_val = 0.2 #how much weight decay is applied when activated
      #optimizer = torch.optim.Adam(model.parameters(), lr =  init_lr)
      optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=5e-4)
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
      starttime = time.time()
      inferenceacc = validate(val_loader, model, criterion)
      print(inferenceacc)
      best_prec1 = inferenceacc
      if act_prec in [4,3]:
        print("Retraining starts!")
        weight_name = "resnet14_cifar_zdim64_retrained_"+str(act_prec)+"bit.pth"
        SAVEPATH = os.path.join(weightfolder,weight_name)     

        for epoch in range(n_epoch):
            #adjust_learning_rate(optimizer, epoch, init_lr, 10)
            #lr_schedule(optimizer, epoch)

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch)

            # evaluate on validation set
            prec1 = validate(val_loader, model, criterion)
            #result_list.append(prec1)

            # remember best prec@1 and save checkpoint
            if (prec1 > best_prec1):
                #Save the weight for the best accuracy 
                torch.save(model.state_dict(), SAVEPATH)
                print("best accuracy achieved, weight saved, accuracy is ", prec1)
            best_prec1 = max(prec1, best_prec1)
            scheduler.step()

      endtime = time.time()
      elapsedtime = endtime - starttime
      print("Activation bit width is ", act_prec,", max psum value is ", maxval, ", psum bitwidth is ", bw)
      print("inference accuracy is ", inferenceacc, " original accuacy is 91.13 for resnet 14 zdim")
      print("inference time is: ", elapsedtime)
      
    if best_prec1 > temp_best:
      temp_best = best_prec1
  result_holder.append(temp_best)
print("the weight pool accuracy for specified bitwidth is ",result_holder)



