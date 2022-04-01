# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 16:40:46 2021

@author: shuru
"""

import os
import math
import numpy as np

def write_c_common_header(f):
    #f.write("{}\n\n".format(LICENSE))
    #f.write(self.tensor_flow_reference_version)
    f.write("#pragma once\n")

def write_c_header_offsets(f, prefix):
    f.write("#define {}_INPUT_OFFSET {}\n".format(prefix, input_zero_point))
    f.write("#define {}_OUTPUT_OFFSET {}\n".format(prefix, output_zero_point))

def write_c_config_header(write_common_parameters=True):
    filename = self.config_data

    self.generated_header_files.append(filename)
    filepath = self.headers_dir + filename

    prefix = self.testdataset.upper()

    print("Writing C header with config data {}...".format(filepath))
    with open(filepath, "w+") as f:
        self.write_c_common_header(f)
        if (write_common_parameters):
            f.write("#define {}_OUT_CH {}\n".format(prefix, self.output_ch))
            f.write("#define {}_IN_CH {}\n".format(prefix, self.input_ch))
            f.write("#define {}_INPUT_W {}\n".format(prefix, self.x_input))
            f.write("#define {}_INPUT_H {}\n".format(prefix, self.y_input))
            f.write("#define {}_DST_SIZE {}\n".format(prefix, self.x_output * self.y_output * self.output_ch
                                                      * self.batches))
            f.write("#define {}_INPUT_SIZE {}\n".format(prefix, self.x_input * self.y_input * self.input_ch))
            if self.relu6:
                f.write("#define {}_OUT_ACTIVATION_MIN {}\n".format(prefix, 0))
                f.write("#define {}_OUT_ACTIVATION_MAX {}\n".format(prefix, 6))
            else:
                f.write("#define {}_OUT_ACTIVATION_MIN {}\n".format(prefix, self.out_activation_min))
                f.write("#define {}_OUT_ACTIVATION_MAX {}\n".format(prefix, self.out_activation_max))
            f.write("#define {}_INPUT_BATCHES {}\n".format(prefix, self.batches))
    #self.format_output_file(filepath)

def generate_c_array(name, array, datatype="q7_t", const="const "):
    if not os.path.exists(headers_dir):
        os.makedirs(headers_dir)

    w = None
    if type(array) is list:
        w = array
        size = len(array)
    else:
        w = array
        w = w.ravel()
        #size = tf.size(array)
        size = np.size(array)
    filename = name + "_data.h"
    filepath = headers_dir + filename

    generated_header_files.append(filename)

    print("Generating C header {}...".format(filepath))
    print(filepath)
    with open(filepath, "w+") as f:
        write_c_common_header(f)
        f.write("#include <stdint.h>\n\n")
        f.write(const + datatype + " " + testdataset + '_' + name + "[%d] =\n{\n" % size)
        for i in range(size - 1):
            f.write("  %d,\n" % w[i])
        f.write("  %d\n" % w[size - 1])
        f.write("};\n")
    #self.format_output_file(filepath)
    
def network_idx_gen(name, idx_list, datatype="uint_8", const="const "):
    if not os.path.exists(headers_dir):
        os.makedirs(headers_dir)

    layer_cnt = 1 #set initial layer_cnt to 1 so that the first layer will be 2, since actual first layer of the network cannot be implemented using z-dimension pooling
    filename = name + "_data.h"
    filepath = headers_dir + filename

    generated_header_files.append(filename)

    print("Generating C header {}...".format(filepath))
    print(filepath)
    with open(filepath, "w+") as f:
        write_c_common_header(f)
        f.write("#include <stdint.h>\n\n")
        for array in idx_list:
            layer_cnt = layer_cnt + 1
            w = None
            if type(array) is list:
                w = array
                size = len(array)
            else:
                w = array
                w = w.ravel()
                #size = tf.size(array)
                size = np.size(array)
                
            f.write(const + datatype + " " + name + '_' + "layer_" + str(layer_cnt) + "[%d] =\n{\n" % size)
            for i in range(size - 1):
                f.write("  %d,\n" % w[i])
            f.write("  %d\n" % w[size - 1])
            f.write("};\n")
    #self.format_output_file(filepath)
    
def array_gen(len, max):
    return np.random.randint(max, size = len)

def array_gen_sim(len, max):
    result = np.zeros((len))
    #implemnt something here
    return result

resnet_10 = [[32,3,3,64],[16,3,64,64],[16,3,64,64],[16,3,64,64],[16,3,64,64],[8,3,64,128],[8,3,128,128],[8,3,128,128],[8,3,128,128]]
resnet_14 = [[32,3,3,64],[16,3,64,64],[16,3,64,64],[16,3,64,64],[16,3,64,64],[8,3,64,128],[8,3,128,128],[8,3,128,128],[8,3,128,128],
            [4,3,128,256],[4,3,256,256],[4,3,256,256],[4,3,256,256]]
resnet_mlperf = [[32,3,3,16],[16,3,16,16],[16,3,16,16],[16,3,16,16],[16,3,16,16],[8,3,16,32],[8,3,32,32],[8,3,32,32],[8,3,32,32],
            [4,3,32,64],[4,3,64,64],[4,3,64,64],[4,3,64,64]]
tiny_conv = [[32,5,3,32],[16,5,32,32],[8,5,32,64]]
mobilenet_v2 = [ ['c', 32, 3, 3] ,
                        ['c', 32, 32, 1] ,
                        ['d', 32, 32, 3] ,
                        ['c', 16, 32, 1] ,
                        ['c', 16, 32, 1] ,
                        ['c', 96, 16, 1] ,
                        ['d', 96, 96, 3] ,
                        ['c', 24, 96, 1] ,
                        ['c', 24, 16, 1] ,
                        ['c', 144, 24, 1] ,
                        ['d', 144, 144, 3] ,
                        ['c', 24, 144, 1] ,
                        ['c', 144, 24, 1] ,
                        ['d', 144, 144, 3] ,
                        ['c', 32, 144, 1] ,
                        ['c', 32, 24, 1] ,
                        ['c', 192, 32, 1] ,
                        ['d', 192, 192, 3] ,
                        ['c', 32, 192, 1] ,
                        ['c', 192, 32, 1] ,
                        ['d', 192, 192, 3] ,
                        ['c', 32, 192, 1] ,
                        ['c', 192, 32, 1] ,
                        ['d', 192, 192, 3] ,
                        ['c', 64, 192, 1] ,
                        ['c', 384, 64, 1] ,
                        ['d', 384, 384, 3] ,
                        ['c', 64, 384, 1] ,
                        ['c', 384, 64, 1] ,
                        ['d', 384, 384, 3] ,
                        ['c', 64, 384, 1] ,
                        ['c', 384, 64, 1] ,
                        ['d', 384, 384, 3] ,
                        ['c', 64, 384, 1] ,
                        ['c', 384, 64, 1] ,
                        ['d', 384, 384, 3] ,
                        ['c', 96, 384, 1] ,
                        ['c', 96, 64, 1] ,
                        ['c', 576, 96, 1] ,
                        ['d', 576, 576, 3] ,
                        ['c', 96, 576, 1] ,
                        ['c', 576, 96, 1] ,
                        ['d', 576, 576, 3] ,
                        ['c', 96, 576, 1] ,
                        ['c', 576, 96, 1] ,
                        ['d', 576, 576, 3] ,
                        ['c', 160, 576, 1] ,
                        ['c', 960, 160, 1] ,
                        ['d', 960, 960, 3] ,
                        ['c', 160, 960, 1] ,
                        ['c', 960, 160, 1] ,
                        ['d', 960, 960, 3] ,
                        ['c', 160, 960, 1] ,
                        ['c', 960, 160, 1] ,
                        ['d', 960, 960, 3] ,
                        ['c', 320, 960, 1] ,
                        ['c', 320, 160, 1] ,
                        ['c', 1280, 320, 1] ]

network_name = 'mobilenet_v2'
network = mobilenet_v2
filtersize = 1 #adjust for different networks

generated_header_files = []
headers_dir = './index_data/' + network_name + '_data/'
input_zero_point = 0
output_zero_point = 0
testdataset = 'full_network_index_data'
lut_size = 64
fw_group_size = 8 #how many weights are grouped together

network_info = []
for layer in network:
    channel_in = layer[2]
    channel_out = layer[1]
    if (channel_in > fw_group_size):
        channel_group = int(channel_in/fw_group_size)
        total_group = channel_group * channel_out * filtersize
        idx_data = array_gen(total_group, lut_size)
        print(idx_data.shape)
        network_info.append(idx_data)
    

'''
layer_1 = array_gen(3*16,lut_size)
layer_2 = array_gen(16*32,lut_size)
layer_3 = array_gen(32*32,lut_size)
layer_4 = array_gen(32*32,lut_size)
network_info = [layer_1, layer_2, layer_3, layer_4]
'''



#generate_c_array('index', test_array, datatype='uint8_t')
network_idx_gen((network_name + '_index'), network_info, datatype='uint8_t')