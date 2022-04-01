# Codes for Bit-serial Weight Pools
Paper title: **Bit-serial weight pools: compression and arbitrary precision execution of neural networks on resource constrained processors**.
Accepted in MLSys 2022. 
The experiment codes contain two parts, one is for runtime evaluation and the other is for neural network accuracy evaluation.


## Runtime evaluation
We have uploaded three videos showing the demonstration of running the runtime codes on real microcontrollers in the runtime folder.
### Hardware requirements
Two microcontrollers are used:
- STM32F207ZG
- STM32F103RB

### Software requirements
- Keil uVision 5.34 (may require license) or similar IDE (need to have program timer)
- STLink Driver (for connecting the microcontroller)
- STM32CubeMX (for generating driver libraries and initialization codes, free to use)
- Python 3.5+ (for input data and index data generation scripts)
- Tensorflow 2 (for input data generation, not required if using pre-generated data in this repo)

### Useful paths
- data generation scripts: Runtime/weight_pool_runtime
- generated test data: Runtime/weight_pool_runtime/TestData_fullnetwork
- generated index data: Runtime/weight_pool_runtime/index_data
- generated weight pool data: Runtime/weight_pool_runtime/lut_zdim64_data.h
- weight-pool based convolution source codes: Runtime/weight_pool_runtime/CMSIS/NN/Source/ConvolutionFunctions/lut_convolve_zdim.c
- Testbenches: /Runtime/weight_pool_runtime/CMSIS/NN/Tests/UnitTest/TestCases/benchmarks

###  Workflow and usage

#### Data generation
**All test data required (described below) to evaluate the networks reported in the paper are already provided. Data generation is not needed for verifying the results. So this step can be skipped if you want to use pre-generated data.**

The first step of the workflow is to generate input data for the testbenches. The scripts for data generation are located in Runtime/weight_pool_runtime. 

There are three types of data that need to be generated: neural network data (layer inputs, bias, network configuration, etc.), weight index data (for the proposed weight-pool networks, neural network weights are replaced with indices pointing to the weight pool array) and The actual weight pool data.

For the neural network data, the generating scripts are data_gen_generic.py and data_gen_mobilenet.py. Use data_gen_generic.py for all networks except for MobileNet and use data_gen_mobilenet.py for MobileNet. The network parameters are already defined in the scripts. When using the generic script, need to modify the 'networkname' and 'selected_network' parameters to the target neural network model. The neural network model parameters are already defined in the script (defined just before the 'networkname' parameter). The default path for generated neural network data is './TestData_fullnetwork/'. Note the neural network data generation scripts is derived from the scripts in ARM CMSIS library and require Tensorflow to execute. 

For the weight index data, the generating scripts are idx_gen_generic.py and idx_gen_mobilenet.py. Similarly, use idx_gen_generic.py for all networks except MobileNet and use idx_gen_mobilenet.py for MobileNet. When using the generic script, need to modify the 'networkname' and 'network' parameters to the target neural network model. The default path for generated index data is './index_data'. Tensorflow is not used for generating index data, only Numpy is used.

For the actual weight pool data, it can be a random C array with N entries, where N is 256 * WEIGHT_POOL_SIZE. The data can be random because it won't affect the runtime. 

If you want to test on another CNN apart from the ones reported in the paper or just a single layer, you can add the network configuration in idx_gen_generic.py and data_gen_generic.py as python lists. The format is provided in the scripts. Then modify the network name in the scripts and run the scripts to generate the data. 

To run the scripts, type `python xxx.py` in the terminal, where xxx.py is the script name you want to run. 

#### Setting up the benchmark
After generating test data, the next step is to set up corresponding testbenches for runtime evaluation. Testbench for all the networks reported in the paper are provided in '/Runtime/weight_pool_runtime/CMSIS/NN/Tests/UnitTest/TestCases/benchmarks'. The testbench files are used as the entry to the program (main() function). All generated data are included in the testbench as C headers. The microcontroller initialization codes (for STM32F207ZG) are also included in the testbenches. **For STM32F207ZG evaluation you don't need to modify anything unless you want to test on other networks.** You need to write a testbench yourself to test on other networks, and the format can refer to existing benchmarks.

For STM32F103RB evaluation, you need to replace the initialization codes (`SystemClock_Config()` function) with the one for F103RB. The initialization codes can be generated from STM32CubeMX. 

##### STM32CubeMX tutorial
STM32CubeMX is needed to generate initialization codes if you want to test on STM32F103RB. It can also add the corresponding driver library to Keil uVision's runtime environment so you don't need to manually import the driver library.

First open the program and start a new project. Then select the corresponding model in the MCU selecter. Then open the Clock configuration tab and set the HCLK to the maximum frequency of the microcontrollers (120 MHz for F207ZG and 72 MHz for F103RB) and press enter, the program will automatically find a solution to it. Then click the generate code button to generate the initialization code and required driver files. More detailed tutorials on STM32CubeMX can be found online, for example [this one](https://www.stmicroelectronics.com.cn/content/ccc/resource/training/technical/product_training/group0/76/8c/01/d7/28/d0/4c/e7/STM32G0-Ecosystem-STM32CubeMX-Tool/files/STM32G0-Ecosystem-STM32CubeMX-Tool.pdf/_jcr_content/translations/en.STM32G0-Ecosystem-STM32CubeMX-Tool.pdf).

#### Parameter configuration
Once the test network is determined and required test data is generated, the next step is to set the correct parameters for experiments. In the paper two types of experiment results are reported (Section 5.4.1 and 5.4.2), which are the impact of activation bitwidth and full network benchmark with different weight pool size and activation bitwidth. There are two parameters that can be adjusted, namely the activation bitwidth and weight pool size. Both parameters are defined in the weight pool convolution source code: 'Runtime/weight_pool_runtime/CMSIS/NN/Source/ConvolutionFunctions/lut_convolve_zdim.c'. The parameters are defined as C macros at the beginning of the code. *LUT_PREC* defines the activation bitwidth and *LUT_SIZE* defines the weight pool size. Make sure to set these two parameters to correct values before compiling the codes. 

#### Compliation
Before testing on microcontrollers, the codes need to be complied first. The instructions here are for Keil uVision IDE. First create a project and set the device to the correct microcontroller under STMicroelectronics. For run-time environment, select CMSIS-CORE, Device-Startup and Device-STM32Cube HAL-(Common, Cortex, GPIO, RCC). If you are not using STM32CubeMX then you need to manually add the driver files into the source group. Then add everything under '/Runtime/weight_pool_runtime/CMSIS/NN/Source' to the source group. The next step is to add the testbench to the source group and only one testbench should be added to the source group to avoid conflicts. 
Then open the 'Options' setting for the target (right click target), and open the C/C++(AC6) tab to set the optimization level to **O2**. Finally, build the project and the binary will be generated and ready to test on actual microcontrollers. 

#### Testing
The final step is to test the compiled code on the target microcontroller. First install the STLink Driver ([click here to download](https://www.st.com/en/development-tools/stsw-link009.html)) and connect the microcontroller to the host. 

Then open the Options setting again to configure the debugger and target. First open the target tab and set the Xtal to the maximum frequency of the target microcontrollers (120 MHz for F207ZG and 72 MHz for F103RB). Then open the Debug tab and on the top-right click use debugger and in the drop-down menu select ST-Link Debugger. Then click the Settings button on the right of the drop-down menu, and open the trace tab to set the Core Clock frequency to the maximum frequency (same as before). 

After finishing the configuration, the next step is to start the debug session. The program will be loaded into the microcontroller and start execution. The runtime is measured by the built-in timer in uVision's debugger and layer-wise runtime can be measured by setting breakpoints accordingly. 

### Expected results
The key results of runtime evaluation are shown in table 7 of the main paper. The results can be reproduced using the description above.

## Accuracy evaluation
### Hardware requirements
- Large-VRAM GPU preferred for faster training and inference

### Software requirements
- Pytorch 1.9.0
- Python 3.6.1 (Python 3.5+ should work)
- kmeans-pytorch (for k-means clustering of pretrained weights, to install: `pip install kmeans-pytorch`)

###  Useful paths
- Scripts for training uncompressed neural networks and generating weight pools: 'Accuracy\accuracy_codes\original_model_training'
- Scripts for training and evaluating weight pool networks: 'Accuracy\accuracy_codes\fw_training'

### Network and datasets
In the paper five neural networks and two datasets are evaluated. The five neural networks are Resnet-mlperf, Resnet-10, Resnet-14, MobileNet-v2 and TinyConv. Network definition can be found in the corresponding training scripts under  'Accuracy\accuracy_codes\original_model_training'. Two datasets are CIFAR-10 and QuickDraw-100. 
#### CIFAR-10
This dataset can be automatically downloaded when running the training scripts.
#### QuickDraw-100
We used the instructions in this [GitHub repo](https://github.com/XJay18/QuickDraw-pytorch) to download and generate the QuickDraw dataset for Pytorch. The scripts for automatically downloading and processing the dataset is provided.

### Workflow and usage
We have updated the workflow and instruction for easier reproduction of paper results. Please see the steps listed below to reproduce the results. **When executing the scripts, please stay in the root directory where the scripts are located in (.../Accuracy/accuracy_codes) to avoid potential problems during the execution.**
There are 4 automation scripts named "run_xxx.sh" to train the network and generate results for different tables. These 4 scripts share two positional command-line arguments:

- Network name: First positional argument, specifies the name of the network to evaluate. Use **"all"** to evaluate all the networks, or evaluate a single network by using its name. Network name are defined as: **(resnet_10,resnet_14,resnet_mlperf,mobilenet_v2,tinyconv)**. 
- Number of epochs: Second positional argument, specifies the number of epochs for training and/or retraining. Leave blank for default values. Note this argument is just for faster functional testing and debugging. This argument should be left blank for results reproduction. 

If both arguments are left blank, by default all networks will be evaluated/trained with the default number of epochs. 
Below are some usage examples:

Run all networks with the default number of epochs: `./run_xxx.sh all`

Run all networks with just one epoch: `./run_xxx.sh all 1`

Run resnet-10 with 10 epochs: `./run_xxx.sh resnet_10 10`


#### Step 1: Dataset preparation
CIFAR-10 dataset will be automatically downloaded and processed when executing the training codes. 
To download the prepare the Quick Draw dataset, simply run **qd_dataset_prepare.sh** inside the 'accuracy_codes' directory and the dataset will be downloaded and prepared. 

#### Step 2: Uncompressed network training and weight pool generation
The first step is to train uncompressed networks to generate the 'pre-trained' weights that will be further used to generate weight pools. After training the next step is to generate the weight pool for this network by applying K-means clustering on the trained weights. The uncompressed model training and weight pool generation are combined into a single script. By default three weight pools with sizes 32, 64 and 128.

Before running the scripts, you need to install Pytorch and *kmeans-pytorch* library (check software requirements).

Usage: `./run_pretraining.sh all` to train and generate cluster centers for all five networks. The weights and cluster centers will be automatically stored in separate folders. You can save training time by manually specifying the number of epochs (see instructions above). 

Results: This step generates the pretrained network accuracy, and is used in the paper table 4 first column. 

#### Step 3: Weight pool network training
Once the uncompressed weights and weight pools are generated, the next step is to compress the network using the proposed weight pool method. This step involves replacing the original weights with weight vectors in the weight pool and retraining the network to refine the weight vector assignment (see Figure 2 in the paper). The source codes for generating weight pool networks are under 'Accuracy\accuracy_codes\fw_training' and named as *network_dataset_wp_zdim_auto.py*. 

Usage: `./run_weightpooltraining.sh all` to train all five networks with weight pools and store the weights. You can save training time by manually specifying the number of epochs (see instructions above). Results will be printed out in the end. 

Results: This step generates the results used in the paper table 4 column 2,3,4. 

#### Step 4: Lookup table precision sweep
This step generates the results for weight pool networks with different lookup table precision. The source codes for sweeping lookup table bitwidths are under 'Accuracy\accuracy_codes\fw_training' and named as *network_dataset_prec_sweep_lut.py*.

Usage: `./run_weightpool_lutprec.sh all` to sweep the activation precision for all networks. No training or retraining is happening in this step. Results will be printed out in the end. 

Results: This step generates the results for the paper table 5. 

#### Step 5: Activation precision sweep
This step generates the results for weight pool networks with different activation precision. The source codes for sweeping activation bitwidths are under 'Accuracy\accuracy_codes\fw_training' and named as *network_dataset_prec_sweep.py*. For lower precision, retraining will be performed as stated in the paper. 

Usage: `./run_weightpool_actprec.sh all` to sweep the activation precision for all networks. You can save retraining time by manually specifying the number of epochs (see instructions above). Results will be printed out in the end. 

Results: This step generates the results for the paper table 6.


### Expected results
The expected results are listed in table 4, table 5 and table 6 in the paper. Table 4 shows the results for weight pool network accuracy for different weight pool sizes. Table 5 shows the accuracy for different lookup table bitwidths. Table 6 shows the accuracy for different activation table bitwidths. Use the three *run_weightpoolxxx.sh* scripts to generate the corresponding results.
