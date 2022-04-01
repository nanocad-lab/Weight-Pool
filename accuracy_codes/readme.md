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
The first step is to train uncompressed networks to generate the 'pre-trained' weights that will be further used to generate weight pools. After training the next step is to generate the weight pool for this network by applying K-means clustering on the trained weights. The uncompressed model training and weight pool generation are combined into a single script. By default three weight pools with sizes 32, 64 and 128. \=]'

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
