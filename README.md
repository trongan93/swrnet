# SWRNET
This repository contains the full source code of the paper:

Trong-An Bui and Pei-Jun Lee, 
"**SWRNet: A Deep Learning Approach for Small Surface Water Area Recognition Onboard Satellite**," 
in _IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing,_ vol. 16, pp. 10369-10380, 2023, doi: 10.1109/JSTARS.2023.3328118.

**Abstract:** 
This article proposes a deep learning approach for small surface water recognition using multispectral satellite imaging, which reduces the computational complexity by 18.66 times and increases the accuracy of surface water recognition by up to 14.1%. The proposed model uses near infrared combined with RGB spectral imagery to increase the accuracy of surface water recognition. In addition, since surface water only accounts for a small percentage of the remote sensing dataset, thus creating an imbalance problem, a proposed loss function is introduced to combine region-based and distribution-based loss. This article introduces an adaptive factor that automatically adjusts the weighting between distribution- and region-based loss functions. The proposed adaptive factor is determined based on the loss value of the previous training step. The mean intersection over union of surface water between predicted and ground truth regions is recorded as 0.80
URL: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10298631&isnumber=9973430

The main training file of the SWRNet Model is **flooding_transfer_learning-rgb_ir.ipynb**.

Please reference this paper in your manuscript when you use this source code.

## Training Data
### Prerequisites

This code has been tested with:
- **CUDA 12.4**
- **Python 3.8**
- **Ubuntu 22.04**
- An **Anaconda** environment

Dataset (Link)

### Step 1: Install the Source Code
Clone the repository from GitHub:
```bash
git clone https://github.com/trongan93/swrnet.git
```
### Step 2: Create the Conda Environment
This source code uses a `conda` environment for managing dependencies.

Create the environment
   ```bash
   conda create -n swrnetvenv python=3.8
```
Activate environment

```bash
conda activate swrnetvenv
```
### Step 3: Set up CUDA
- To ensure compatibility with your hardware and PyTorch version, set up CUDA by visiting the [PyTorch installation guide](https://pytorch.org/get-started/locally).
- Use the following command to install PyTorch and CUDA (adjust the version if necessary):
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

### Step 4: Install the Modified **ml4floods** Library

To fit the requirements of this project, the **ml4floods** library has been modified.

#### References:
- Official Documentation: [Source](https://spaceml-org.github.io/ml4floods/content/config.html)
- Modified Repository: [Source Edit](https://github.com/trongan93/ml4floods)

#### Installation Steps:
Clone the modified **ml4floods** repository into the same directory as the cloned `swrnet` repo:
   ```bash
   git clone https://github.com/trongan93/ml4floods.git
```
Install
```bash
cd ml4floods
pip install -e .
```

### Step 5: Install Required Libraries

```bash
pip install albumentations==1.3.0 kornia==0.6.8 ptflops==0.6.9 pytorch-lightning==1.8.6
```

### Step 6: Training
The main training file of the SWRNet Model is `flooding_transfer_learning-rgb_ir.ipynb`.

- You can adjust the number of GPUs
`os.environ["CUDA_VISIBLE_DEVICES"]="0,1"` with 2 GPUs
- You can adjust the number of `epochs`, `batch size`, `number works` accordingly.
- If you don't use `wandb`, set it to **False** `setup_weights_and_biases = False`

After training, in the "`train_models/training_flooding_bgri`" folder there are 2 files "`.json`" and "`.pt`"

## Run inference

This guide will walk you through running inference on RGBIR data using the `Inference_on_rgbir_data_2024.ipynb` file. You can use the [pre-trained model](https://drive.google.com/file/d/1O8Zo2qRitUUC-RsJmZspPJraYf0FMSsE/view?usp=sharing).
- Locate the section in the notebook where the `config_fp` and `path_to_models ` are defined.
- Set these paths to the `.pt` and `.json` files in the `train_models/training_flooding_bgri` folder
