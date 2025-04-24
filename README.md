# STOF

This folder contains the system prototype of STOF (pap899) at SC '25, titled "STOF: Optimizing Sparse Transformer via Flexible Masking and Operator Fusion on GPU". including Figure 10-11, Figure 12, Figure 13 and Figure 14.

## Abstract

The repository is organized as below:

+ `data/`: orignal data log in `data/MHA_performance` for Figure 10-11, `data/End2End_performance` for Figure 12, `data/Ablation_Studdy` for Figure 13, `data/Overhead_Analysis` for Figure 14. `data/Tuning_Cost` for Table 4.

+ `plot/`: quick poltting reproduction code to get the images in the paper, including `fig3/`, `fig4/`, `fig10-11/`, `fig12/`, `fig13/`, and `fig14/`. 

+ `script/`: `.sh` executable script to install the custom operator in STOF and execute it in full to reproduce the experimental results in the paper. Including `env_install`, `fig10-11.sh`, `fig12.sh`, `fig13.sh`, and `fig14.sh`. 

+ `src/`: The core source code implemented in STOF, especially the unified MHA kernels, is in `src/ops/src/***.cu` bound by `src/setup.py`. The baselines that can be run directly include PyTorch Native, PyTorch Compiled, ByteTransformer, FlashAttention2, and FlexAttention. MCFuser and Bolt need to be executed by switching the conda environment, wihich will be introduced later.

## Getting Started

We recommend using the image `nvcr.io/nvidia/pytorch:24.09-py3` to directly obtain the container containing the basic environment such as PyTorch Compile.

```shell
# pull docker images and enter conatiner
docker pull nvcr.io/nvidia/pytorch:24.09-py3
docker run --gpus all --name AD-pap899-SC25 -itd nvcr.io/nvidia/pytorch:24.09-py3 /bin/bash

# clone the repository and encter the directory
git clone https://github.com/AD-pap899-SC25/SC25-STOF-AD.git
cd SC25-STOF-AD

# enter script directory 
cd script
# install operators and check the environment
# according to running device input sm_{CUDAARCH}, e.g.,  A100:sm_80 4090:sm_89, 
# so that for A100: bash env_install.sh 80, and for 4090: bash env_install.sh 89
bash env_install.sh 80

# for Figure10-11
bash fig10-11.sh

# for Figure12
bash fig12.sh

# for Figure13
bash fig13.sh

# for Figure14
bash fig14.sh

# for STOF in Table 4
bash table4_STOF.sh
```

### Comparisons that need to be run separately in the Artifact

For the comparison of Blselines MCFuser and Bolt, a lot of compilation and installation processes related to tvm and CUTLASS are involved. In order to reproduce this part of the experiment smoothly, we have uploaded the relevant necessary configuration files to [Google Drive](https://drive.google.com/file/d/17N-PfI0klMa1jHE-1YcpV5oNzjfcFxE4/view?usp=sharing). After downloading them, you need to execute the relevant installation script `script/MCFuser_install.sh`. The exact steps are as follows:

```shell
cd SC25-STOF-AD/src

# download ae-mcfuser-test.tar.gz from Google Drive
# compressed package this file 
tar -xzvf ae-mcfuser-test.tar.gz

# rename this directory
mv ae-mcfuser-test3 ./MCFuser/mcfuser

cd ../script

# install MCFuser and Bolt
bash MCFuser_install.sh

# for MCFuser and Bolt in Table 4
bash table4.sh
```