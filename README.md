# STOF

This folder contains the system prototype of STOF (pap899) at SC '25, titled "STOF: Optimizing Sparse Transformer via Flexible Masking and Operator Fusion on GPU", including the scripts for reproducing and plotting the results in the paper.

## Abstract

The repository is organized as below:

+ `data/`: orignal data logs in `data/MHA_performance` for Figure 10-11, `data/End2End_performance` for Figure 12, `data/Ablation_Studdy` for Figure 13, `data/Overhead_Analysis` for Figure 14, `data/Tuning_Cost` for Table 4.

+ `plot/`: quick poltting reproduction code to get the figures in the paper, including `fig3/`, `fig4/`, `fig10-11/`, `fig12/`, `fig13/`, and `fig14/`. 

+ `scripts/`: shell scripts to completely reproduce the experimental results in the paper, including `env_install.sh`, `fig10-11.sh`, `fig12.sh`, `fig13.sh`, `fig14.sh` and `table4.sh`. 

+ `src/`: The core source code implemented in STOF. PyTorch Native/Compile, ByteTransformer, FlashAttention2, and FlexAttention are in the same environment as STOF and can be executed directly. MCFuser and Bolt need to be executed separately due to their specific compilation environment.

## Result Reproduction

We recommend using the image `nvcr.io/nvidia/pytorch:24.09-py3` to directly obtain the container with the basic environment.
```shell
# pull docker images and enter container
docker pull nvcr.io/nvidia/pytorch:24.09-py3
docker run --gpus all --name AD-pap899-SC25 --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it nvcr.io/nvidia/pytorch:24.09-py3 /bin/bash

# update PyTorch version
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# clone the repository and enter the directory
git clone https://github.com/AD-pap899-SC25/SC25-STOF-AD.git
cd SC25-STOF-AD

# enter scripts directory 
cd scripts
# install operators and check the environment
# according to running device input sm_{CUDAARCH}, e.g.,  A100:sm_80 4090:sm_89, 
# so that for A100: bash env_install.sh 80, and for 4090: bash env_install.sh 89
bash env_install.sh 80

# for Figure 10-11
bash fig10-11.sh

# for Figure 12
bash fig12.sh

# for Figure 13
bash fig13.sh

# for Figure 14
bash fig14.sh

# for Table 4 (STOF)
bash table4_STOF.sh
```

## Other Results

For the comparison of MCFuser and Bolt, we have uploaded the relevant necessary configuration files to [Google Drive](https://drive.google.com/file/d/17N-PfI0klMa1jHE-1YcpV5oNzjfcFxE4/view?usp=sharing). The installation and reproduction steps are as follows:
```shell
cd SC25-STOF-AD/src

# download ae-mcfuser-test.tar.gz from Google Drive
# uncompress the package
tar -xzvf ae-mcfuser-test.tar.gz

# rename this directory
mv ae-mcfuser-test3 ./MCFuser/mcfuser

cd ../scripts

# install MCFuser and Bolt
bash MCFuser_install.sh

# for Table 4 (MCFuser and Bolt)
bash table4.sh
```