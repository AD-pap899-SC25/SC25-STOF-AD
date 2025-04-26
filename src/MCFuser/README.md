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