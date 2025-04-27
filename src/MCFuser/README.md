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