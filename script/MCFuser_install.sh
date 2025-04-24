

cd ../src/MCFuser

cp -r .triton ~/
tar -xvf triton.tar.gz
git config --global --add safe.directory /mcfuser/src/triton
git config --global --add safe.directory /mcfuser/src/triton/third_party/amd
pip install --force-reinstall pybind11==2.11.1
apt update
apt-get install libgtest-dev -y  
cd triton/python 

cd ..
git submodule update --init --recursive 
cd python
python setup.py install
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages/triton-2.1.0-py3.10-linux-x86_64.egg:$PYTHONPATH"

# install tvm-mcfuser
cd ../../
tar -xzvf tvm-mcfuser.tar.gz
cd tvm-mcfuser/python 
pip install Cython==3.0.0a10
python setup.py install


cd ../../
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda-repo-ubuntu2004-11-2-local_11.2.0-460.27.04-1_amd64.deb
dpkg -i cuda-repo-ubuntu2004-11-2-local_11.2.0-460.27.04-1_amd64.deb
apt-key add /var/cuda-repo-ubuntu2004-11-2-local/7fa2af80.pub
apt update
apt install cuda-nvrtc-11-2 -y
export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH


wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
apt update && apt install -y software-properties-common
add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
apt-get update


apt-get install cuda-compat-11-0 -y
apt install -y cuda-toolkit-11-0
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH


apt install llvm-13 llvm-13-dev libllvm13 -y
export LD_LIBRARY_PATH=/usr/lib/llvm-13/lib:$LD_LIBRARY_PATH


dpkg -i libcudnn8_8.0.5.39-1+cuda11.0_amd64.deb


apt-get install python3-tinydb  -y


pip install transformers==4.16.0
 

cp -r /usr/local/lib/python3.10/dist-packages/tvm-0.13.dev0-py3.10-linux-x86_64.egg/tvm/3rdparty  /usr/local/lib/python3.10/dist-packages/

export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages/triton-2.1.0-py3.10-linux-x86_64.egg:$PYTHONPATH"

