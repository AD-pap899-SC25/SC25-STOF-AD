#!/bin/bash
# bash env_install.sh 80

CUDAARCH=${1:-80}  #  A100:80 4090:89

echo "[SHELL INFO] CUDA Arch: sm_${CUDAARCH}"
echo "[SHELL INFO] Platform: ${PLATFORM}"

rm -f *.so
echo "[SHELL INFO] remove archive *.so file and compile again ... ..."

rm ~/.triton/cache
echo "[SHELL INFO] delete triton cache ... ..."

cd ../src

python setup.py ${CUDAARCH} build_ext --inplace
echo "[SHELL INFO] CUDA extension compiled success !"

cd Bytetr_MCFuser
rm -f *.so
python setup.py ${CUDAARCH} build_ext --inplace
cd ../
echo "[SHELL INFO] Bytetransformer CUDA extension compiled success !"


python correct_verify_attn.py
echo "[SHELL INFO] Correctness verification for Attention completed !"

python correct_verify_end2end.py
echo "[SHELL INFO] Correctness verification for End2End completed !"

