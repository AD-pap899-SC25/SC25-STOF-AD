
export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages/triton-2.1.0-py3.10-linux-x86_64.egg:$PYTHONPATH"

rm  /usr/local/lib/python3.10/dist-packages/transformers/models/bert/modeling_bert.py
cp  ../src/MCFuser/mcfuser/ae/scripts/e2e/bert_modeling_bert.py /usr/local/lib/python3.10/dist-packages/transformers/models/bert/modeling_bert.py
script -a -c "python supb_bert.py" table4.txt


rm  /usr/local/lib/python3.10/dist-packages/transformers/models/bert/modeling_bert.py
cp  ../src/MCFuser/mcfuser/ae/scripts/e2e/gpt_modeling_bert.py /usr/local/lib/python3.10/dist-packages/transformers/models/bert/modeling_bert.py
script -a -c "python supb_gpt.py" table4.txt


rm  /usr/local/lib/python3.10/dist-packages/transformers/models/bert/modeling_bert.py
cp  ../src/MCFuser/mcfuser/ae/scripts/e2e/t5_modeling_bert.py /usr/local/lib/python3.10/dist-packages/transformers/models/bert/modeling_bert.py
script -a -c "python supb_t5.py" table4.txt