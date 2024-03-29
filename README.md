# DiffTCL
 
:rocket: The first version of the code will be open-sourced soon.


conda create -n difftcl python=3.8.16

pip install -r requirements


python3 train.py --config configs/diff-pre-training.yaml


python3 scripts/extract_model_weights.py -c path/to/checkpoint/file

sh scripts/train.sh 4 <port>
