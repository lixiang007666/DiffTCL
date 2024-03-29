#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

dataset='tcl-ms'
method='tcl'
exp='r101'
split='732'

config=configs/${dataset}.yaml
# labeled_id_path=splits/GOALS/$split/labeled.txt
# unlabeled_id_path=splits/GOALS/$split/unlabeled.txt
labeled_id_path=splits/ms/$split/labeled.txt
unlabeled_id_path=splits/ms/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/$split

mkdir -p $save_path

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.log
