#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

########################## Evaluate Refcoco+ ##########################
user_dir=../../polyformer_module
bpe_dir=../../utils/BPE
selected_cols=0,1,2,3,4,5


train_on='refgrasp_rgbd'
test_on='refgrasp_single'
num_bins=64
batch_size=64



split='test'
ckpt_path=../../checkpoints/checkpoint_rgbd.pt
data=../../datasets/refgrasp/test/test_location_single.tsv
result_path=../../results_train_${train_on}_test_${test_on}
vis_dir=${result_path}/vis/${split}
result_dir=${result_path}/result/${split}
CUDA_VISIBLE_DEVICES=0 python3 ../../evaluate.py \
    ${data} \
    --path=${ckpt_path} \
    --user-dir=${user_dir} \
    --task=refcoco \
    --batch-size=${batch_size} \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --num-bins=${num_bins} \
    --result_dir=${result_dir} \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"

