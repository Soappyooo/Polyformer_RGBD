#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

########################## Evaluate Refcoco+ ##########################
user_dir=../../polyformer_module
bpe_dir=../../utils/BPE
selected_cols=0


train_on='roborefit'
test_on='ocid_vlg'
num_bins=64
batch_size=64



split='test'
ckpt_path=../../../models/checkpoint_roborefit.pt
data=../../datasets/ocid_vlg/test.tsv
result_path=../../results_train_${train_on}_test_${test_on}
vis_dir=${result_path}/vis/${split}
result_dir=${result_path}/result/${split}
python3 ../../evaluate.py \
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
    --num-workers=8 \
    --num-bins=${num_bins} \
    --vis_dir=${vis_dir} --vis \
    --result_dir=${result_dir} \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"

