#!/bin/bash



########################## Evaluate Refcoco+ ##########################
user_dir=../../polyformer_module
bpe_dir=../../utils/BPE
selected_cols=0,1,2,3,4,5


model='polyformer_b_refgrasp'
num_bins=64
batch_size=64



dataset='refgrasp'
split='test'
ckpt_path=../../run_scripts/finetune/polyformer_b_checkpoints/25_5e-5_512/checkpoint_best.pt
data=../../datasets/refgrasp/refgrasp_no_depth/test/test.tsv
result_path=../../results_${model}/${dataset}/epoch_${epoch}
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
    --num-workers=0 \
    --num-bins=${num_bins} \
    --vis_dir=${vis_dir} --vis \
    --result_dir=${result_dir} \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"

