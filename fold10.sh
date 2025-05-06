#!/bin/bash

# model_mel="vqvae_mel_n512d1024_full"
# model_cls="vqvae_cls_n512d1024_full"
model_name="CNNDual"
use_gpu_num=1

for i in $(seq 1 10);do
    for j in $(seq 0 9);do
        echo "----------executing sub-${i} fold-${j}----------";
        # echo $(date +%F%n%T)
        # python vqvae_mel.py --config ${model_mel} --sub ${i} --fold_num ${j} --save_logtxt --use_gpu_num ${use_gpu_num};
        # echo $(date +%F%n%T)
        # python vqvae_cls.py --config ${model_cls} --sub ${i} --fold_num ${j} --pretrain_model ${model_mel} --save_logtxt --use_gpu_num ${use_gpu_num};
        # echo $(date +%F%n%T)

        echo $(date +%F%n%T)
        python Model.py --model_name ${model_name} --model_config ${model_name} --sub ${i} --fold_num ${j} --save_logtxt --use_gpu_num ${use_gpu_num};
        echo $(date +%F%n%T)
    done;
done;

# python analyze.py --model_name ${model_name}