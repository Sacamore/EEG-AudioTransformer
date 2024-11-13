#!/bin/bash

for i in $(seq 1 10);do
    for j in $(seq 0 9);do
        echo "----------executing sub-${i} fold-${j}----------";
        echo $(date +%F%n%T)
        python vqvae_mel.py --config vqvae_mel_n512d1024_full --sub ${i} --fold_num ${j} --save_logtxt --use_gpu_num 0;
        echo $(date +%F%n%T)
        python vqvae_cls.py --config vqvae_cls_n512d1024_full --sub ${i} --fold_num ${j} --pretrain_model vqvae_mel_n512d1024_full --save_logtxt --use_gpu_num 0;
        echo $(date +%F%n%T)
    done;
done;

# python analyze.py