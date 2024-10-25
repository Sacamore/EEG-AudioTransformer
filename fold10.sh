#!/bin/bash

for((i=1;i<=10;i++));do
    for((j=0;j<10;j++));do
        echo "----------executing sub-${i} fold-${j}----------";
        python vqvae_mel.py --config vqvae_mel_n512d1024 --sub ${i} --fold_num ${j} --save_tensorboard false --save_logtxt true;
    done;
done;