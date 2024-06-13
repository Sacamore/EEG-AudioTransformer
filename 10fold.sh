#!/bin/bash
for ((i=1;i<10;i++))
do
    python vqvae_mel.py --config vqvae_mel_n512_d1024 --fold_num $i
    python vqvae_cls.py --config vqvae_cls_n512_d1024 --fold_num $i --pretrain_model vqvae_mel_n512_d1024_$i
done
echo "10 fold mel vqvae done"