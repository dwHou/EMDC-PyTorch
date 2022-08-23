#!/usr/bin/env bash 

nohup python3 -u train_model.py --arch EMDC \
--name w10e150_loss14 \
--decay_type warmup_cosine --num_warmup_epochs 10 --num_epochs 150 \
--loss criterion1 \
--loss criterion4 \
--gpu 0  --learning_rate 1e-3 --train_batch_size 10 >./logs/EMDC_w10e150_loss14.log 2>&1 &
