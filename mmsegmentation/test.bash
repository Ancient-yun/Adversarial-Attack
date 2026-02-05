#!/bin/bash
#configs_attack/ade20k/config_mask2_swin_B.py
#configs_attack/ade20k/config_seg.py
#configs_attack/ade20k/config_pspnet.py
#configs_attack/ade20k/config_deeplabv3.py
#configs_attack/ade20k/config_setr.py

#configs_attack/cityscapes/config_mask2_swin_B.py
#configs_attack/cityscapes/config_seg.py
#configs_attack/cityscapes/config_pspnet.py
#configs_attack/cityscapes/config_deeplabv3.py
#configs_attack/cityscapes/config_setr.py
# export CUDA_VISIBLE_DEVICES=0
#'margin', 'prob', 'discrepancy', 'baseline', 'reduction', 'adap_reduction'

datasets=("cityscapes")
models=("config_deeplabv3" "config_pspnet" "config_seg" "config_setr")
# loss=("discrepancy" "baseline")
loss=("reduction" "adap_reduction")

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        for l in "${loss[@]}"; do
            python px_eval.py \
                --config configs_attack/${dataset}/${model}.py \
                --loss ${l} \
                --restarts 100 \
                --max_iterations 10 \
                --num_images 100
        done
    done
done



for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        for l in "${loss[@]}"; do
            python rs_eval.py \
                --config configs_attack/${dataset}/${model}.py \
                --loss ${l} \
                --iters 100 \
                --n_queries 10 \
                --num_images 100
        done
    done
done

threshold=(0.5 0,6)
for threshold in ; do
    python pw_eval.py --config configs_attack/VOC2012/config_pspnet.py \
        --attack_mode scheduling \
        --max_query 1000 \
        --npix 1960 \
        --num_images 100 \
        --success_threshold ${threshold}
done

for model in deeplabv3 pspnet; do
    for threshold in 0.5 0.6; do
        python spaevo_eval.py --config configs_attack/VOC2012/config_${model}.py \
            --max_query 1000 \
            --num_images 100 \
            --n_pix 1960 \
            --pop_size 100 \
            --success_threshold ${threshold} \
            --verbose
    done
done






for model in deeplabv3 pspnet seg setr; do
    for threshold in 0.2 0.3; do
        python pw_eval.py --config configs_attack/cityscapes/config_${model}.py \
            --max_query 1000 \
            --npix 15680 \
            --num_images 100 \
            --attack_mode scheduling \
            --success_threshold ${threshold} \
            --verbose
    done
done

for model in deeplabv3 pspnet seg setr; do
    for threshold in 0.2 0.3; do
        python spaevo_eval.py --config configs_attack/cityscapes/config_${model}.py \
            --max_query 1000 \
            --num_images 100 \
            --n_pix 1960 \
            --pop_size 100 \
            --success_threshold ${threshold} \
            --verbose
    done
done

for model in deeplabv3; do
    for threshold in 0.5; do
        for npix in 16384 3276; do
            python pw_eval.py --config configs_attack/VOC2012/config_${model}.py \
                --max_query 1000 \
                --npix ${npix} \
                --num_images 100 \
                --attack_mode scheduling \
                --success_threshold ${threshold} \
                --verbose
        done
    done
done

for model in deeplabv3; do
    for threshold in 0.5; do
        for npix in 16384 32768; do
            python spaevo_eval.py --config configs_attack/VOC2012/config_${model}.py \
                --max_query 1000 \
                --num_images 10 \
                --n_pix ${npix} \
                --pop_size 100 \
                --success_threshold ${threshold} \
                --verbose
        done
    done
done

for model in deeplabv3; do
    for threshold in 0.5; do
        for npix in 256 512 1024 2048 4096 8192 16384; do
            python spaevo_eval.py --config configs_attack/VOC2012/config_${model}.py \
                --max_query 1000 \
                --num_images 10 \
                --n_pix ${npix} \
                --pop_size 100 \
            --success_threshold ${threshold} \
            --verbose
        done
    done
done

python pw_eval.py --config configs_attack/VOC2012/config_deeplabv3.py \
    --max_query 1000 \
    --npix  1024 \
    --num_images 100 \
    --attack_mode scheduling \
    --success_threshold 0.5 \
    --verbose

python spaevo_eval.py --config configs_attack/VOC2012/config_deeplabv3.py \
    --max_query 1000 \
    --num_images 100 \
    --n_pix 1960 \
    --success_threshold 0.5 \
    --verbose