#!/usr/bin/env bash
set -euo pipefail

# GPU 1 (빠른 GPU, 6개, ~15시간)
#   deeplabv3 × false × {2e-2, 3e-2, 4e-2, 5e-2}
#   pspnet    × true  × {1e-2, 2e-2}

DATASET="${DATASET-cityscapes}"
MAX_ITER=200
SAVE_INTERVAL=200
EARLY_STOP=false

case "${DATASET}" in
  VOC2012)
    DATA_DIR="../datasets/VOC2012"
    BASE_DIR="./data/sPGD/results/VOC2012/spgd_seg"
    ;;
  cityscapes)
    DATA_DIR="../datasets/cityscapes"
    BASE_DIR="./data/sPGD/results/cityscapes/spgd_seg"
    ;;
  ade20k)
    DATA_DIR="../datasets/ade20k"
    BASE_DIR="./data/sPGD/results/ade20k/spgd_seg"
    ;;
  *)
    echo "Unsupported DATASET: ${DATASET}"
    exit 1
    ;;
esac

# deeplabv3 × false × 4개
MODEL="deeplabv3"
UNPROJECTED="false"
for ATTACK_PIXEL in "2e-2" "3e-2" "4e-2" "5e-2"; do
  echo "Running sPGD-seg attack for ${DATASET} ${MODEL} (attack_pixel=${ATTACK_PIXEL}, unprojected_gradient=${UNPROJECTED})"
  python spgd_seg_voc_attack.py \
    --dataset "${DATASET}" \
    --model "${MODEL}" \
    --data_dir "${DATA_DIR}" \
    --base_dir "${BASE_DIR}" \
    --attack_pixel "${ATTACK_PIXEL}" \
    --max_iter "${MAX_ITER}" \
    --save_interval "${SAVE_INTERVAL}" \
    --early_stop "${EARLY_STOP}" \
    --unprojected_gradient "${UNPROJECTED}"
done

# pspnet × true × 2개
MODEL="pspnet"
UNPROJECTED="true"
for ATTACK_PIXEL in "1e-2" "2e-2"; do
  echo "Running sPGD-seg attack for ${DATASET} ${MODEL} (attack_pixel=${ATTACK_PIXEL}, unprojected_gradient=${UNPROJECTED})"
  python spgd_seg_voc_attack.py \
    --dataset "${DATASET}" \
    --model "${MODEL}" \
    --data_dir "${DATA_DIR}" \
    --base_dir "${BASE_DIR}" \
    --attack_pixel "${ATTACK_PIXEL}" \
    --max_iter "${MAX_ITER}" \
    --save_interval "${SAVE_INTERVAL}" \
    --early_stop "${EARLY_STOP}" \
    --unprojected_gradient "${UNPROJECTED}"
done
