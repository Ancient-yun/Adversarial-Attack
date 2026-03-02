#!/usr/bin/env bash
set -euo pipefail

# GPU 2 (중간 GPU, 6.72t)
#   deeplabv3 × false × {2e-2, 3e-2, 4e-2}  (3 × 2.24t = 6.72t)
#   합계: 6.72t

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

MODEL="deeplabv3"
UNPROJECTED="false"

for ATTACK_PIXEL in "2e-2" "3e-2" "4e-2"; do
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
