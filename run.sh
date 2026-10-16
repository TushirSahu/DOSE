#!/bin/bash
# ==============================================================================
# DOSE: train + evaluate over datasets x pair budgets x seeds, then aggregate.
#
# Paper configuration (the train.py defaults): ConvNeXt-Tiny, 150 epochs, batch 32 per GPU,
# light augmentation, L = L_RMM + 0.5 L_rank + 0.5 L_pp, Rank-Centrality dominance.
#
# Run from the repository root:   bash run.sh
# Set the dataset paths below first. Each dataset root holds one sub-folder per grade.
# Outputs:  CKPT/<DS><BUDGET>_<tag>_s<seed>/dose_final.pth
#           EVAL/<DS><BUDGET>_<tag>_s<seed>/summary.json
# ==============================================================================
set -e

CKPT="./checkpoints"
EVAL="./eval"
NGPU=4
PORT=12400
SEEDS=(1234 42 0 1 1337)
BUDGETS=(30 50 70)

# name | train_dir | test_dir | n_classes
DATASETS=(
  "ACNE   ./Acne_Split/train        ./Acne_Split/test        4"
  "IDRID  ./IDRID/train_images      ./IDRID/test_images      5"
  "LIMUC  ./LIMUC_organised/train   ./LIMUC_organised/test   4"
)

# Pair budget (%) -> average annotated partners per image (--pairs_per_image).
budget_pairs() { case "$1" in
  ACNE:30) echo 160 ;;  ACNE:50) echo 290 ;;  ACNE:70) echo 440 ;;
  IDRID:30) echo 490 ;; IDRID:50) echo 790 ;; IDRID:70) echo 1090 ;;
  LIMUC:30) echo 1600 ;; LIMUC:50) echo 2400 ;; LIMUC:70) echo 3300 ;;
  *) echo "ERROR: unknown dataset/budget $1" >&2; return 1 ;;
esac }

# run <tag> [extra train.py flags...]  -> one config over all seeds
run () {
  local tag="$1"; shift
  for seed in "${SEEDS[@]}"; do
    PORT=$((PORT+1))
    local name="${DS}${B}_${tag}_s${seed}"
    echo ""; echo "#### ${name} (pairs_per_image=${PPI}) ####"
    python ./train.py \
      --data_dir "${TRAIN_DIR}" --n_classes ${NCLS} --pairs_per_image ${PPI} \
      --num_gpus ${NGPU} --ddp_port ${PORT} --seed "${seed}" \
      --output_dir ${CKPT}/${name} "$@"
    python ./evaluate.py --checkpoint ${CKPT}/${name}/dose_final.pth \
      --train-root "${TRAIN_DIR}" --test-root "${TEST_DIR}" --n-classes ${NCLS} \
      --pairs-per-image ${PPI} --output-dir ${EVAL}/${name}
  done
}

for cfg in "${DATASETS[@]}"; do
  read -r DS TRAIN_DIR TEST_DIR NCLS <<< "${cfg}"
  [ -d "${TRAIN_DIR}" ] || { echo "skip ${DS}: ${TRAIN_DIR} missing"; continue; }
  for B in "${BUDGETS[@]}"; do
    PPI=$(budget_pairs "${DS}:${B}")
    echo ""; echo "=================  ${DS} (K=${NCLS})  budget=${B}%  ================="
    run dose                                   # DOSE (Tables 1-2)
    # run single --single_margin               # single-margin ablation (Table 3)
  done
done

python ./aggregate_results.py --eval_dir ${EVAL}
