#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Pairwise-ranking baselines (Table 1). Same backbone, schedule and comparison
# graph as DOSE; only the loss changes:
#
#   marginrank : hinge ranking loss on the severity head only
#   ranknet    : logistic ranking loss on the severity head only
#
# Both are read out through the same evaluate.py routes as DOSE. Output tree,
# consumed by compare_baselines.py:
#   <EVAL_ROOT>/<DATASET>__<BUDGET>__<METHOD>__s<SEED>/summary.json
#
# Run from the repository root:  bash baselines/run_baselines.sh
# ---------------------------------------------------------------------------
set -euo pipefail

TRAIN_PY=./train.py
EVAL_PY=./evaluate.py
EVAL_ROOT=./baselines/eval
CKPT_ROOT=./baselines/ckpt

BACKBONE=convnext_tiny
EPOCHS=150
BATCH=32
LR=1e-4
NUM_GPUS=4
IMG=224
AUG=strong                          # augmentation used for the reported baseline runs

DATASETS="ACNE IDRID LIMUC"
BUDGETS="30 50 70"                  # % pair budget
SEEDS="1234 42 0"
METHODS="marginrank ranknet"

# Per-dataset train/test roots and class count. EDIT to match your paths.
dataset_train() { case "$1" in
  ACNE)  echo ./Acne_Split/train ;;
  IDRID) echo ./IDRID/train_images ;;
  LIMUC) echo ./LIMUC_organised/train ;;
esac }
dataset_test() { case "$1" in
  ACNE)  echo ./Acne_Split/test ;;
  IDRID) echo ./IDRID/test_images ;;
  LIMUC) echo ./LIMUC_organised/test ;;
esac }
dataset_nclasses() { case "$1" in
  ACNE) echo 4 ;; IDRID) echo 5 ;; LIMUC) echo 4 ;;
esac }

# Budget (%) -> pairs_per_image (same values as run.sh).
budget_pairs() { case "$1" in
  IDRID:30) echo 490 ;; IDRID:50) echo 790 ;; IDRID:70) echo 1090 ;;
  LIMUC:30) echo 1600 ;; LIMUC:50) echo 2400 ;; LIMUC:70) echo 3300 ;;
  ACNE:30) echo 160 ;; ACNE:50) echo 290 ;; ACNE:70) echo 440 ;;
  *) echo "ERROR: Unknown dataset/budget: $1" >&2 ; return 1 ;;
esac }

# Ranking-only baselines: no L_RMM, no dominance pseudo-pairs.
method_flags() { case "$1" in
  marginrank) echo "--lambda_clip 0 --lambda_rank 1.0 --no_dom_pairs --rank_loss hinge" ;;
  ranknet)    echo "--lambda_clip 0 --lambda_rank 1.0 --no_dom_pairs --rank_loss logistic" ;;
  *) echo "UNKNOWN_METHOD" ; return 1 ;;
esac }

port=12500
for DS in $DATASETS; do
  TR=$(dataset_train "$DS"); TE=$(dataset_test "$DS"); NC=$(dataset_nclasses "$DS")
  for B in $BUDGETS; do
    PAIRS=$(budget_pairs "${DS}:${B}")
    echo ""; echo "=== ${DS}  (K=${NC})  budget=${B}%  pairs_per_image=${PAIRS} ==="
    for M in $METHODS; do
      FLAGS=$(method_flags "$M")
      for SEED in $SEEDS; do
        tag="${DS}__${B}__${M}__s${SEED}"
        ckpt_dir="${CKPT_ROOT}/${tag}"
        eval_dir="${EVAL_ROOT}/${tag}"
        port=$((port + 1))
        echo "=== TRAIN ${tag} ==="
        python "$TRAIN_PY" \
          --data_dir "$TR" --backbone "$BACKBONE" \
          --n_classes "$NC" --image_size "$IMG" --aug "$AUG" \
          --epochs "$EPOCHS" --batch_size "$BATCH" --lr "$LR" \
          --pairs_per_image "$PAIRS" \
          --num_gpus "$NUM_GPUS" --ddp_port "$port" --seed "$SEED" \
          $FLAGS \
          --output_dir "$ckpt_dir"

        echo "=== EVAL ${tag} ==="
        python "$EVAL_PY" \
          --checkpoint "${ckpt_dir}/dose_final.pth" \
          --train-root "$TR" --test-root "$TE" --n-classes "$NC" \
          --image-size "$IMG" --pairs-per-image "$PAIRS" --seed "$SEED" \
          --output-dir "$eval_dir"
      done
    done
  done
done

echo
echo "Done. Aggregate with:"
echo "  python baselines/compare_baselines.py --runs-root ${EVAL_ROOT} --latex"
