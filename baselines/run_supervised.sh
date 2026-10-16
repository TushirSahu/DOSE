#!/usr/bin/env bash
# Supervised upper bound over all datasets x backbones x seeds, then aggregate.
#
# Fine-tune full backbone (the ceiling) by default. Set FREEZE=1 for the linear
# probe instead. Resumable: skips any run whose summary.json already exists.
#
#   bash baselines/run_supervised.sh                       # ConvNeXt-Tiny, all datasets, 5 seeds
#   BACKBONES="resnet50 densenet121" bash baselines/run_supervised.sh
#   FREEZE=1 bash baselines/run_supervised.sh              # linear-probe variant
#
# Run from the repository root. Edit dataset_train/dataset_test to match your paths
# (same as run.sh), then run. Output: OUT/<DS>__<BB>__s<seed>/summary.json
set -u

TRAIN_PY=./baselines/train_supervised.py
AGG_PY=./baselines/aggregate_supervised.py
OUT=${OUT:-./baselines/sup}
EPOCHS=${EPOCHS:-50}
BATCH=${BATCH:-32}
IMG=${IMG:-224}
FREEZE=${FREEZE:-0}
SEEDS=${SEEDS:-"1234 42 0 1 1337"}
DATASETS=${DATASETS:-"ACNE IDRID LIMUC"}
BACKBONES=${BACKBONES:-"convnext_tiny"}
mkdir -p "${OUT}"

freeze_flag=""; tag_suffix=""
[ "${FREEZE}" = "1" ] && { freeze_flag="--freeze"; tag_suffix="_probe"; }

# Must match run.sh so the upper bound trains on the same data.
dataset_train() { case "$1" in
  ACNE) echo ./Acne_Split/train ;; IDRID) echo ./IDRID/train_images ;; LIMUC) echo ./LIMUC_organised/train ;;
esac }
dataset_test() { case "$1" in
  ACNE) echo ./Acne_Split/test ;; IDRID) echo ./IDRID/test_images ;; LIMUC) echo ./LIMUC_organised/test ;;
esac }
dataset_nclasses() { case "$1" in
  ACNE) echo 4 ;; IDRID) echo 5 ;; LIMUC) echo 4 ;;
esac }

done_n=0; skip_n=0; fail_n=0
for DS in ${DATASETS}; do
  TR=$(dataset_train "$DS"); TE=$(dataset_test "$DS"); NC=$(dataset_nclasses "$DS")
  if [ ! -d "${TR}" ]; then
    echo "WARN: train dir '${TR}' missing for ${DS}, skipping dataset" >&2; continue
  fi
  for BB in ${BACKBONES}; do
    for SEED in ${SEEDS}; do
      tag="${DS}__${BB}__s${SEED}${tag_suffix}"
      dir="${OUT}/${tag}"
      if [ -f "${dir}/summary.json" ]; then
        echo "skip ${tag} (done)"; skip_n=$((skip_n+1)); continue
      fi
      echo "==> ${tag}"
      if python "${TRAIN_PY}" \
           --train-root "${TR}" --test-root "${TE}" \
           --backbone "${BB}" --n-classes "${NC}" \
           --image-size "${IMG}" --epochs "${EPOCHS}" --batch-size "${BATCH}" \
           --seed "${SEED}" ${freeze_flag} --output-dir "${dir}"; then
        done_n=$((done_n+1))
      else
        echo "FAIL ${tag}" >&2; fail_n=$((fail_n+1))
      fi
    done
  done
done

echo
echo "trained=${done_n}  skipped=${skip_n}  failed=${fail_n}"
echo "=== aggregate (mean +- std over seeds) ==="
python "${AGG_PY}" --runs-root "${OUT}" --latex
