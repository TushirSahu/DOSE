# DOSE: Dominance-Guided Ordinal Severity Estimation from Sparse Relative Annotations

Official PyTorch implementation of **DOSE** (WACV 2027).

DOSE learns medical severity grading from **sparse pairwise comparisons only**: each annotation
says whether one image is more severe, less severe, or comparable to another. No absolute grade is
ever used as a training target.

- **Dominance estimation.** The observed comparisons are aggregated into a global comparison
  graph, and Rank Centrality gives every training image a dominance score.
- **Dominance-aware multi-margin loss.** Dominance gaps are turned into ordinal distances that set
  cumulative margins in the embedding space, so more different images are pushed further apart.
- **Severity head.** A scalar head is trained on the annotated pairs plus confident
  dominance pseudo-pairs, giving a continuous severity axis.
- **Label-free calibration.** The axis is cut into K grades by tie-anchored segmentation
  (Tie-seg), compared with GMM, estimated-prior and known-prior (oracle) cut-offs.

## Installation

```bash
git clone <this-repo-url> DOSE && cd DOSE
pip install torch torchvision numpy scikit-learn pillow tqdm
```

## Data

The experiments use **ACNE04**, **IDRiD** (disease grading) and **LIMUC**. Download them from
their original sources and arrange each split with one folder per grade:

```
Acne_Split/
  train/  0/  1/  2/  3/
  test/   0/  1/  2/  3/
```

Folder names should be the grade number (`0`, `1`, ...). Names such as `grade_2` or `Mayo 3`
also work.

**Relative annotations.** The benchmarks only provide absolute grades, so comparisons are
simulated from them: each training image is compared with on average `--pairs_per_image` random
partners, and every compared pair gets the label `+1 / -1 / 0` (more / less / equally severe).
The model only ever sees these signs. Absolute grades are used for test metrics and, in the
Known-prior oracle, for the training class proportions.

Pair budgets used in the paper:

| Dataset | K | 30% | 50% | 70% |
|---|---|---|---|---|
| ACNE04 | 4 | 160 | 290 | 440 |
| IDRiD  | 5 | 490 | 790 | 1090 |
| LIMUC  | 4 | 1600 | 2400 | 3300 |

(values of `--pairs_per_image`; the training log prints the resulting pair coverage.)

## Training

The defaults reproduce the paper configuration: ConvNeXt-Tiny, 150 epochs, batch size 32 per GPU,
light augmentation, and `L = L_RMM + 0.5 L_rank + 0.5 L_pp`.

```bash
# 4-GPU DistributedDataParallel (default)
python train.py --data_dir ./Acne_Split/train --n_classes 4 --pairs_per_image 160 \
    --num_gpus 4 --seed 1234 --output_dir ./checkpoints/ACNE30_dose_s1234

# single GPU / CPU
python train.py --data_dir ./Acne_Split/train --n_classes 4 --pairs_per_image 160 \
    --no-distributed --output_dir ./checkpoints/ACNE30_dose_s1234
```

The paper results use 4 GPUs with 32 images each.

## Evaluation

```bash
python evaluate.py --checkpoint ./checkpoints/ACNE30_dose_s1234/dose_final.pth \
    --train-root ./Acne_Split/train --test-root ./Acne_Split/test \
    --n-classes 4 --pairs-per-image 160 --output-dir ./eval/ACNE30_dose_s1234
```

This prints accuracy, QWK, MAE and macro-F1 for the four calibration routes and writes
`summary.json`:

| Route | Labels used |
|---|---|
| Tie-seg | none (tie annotations only) |
| GMM | none |
| Est-prior | none |
| Known-prior | true training class proportions (oracle, reference only) |

## Reproducing the paper

Set the dataset paths at the top of each script, then run from the repository root.

```bash
bash run.sh                              # DOSE: datasets x budgets x seeds, then aggregate (Tables 1-2)
bash baselines/run_baselines.sh          # MarginRank and RankNet baselines (Table 1)
bash baselines/run_supervised.sh         # fully supervised cross-entropy upper bound (Table 1)
```

- **Margin ablation (Table 3):** uncomment `run single --single_margin` in `run.sh`.
- **Aggregate any set of runs:** `python aggregate_results.py --eval_dir ./eval --latex`
- The reported baseline runs used strong augmentation (`--aug strong`), as set in
  `baselines/run_baselines.sh`; DOSE uses the default light augmentation.

## Main arguments (`train.py`)

| Argument | Default | Description |
|---|---|---|
| `--pairs_per_image` | 170 | average annotated partners per image (pair budget) |
| `--backbone` | `convnext_tiny` | any torchvision `resnet*`, `densenet*` or `convnext*` |
| `--lambda_clip` / `--lambda_rank` / `--lambda_pp` | 1.0 / 0.5 / 0.5 | loss weights |
| `--rank_loss` | `hinge` | severity-head ranking loss (`hinge` or `logistic`) |
| `--pp_tau` | 0.1 | minimum dominance gap for a pseudo-pair |
| `--no_dom_pairs` | off | disable dominance pseudo-pairs |
| `--single_margin` | off | one shared margin instead of cumulative margins (ablation) |
| `--dominance` | `transitive` | `transitive` (Rank Centrality) or `copeland` (ablation) |
| `--aug` | `light` | `light` (paper) or `strong` |

## Repository structure

```
train.py              training (DDP or single device)
evaluate.py           four score-to-grade calibration routes + metrics
model.py              encoder, projection head and scalar severity head
dataset.py            grade-folder dataset and sparse comparison graph
losses.py             multi-/single-margin loss, ranking losses, dominance scores
calibration.py        tie-anchored segmentation (Tie-seg)
eval_utils.py         test loader, GMM/prior cut-offs, metrics
aggregate_results.py  mean +- std over seeds
run.sh                full DOSE experiment
baselines/            MarginRank, RankNet and supervised cross-entropy baselines
```

## Citation

```bibtex
@inproceedings{dose2027,
  title     = {DOSE: Dominance-Guided Ordinal Severity Estimation from Sparse Relative Annotations},
  author    = {},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE).
