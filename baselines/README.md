# Baselines

Scripts for the comparison rows of Table 1. Run everything from the repository root.

## Pairwise-ranking baselines (MarginRank, RankNet)

Same backbone, schedule and comparison graph as DOSE; only the loss changes. Both train the
severity head with a ranking loss alone (no `L_RMM`, no dominance pseudo-pairs) and are read out
through the same four `evaluate.py` routes.

| Method | Loss on the severity head | `train.py` flags |
|---|---|---|
| MarginRank | hinge `max(0, m - r·δ)` | `--lambda_clip 0 --lambda_rank 1.0 --no_dom_pairs --rank_loss hinge` |
| RankNet | logistic `log(1 + exp(-r·δ))` | `--lambda_clip 0 --lambda_rank 1.0 --no_dom_pairs --rank_loss logistic` |

```bash
bash baselines/run_baselines.sh
python baselines/compare_baselines.py --runs-root ./baselines/eval --latex
```

## Fully supervised upper bound (Cross-entropy)

ImageNet backbone + one linear layer trained with cross-entropy on all absolute grades.

```bash
bash baselines/run_supervised.sh          # trains, then prints mean ± std over seeds
```

## Files

- `run_baselines.sh` — trains and evaluates MarginRank / RankNet into
  `baselines/eval/<DATASET>__<BUDGET>__<METHOD>__s<SEED>/summary.json`.
- `compare_baselines.py` — aggregates those runs into a method × route table (text + LaTeX).
- `train_supervised.py` — supervised cross-entropy model (`--freeze` for a linear probe).
- `test_supervised.py` — re-scores a saved supervised checkpoint.
- `run_supervised.sh` / `aggregate_supervised.py` — run all datasets × seeds and aggregate.
