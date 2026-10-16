"""
Aggregate supervised-upper-bound runs into mean +- std table rows, so the row
matches the 5-seed reporting of every other method.

Point it at a dir of run folders, each holding a summary.json from
train_supervised.py. Folder name is parsed as <DATASET>__<BACKBONE>__s<SEED>
(same '__' contract as the baseline runs); anything after the seed is ignored.

    python aggregate_supervised.py --runs-root ./sup [--latex]
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def parse(path):
    name = os.path.basename(os.path.dirname(path))
    parts = name.split("__")
    if len(parts) < 2:
        return None
    dataset, backbone = parts[0], parts[1]
    return dataset, backbone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    # (dataset, backbone) -> metric -> [values across seeds]
    agg = defaultdict(lambda: defaultdict(list))
    for p in sorted(glob.glob(os.path.join(args.runs_root, "*", "summary.json"))):
        meta = parse(p)
        if meta is None:
            print(f"[skip] {p} (name not <DS>__<BB>__s<seed>)")
            continue
        best = json.load(open(p)).get("best", {})
        for k in ("acc", "qwk", "mae", "macro_f1"):
            if best.get(k) is not None:
                agg[meta][k].append(float(best[k]))

    if not agg:
        raise SystemExit(f"no summary.json under {args.runs_root}")

    def ms(v, p):
        a = np.asarray(v)
        return f"{a.mean():.{p}f}+-{a.std():.{p}f}"

    print(f"{'Dataset':8s} {'Backbone':14s} {'n':>2s}  {'Acc':>12s} {'QWK':>13s} "
          f"{'MAE':>13s} {'F1':>12s}")
    print("-" * 80)
    for (ds, bb), m in sorted(agg.items()):
        n = len(m.get("acc", []))
        print(f"{ds:8s} {bb:14s} {n:2d}  {ms(m['acc'],1):>12s} {ms(m['qwk'],3):>13s} "
              f"{ms(m['mae'],3):>13s} {ms(m['macro_f1'],1):>12s}")

    if args.latex:
        print("\n% LaTeX rows (\\ms from main_results_table.tex)")
        for (ds, bb), m in sorted(agg.items()):
            def c(v, p):
                a = np.asarray(v)
                return f"\\ms{{{a.mean():.{p}f}}}{{{a.std():.{p}f}}}"
            print(f"{ds} & {bb} & {c(m['acc'],1)} & {c(m['qwk'],3)} & "
                  f"{c(m['mae'],3)} & {c(m['macro_f1'],1)} \\\\")


if __name__ == "__main__":
    main()
