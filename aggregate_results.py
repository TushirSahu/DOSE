import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np


ROUTES = ["tie_classifier", "score_thresh_gmm", "score_thresh_estimated", "score_thresh_known"]
METRICS = ["acc", "qwk", "mae", "macro_f1"]  # per_class_recall handled separately
ROUTE_LABEL = {
    "tie_classifier":         "Tie-seg (label-free)",
    "score_thresh_gmm":       "GMM (label-free)",
    "score_thresh_estimated": "Est-prior (label-free)",
    "score_thresh_known":     "Known-prior (oracle)",
}


def load_jsons(eval_dir: Path):
    """Return dict: (ds, tag) -> list of summary dicts across seeds."""
    pattern = re.compile(r"^(?P<ds>[^_]+)_(?P<tag>.+)_s(?P<seed>\d+)$")
    groups = defaultdict(list)
    for d in sorted(eval_dir.iterdir()):
        m = pattern.match(d.name)
        if not m:
            continue
        jf = d / "summary.json"
        if not jf.exists():
            print(f"  [warn] missing {jf}, skipping")
            continue
        with open(jf) as f:
            groups[(m.group("ds"), m.group("tag"))].append(json.load(f))
    return groups


def agg(values):
    """Return (mean, std) over seeds."""
    arr = np.array(values, dtype=float)
    return arr.mean(), arr.std(ddof=0)


def fmt(mean, std, decimals=1):
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def summarise(runs: list, route: str, n_classes: Optional[int] = None):
    """Aggregate one route across all seed runs. Returns dict of metric -> (mean,std)."""
    out = {}
    for metric in METRICS:
        vals = []
        for r in runs:
            try:
                vals.append(r["results"][route][metric])
            except KeyError:
                pass
        if vals:
            out[metric] = agg(vals)

    # per-class recall: list of K floats per run
    pcr_all = []
    for r in runs:
        try:
            pcr_all.append(r["results"][route]["per_class_recall"])
        except KeyError:
            pass
    if pcr_all:
        K = max(len(p) for p in pcr_all)
        per_class = []
        for k in range(K):
            vals = [p[k] for p in pcr_all if k < len(p)]
            per_class.append(agg(vals))
        out["per_class_recall"] = per_class

    out["n_seeds"] = len(runs)
    return out


def print_table(groups, route, latex=False):
    lines = []
    header = f"\n{'='*70}\n  Route: {ROUTE_LABEL.get(route, route)}"
    print(header)
    lines.append(header)

    latex_rows = []
    for (ds, tag), runs in sorted(groups.items()):
        s = summarise(runs, route)
        n = s.get("n_seeds", "?")
        acc_m, acc_s   = s.get("acc",      (float("nan"), 0))
        qwk_m, qwk_s   = s.get("qwk",      (float("nan"), 0))
        mae_m, mae_s   = s.get("mae",      (float("nan"), 0))
        f1_m,  f1_s    = s.get("macro_f1", (float("nan"), 0))

        pcr = s.get("per_class_recall", [])
        pcr_str = "  ".join(fmt(m, sd) for m, sd in pcr)

        p1 = f"\n  {ds} / {tag}  (n={n} seeds)"
        p2 = f"    Acc       {fmt(acc_m, acc_s, 1)} %"
        p3 = f"    QWK       {fmt(qwk_m, qwk_s, 3)}"
        p4 = f"    MAE       {fmt(mae_m, mae_s, 3)}"
        p5 = f"    Macro-F1  {fmt(f1_m, f1_s, 1)} %"
        print(p1)
        print(p2)
        print(p3)
        print(p4)
        print(p5)
        lines.extend([p1, p2, p3, p4, p5])

        if pcr_str:
            p6 = f"    Per-class recall:  {pcr_str}"
            print(p6)
            lines.append(p6)

        if latex:
            escaped_tag = tag.replace("_", "\\_")
            row = (
                f"{ds} & {escaped_tag} & "
                f"${fmt(acc_m, acc_s, 1)}$ & "
                f"${fmt(qwk_m, qwk_s, 3)}$ & "
                f"${fmt(mae_m, mae_s, 3)}$ & "
                f"${fmt(f1_m, f1_s, 1)}$ "
            )
            latex_rows.append(row)

    if latex and latex_rows:
        l1 = f"\n{'─'*70}\n  LaTeX rows ({route}):\n{'─'*70}"
        print(l1)
        lines.append(l1)
        for row in latex_rows:
            l2 = "  " + row
            print(l2)
            lines.append(l2)

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True,
                    help="Directory containing <DS>_<tag>_s<seed>/ subdirs")
    ap.add_argument("--route", default=None,
                    help="Which eval route to show (default: all). "
                         f"Choices: {ROUTES}")
    ap.add_argument("--latex", action="store_true",
                    help="Also print LaTeX-ready table rows")
    ap.add_argument("--output_file", default="./aggregated_results.txt",
                    help="Path to the txt file where all metrics will be saved")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        raise SystemExit(f"eval_dir not found: {eval_dir}")

    groups = load_jsons(eval_dir)
    if not groups:
        raise SystemExit("No <DS>_<tag>_s<seed>/summary.json files found.")

    output_lines = []
    routes = [args.route] if args.route else ROUTES
    for route in routes:
        route_content = print_table(groups, route, latex=args.latex)
        output_lines.append(route_content)

    footer = (
        f"\n{'='*70}\n"
        f"  Configs found: {len(groups)}  |  Seeds per config vary (see n= above)\n"
        f"  To export LaTeX rows: add --latex flag\n"
        f"{'='*70}\n"
    )
    print(footer)
    output_lines.append(footer)

    # Save to text file
    out_path = Path(args.output_file)
    try:
        out_path.write_text("\n".join(output_lines), encoding="utf-8")
        print(f"Successfully saved all metrics to: {out_path}\n")
    except Exception as e:
        print(f"Error saving to file {out_path}: {e}\n")


if __name__ == '__main__':
    main()