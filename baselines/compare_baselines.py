"""
Baseline comparison table: reads the per-run summary.json files written by evaluate.py and
aggregates them into a method x calibration-route table (mean +- std over seeds), per dataset
and pair budget.

Directory naming (produced by run_baselines.sh):
    <runs-root>/<DATASET>__<BUDGET>__<METHOD>__s<SEED>/summary.json

Example:
    python compare_baselines.py --runs-root ./baselines/eval --latex
"""
import argparse
import glob
import json
import os

import numpy as np

# Calibration routes to report, in column order. Keys match evaluate.py's results dict.
DEFAULT_ROUTES = ["tie_classifier", "score_thresh_gmm",
                  "score_thresh_estimated", "score_thresh_known"]

ROUTE_LABELS = {
    "tie_classifier":         "Tie-seg",
    "score_thresh_gmm":       "GMM",
    "score_thresh_estimated": "Est-prior",
    "score_thresh_known":     "Known-prior",
}

# Method row order + pretty labels. Methods not listed here are appended in the
# order first seen, using their raw token as the label.
METHOD_ORDER = ["marginrank", "ranknet", "dose"]
METHOD_LABELS = {
    "marginrank":       "MarginRank",
    "ranknet":          "RankNet",
    "dose":             "DOSE (ours)",
    "ranknet_hinge":    "MarginRank",      # run names from earlier versions
    "ranknet_logistic": "RankNet",
}

BUDGET_ORDER = ["30", "50", "70"]


def parse_run_dir(path):
    """('ACNE','30','ranknet','1234') from a .../<D>__<B>__<M>__s<seed> dir."""
    name = os.path.basename(os.path.dirname(path))
    parts = name.split("__")
    if len(parts) != 4:
        return None
    dataset, budget, method, seedtok = parts
    seed = seedtok[1:] if seedtok.lower().startswith("s") else seedtok
    return dataset, budget, method, seed


def collect_summaries(runs_root, routes):
    """nested[dataset][budget][method][route] -> list of acc; plus qwk list."""
    nested, qwk = {}, {}
    skipped = []
    for p in sorted(glob.glob(os.path.join(runs_root, "*", "summary.json"))):
        meta = parse_run_dir(p)
        if meta is None:
            skipped.append(p)
            continue
        dataset, budget, method, _seed = meta
        with open(p) as f:
            res = json.load(f).get("results", {})
        d = nested.setdefault(dataset, {}).setdefault(budget, {}).setdefault(method, {})
        for route in routes:
            acc = res.get(route, {}).get("acc")
            if acc is not None:
                d.setdefault(route, []).append(float(acc))
        # QWK is reported off the tie-seg route (the deployed readout).
        q = res.get("tie_classifier", {}).get("qwk")
        if q is not None:
            qwk.setdefault(dataset, {}).setdefault(budget, {}).setdefault(method, []).append(float(q))
    return nested, qwk, skipped


def ms(vals, prec=1):
    if not vals:
        return "  --  "
    a = np.asarray(vals, dtype=float)
    return f"{a.mean():.{prec}f}±{a.std():.{prec}f}"


def ordered(keys, order):
    known = [k for k in order if k in keys]
    rest = [k for k in keys if k not in order]
    return known + rest


def print_text(nested, qwk, routes):
    for dataset in nested:
        print("\n" + "=" * 78)
        print(f"  {dataset}")
        print("=" * 78)
        hdr = ["Method", "Budget"] + [ROUTE_LABELS.get(r, r) for r in routes] + ["QWK"]
        widths = [32, 8] + [13] * len(routes) + [13]
        print("".join(h.ljust(w) for h, w in zip(hdr, widths)))
        print("-" * sum(widths))
        budgets = ordered(nested[dataset].keys(), BUDGET_ORDER)
        methods = sorted({m for b in nested[dataset].values() for m in b})
        for method in ordered(methods, METHOD_ORDER):
            label = METHOD_LABELS.get(method, method)
            for bi, budget in enumerate(budgets):
                cell = nested[dataset].get(budget, {}).get(method)
                if not cell:
                    continue
                row = [label if bi == 0 else "", f"{budget}%"]
                row += [ms(cell.get(r, [])) for r in routes]
                row += [ms(qwk.get(dataset, {}).get(budget, {}).get(method, []), prec=3)]
                print("".join(c.ljust(w) for c, w in zip(row, widths)))
            print("-" * sum(widths))


def print_latex(nested, qwk, routes):
    print("\n% ---- LaTeX (booktabs); \\ms helper from main_results_table.tex ----")
    col = "ll" + "c" * (len(routes) + 1)
    print("\\begin{tabular}{" + col + "}")
    print("\\toprule")
    head = ["Method", "Budget"] + [ROUTE_LABELS.get(r, r) for r in routes] + ["QWK"]
    print(" & ".join(head) + " \\\\")
    print("\\midrule")
    for dataset in nested:
        print(f"\\multicolumn{{{len(routes) + 3}}}{{l}}{{\\emph{{{dataset}}}}} \\\\")
        budgets = ordered(nested[dataset].keys(), BUDGET_ORDER)
        methods = sorted({m for b in nested[dataset].values() for m in b})
        for method in ordered(methods, METHOD_ORDER):
            label = METHOD_LABELS.get(method, method)
            for bi, budget in enumerate(budgets):
                cell = nested[dataset].get(budget, {}).get(method)
                if not cell:
                    continue
                def mscell(vals, p=1):
                    if not vals:
                        return "--"
                    a = np.asarray(vals, float)
                    return f"\\ms{{{a.mean():.{p}f}}}{{{a.std():.{p}f}}}"
                cols = [label if bi == 0 else "", f"{budget}\\%"]
                cols += [mscell(cell.get(r, [])) for r in routes]
                cols += [mscell(qwk.get(dataset, {}).get(budget, {}).get(method, []), p=3)]
                print(" & ".join(cols) + " \\\\")
        print("\\midrule")
    print("\\bottomrule")
    print("\\end{tabular}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True,
                    help="Dir of <D>__<B>__<METHOD>__s<seed>/summary.json folders.")
    ap.add_argument("--routes", nargs="+", default=DEFAULT_ROUTES)
    ap.add_argument("--latex", action="store_true", help="Also print a LaTeX table.")
    args = ap.parse_args()

    nested, qwk, skipped = collect_summaries(args.runs_root, args.routes)

    if skipped:
        print(f"[warn] {len(skipped)} summaries skipped (name not "
              f"<D>__<B>__<METHOD>__s<seed>):")
        for p in skipped[:8]:
            print(f"        {p}")
    if not nested:
        raise SystemExit(f"No parseable runs under {args.runs_root}")

    print_text(nested, qwk, args.routes)
    if args.latex:
        print_latex(nested, qwk, args.routes)


if __name__ == "__main__":
    main()
