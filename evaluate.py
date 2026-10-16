"""Evaluate a DOSE checkpoint: convert the learned severity axis into K grades with the four
score-to-grade routes of Sec. 3.5 and report metrics on the test set.

    Tie-seg     (label-free) : tie-anchored segmentation of the training scores (Eq. 12)
    GMM         (label-free) : midpoints between sorted Gaussian-mixture means (Eq. 14)
    Est-prior   (label-free) : score quantiles at the GMM mixture weights (Eq. 15)
    Known-prior (oracle)     : score quantiles at the true training class proportions (Eq. 16)
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader
from torchvision import transforms

from calibration import assign, build_tie_pairs, tie_segmentation
from dataset import RelativeComparisonDataset
from eval_utils import OrdinalImageFolder, compute_metrics, fit_axis_thresholds
from model import create_model


def eval_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device)
    a = dict(ckpt.get("args", {}))
    model = create_model(backbone=a.get("backbone", "convnext_tiny"),
                         projection_dim=int(a.get("projection_dim", 128)),
                         pretrained=False).to(device)
    state = {k.replace("module.", "", 1): v for k, v in ckpt["model_state_dict"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded {checkpoint}  (missing={len(missing)} unexpected={len(unexpected)})")
    model.eval()
    return model, a, ckpt.get("clip_margins")


@torch.no_grad()
def extract_train(model, root, args, device):
    """Training-set severity scores (indexed by global id) and the comparison graph, whose tie
    pairs anchor the Tie-seg cut-offs."""
    ds = RelativeComparisonDataset(data_dir=root, transform=eval_transform(args.image_size),
                                   pairs_per_image=args.pairs_per_image)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    scores = np.zeros(len(ds.samples), dtype=np.float64)
    for images, ids in loader:
        _, s = model(images.to(device))
        scores[np.asarray(ids)] = s.cpu().numpy()
    return ds, scores, ds.get_gt_grades().numpy()


@torch.no_grad()
def extract_test(model, root, args, device):
    ds = OrdinalImageFolder(root, args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    scores, labels = [], []
    for images, y, _ in loader:
        _, s = model(images.to(device))
        scores.append(s.cpu().numpy())
        labels.extend(y.numpy().tolist())
    return np.concatenate(scores), np.asarray(labels, dtype=int)


def report(name, y_true, y_pred, score, args):
    m = compute_metrics(y_true, y_pred.astype(int), score, args)
    cm = np.asarray(m["confusion_matrix"], dtype=float)
    recall = [round(float(cm[c, c] / cm[c].sum()), 3) if cm[c].sum() > 0 else None
              for c in range(args.n_classes)]
    print(f"  {name:<26}: Acc {m['exact_accuracy_percent']:5.2f}%  QWK {m['qwk']:.4f}  "
          f"MAE {m['mae']:.4f}  F1 {m['macro_f1_percent']:5.2f}%  recall {recall}")
    return {"acc": m["exact_accuracy_percent"], "qwk": m["qwk"], "mae": m["mae"],
            "macro_f1": m["macro_f1_percent"], "per_class_recall": recall,
            "confusion_matrix": cm.astype(int).tolist()}  # rows=true, cols=pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--test-root", required=True)
    ap.add_argument("--output-dir", default="./eval")
    ap.add_argument("--n-classes", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--pairs-per-image", type=int, default=170,
                    help="Same value as training, to rebuild the comparison graph for Tie-seg.")
    ap.add_argument("--lam", type=float, default=1.0, help="Tie-cut penalty lambda_tie in Tie-seg.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-pairs", type=int, default=500_000)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    K = args.n_classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, margins = load_model(args.checkpoint, device)

    ds, tr_scores, tr_grades = extract_train(model, args.train_root, args, device)
    te_scores, te_true = extract_test(model, args.test_root, args, device)

    tie_pairs = build_tie_pairs(ds)
    th_tie = tie_segmentation(tr_scores, tie_pairs, K, lam=args.lam)

    g = GaussianMixture(K, n_init=10, random_state=args.seed).fit(tr_scores.reshape(-1, 1))
    est_prior = g.weights_[np.argsort(g.means_.flatten())]
    known_prior = np.bincount(tr_grades, minlength=K).astype(float)
    known_prior /= known_prior.sum()

    def thresh_pred(th):
        return np.clip(np.digitize(te_scores, th), 0, K - 1)

    print(f"\n[DOSE eval]  tie_pairs={len(tie_pairs)}  thresholds(tie)={np.round(th_tie, 3).tolist()}")
    results = {
        "tie_classifier": report("Tie-seg (label-free)", te_true,
                                 assign(te_scores, th_tie, K), te_scores, args),
        "score_thresh_estimated": report("Est-prior (label-free)", te_true,
                                         thresh_pred(fit_axis_thresholds(tr_scores, K, "prior", priors=est_prior)),
                                         te_scores, args),
        "score_thresh_gmm": report("GMM (label-free)", te_true,
                                   thresh_pred(fit_axis_thresholds(tr_scores, K, "gmm")), te_scores, args),
        "score_thresh_known": report("Known-prior (oracle)", te_true,
                                     thresh_pred(fit_axis_thresholds(tr_scores, K, "prior", priors=known_prior)),
                                     te_scores, args),
    }

    out = {"checkpoint": args.checkpoint,
           "n_tie_pairs": int(len(tie_pairs)),
           "thresholds_tie": np.round(th_tie, 4).tolist(),
           "clip_margins": margins,
           "estimated_prior": np.round(est_prior, 3).tolist(),
           "known_prior": np.round(known_prior, 3).tolist(),
           "results": results}
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone. Summary → {os.path.join(args.output_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
