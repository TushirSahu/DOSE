"""
Fully supervised upper bound (Table 1, "Cross-entropy"): an ImageNet-initialised backbone with
one linear layer, trained with softmax cross-entropy on all absolute grades. Unlike DOSE it uses
every absolute label, so it is a reference ceiling rather than a competitor.

    fine-tune (default) : train the whole backbone end-to-end.
    --freeze            : linear probe; only the linear layer is trained.

Metrics (Acc / QWK / MAE / macro-F1) match evaluate.py, and the test set is loaded with the same
OrdinalImageFolder, so the grade <-> index mapping is identical.

Example (from the repository root):
    python baselines/train_supervised.py \
        --train-root ./LIMUC_organised/train --test-root ./LIMUC_organised/test \
        --backbone convnext_tiny --n-classes 4 --epochs 50 \
        --output-dir ./baselines/sup/LIMUC__convnext_tiny__s1234
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix

# Same test loader as evaluate.py, so the grade <-> index mapping matches the DOSE tables.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_utils import OrdinalImageFolder  # noqa: E402

NORM = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def build_backbone(name, n_classes, pretrained=True):
    """Return (model, feature_dim). Final classifier replaced by Linear(feat, K)."""
    w = "IMAGENET1K_V1" if pretrained else None
    if name == "convnext_tiny":
        m = models.convnext_tiny(weights=w)
        feat = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(feat, n_classes)
        head = [m.classifier[2]]
    elif name == "resnet50":
        m = models.resnet50(weights=w)
        feat = m.fc.in_features
        m.fc = nn.Linear(feat, n_classes)
        head = [m.fc]
    elif name == "densenet121":
        m = models.densenet121(weights=w)
        feat = m.classifier.in_features
        m.classifier = nn.Linear(feat, n_classes)
        head = [m.classifier]
    else:
        raise ValueError(f"unknown backbone {name}")
    return m, head


def loaders(args):
    tr_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(), transforms.Normalize(*NORM),
    ])
    tr = ImageFolder(args.train_root, transform=tr_tf)
    te = OrdinalImageFolder(args.test_root, args.image_size)   # same loader as evaluate.py

    # Guard the label mapping: ImageFolder sorts class dirs as strings. For grade
    # folders "0".."9" that already equals numeric order, but assert it so a
    # mislabelled bound can never slip into the table.
    grades = sorted(tr.class_to_idx, key=lambda c: tr.class_to_idx[c])
    assert grades == sorted(grades, key=int), \
        f"train class order {grades} is not grade order; rename folders 0..K-1"
    assert len(tr.classes) == args.n_classes, \
        f"found {len(tr.classes)} train classes, expected {args.n_classes}"

    dl = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    return (DataLoader(tr, shuffle=True, drop_last=True, **dl),
            DataLoader(te, shuffle=False, **dl))


@torch.no_grad()
def evaluate(model, loader, device, K):
    model.eval()
    y_true, y_pred = [], []
    for images, y, *_ in loader:                 # OrdinalImageFolder yields (img, y, path)
        logits = model(images.to(device))
        y_pred.extend(logits.argmax(1).cpu().tolist())
        y_true.extend(y.tolist())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(K)))
    return {
        "acc": 100.0 * float((y_true == y_pred).mean()),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic",
                                       labels=list(range(K)))),
        "mae": float(np.abs(y_true - y_pred).mean()),
        "macro_f1": 100.0 * float(f1_score(y_true, y_pred, average="macro",
                                           labels=list(range(K)), zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--test-root", required=True)
    ap.add_argument("--backbone", default="convnext_tiny",
                    choices=["convnext_tiny", "resnet50", "densenet121"])
    ap.add_argument("--n-classes", type=int, required=True)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None,
                    help="default 1e-3 (linear probe) or 3e-4 (fine-tune)")
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--freeze", action="store_true",
                    help="linear probe: freeze backbone, train head only")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="./sup_eval")
    ap.add_argument("--limit-batches", type=int, default=0,
                    help="smoke test: stop each epoch after N batches")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K = args.n_classes

    model, head = build_backbone(args.backbone, K, pretrained=True)
    if args.freeze:
        for p in model.parameters():
            p.requires_grad_(False)
        for layer in head:
            for p in layer.parameters():
                p.requires_grad_(True)
    model.to(device)

    tr_loader, te_loader = loaders(args)
    params = [p for p in model.parameters() if p.requires_grad]
    lr = args.lr if args.lr is not None else (1e-3 if args.freeze else 3e-4)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    mode = "linear-probe (frozen)" if args.freeze else "fine-tune (full)"
    print(f"[supervised upper bound] {args.backbone}  {mode}  K={K}  lr={lr}  "
          f"train={len(tr_loader.dataset)}  test={len(te_loader.dataset)}")

    best = None
    for ep in range(args.epochs):
        model.train()
        # In probe mode the frozen backbone still has BN/dropout; eval() it so the
        # running stats and features are fixed, only the head learns.
        if args.freeze:
            model.eval()
            for layer in head:
                layer.train()
        run = 0.0
        for i, (images, y) in enumerate(tr_loader):
            if args.limit_batches and i >= args.limit_batches:
                break
            images, y = images.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(images), y)
            loss.backward(); opt.step()
            run += loss.item()
        sched.step()
        m = evaluate(model, te_loader, device, K)
        print(f"  ep {ep + 1:3d}/{args.epochs}  loss {run / max(i + 1, 1):.3f}  "
              f"Acc {m['acc']:5.2f}  QWK {m['qwk']:.4f}  MAE {m['mae']:.4f}  "
              f"F1 {m['macro_f1']:5.2f}")
        if best is None or m["acc"] > best["acc"]:
            best = {**m, "epoch": ep + 1}
            torch.save({"model_state_dict": model.state_dict(), "args": vars(args),
                        "best": best}, os.path.join(args.output_dir, "sup_best.pth"))

    out = {"backbone": args.backbone, "mode": mode, "n_classes": K,
           "seed": args.seed, "lr": lr, "best": best}
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nBEST  Acc {best['acc']:.2f}  QWK {best['qwk']:.4f}  MAE {best['mae']:.4f}  "
          f"F1 {best['macro_f1']:.2f}  (ep {best['epoch']})")
    print(f"Summary -> {os.path.join(args.output_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
