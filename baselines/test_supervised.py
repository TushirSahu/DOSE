"""
Re-evaluate a saved supervised checkpoint on a test set. The number is identical
to the `best` row train_supervised.py already printed; this exists only so a
frozen checkpoint can be scored again without retraining (reproducibility, or a
different test split).

    python baselines/test_supervised.py \
        --checkpoint ./baselines/sup/LIMUC__convnext_tiny__s1234/sup_best.pth \
        --test-root ./LIMUC_organised/test
"""
import argparse
import json

import torch

from train_supervised import build_backbone, evaluate, OrdinalImageFolder
from torch.utils.data import DataLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test-root", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    a = ckpt["args"]
    K = a["n_classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _ = build_backbone(a["backbone"], K, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    te = OrdinalImageFolder(args.test_root, a["image_size"])
    loader = DataLoader(te, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    m = evaluate(model, loader, device, K)
    print(f"{a['backbone']}  K={K}  n={len(te)}")
    print(f"  Acc {m['acc']:.2f}  QWK {m['qwk']:.4f}  MAE {m['mae']:.4f}  "
          f"F1 {m['macro_f1']:.2f}")
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
