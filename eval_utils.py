import os
import re
from typing import List, Tuple

import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


class OrdinalImageFolder(Dataset):
    """Test images in grade folders; returns (image, grade, path). Grades are used only for metrics."""

    def __init__(self, root_dir: str, image_size: int):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.samples = self._find_samples(root_dir)
        if not self.samples:
            raise RuntimeError(f"No labelled images found in {root_dir}")
        labels = np.array([x[1] for x in self.samples])
        dist = {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
        print(f"Loaded {len(self.samples)} images from {root_dir}")
        print(f"Label distribution: {dist}")

    @staticmethod
    def _label_from_path(path: str) -> int:
        parent = os.path.basename(os.path.dirname(path)).lower().strip()
        for pattern in [r"mayo[_\s-]*(\d+)", r"grade[_\s-]*(\d+)",
                        r"class[_\s-]*(\d+)", r"^(\d+)$"]:
            m = re.search(pattern, parent)
            if m:
                return int(m.group(1))
        return -1

    def _find_samples(self, root_dir: str) -> List[Tuple[str, int]]:
        samples = []
        for root, _, files in os.walk(root_dir):
            for name in files:
                if name.lower().endswith(IMAGE_EXTS):
                    path = os.path.join(root, name)
                    label = self._label_from_path(path)
                    if label >= 0:
                        samples.append((path, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), label, path


def fit_axis_thresholds(train_scores, n_classes, mode, priors=None):
    """K-1 cut-offs on the 1-D severity axis. No per-image labels are used.

    mode='prior' : empirical score quantiles at the cumulative class proportions (Eqs. 15-16).
    mode='gmm'   : midpoints between sorted 1-D Gaussian-mixture means (Eq. 14).
    """
    if mode == "prior":
        priors = np.asarray(priors, dtype=float)
        qs = np.cumsum(priors / priors.sum())[:-1]
        return np.quantile(train_scores, qs)
    if mode == "gmm":
        from sklearn.mixture import GaussianMixture
        g = GaussianMixture(n_classes, n_init=10, random_state=42).fit(train_scores.reshape(-1, 1))
        centers = np.sort(g.means_.flatten())
        return (centers[:-1] + centers[1:]) / 2.0
    raise ValueError(f"Unknown threshold mode: {mode}")


def pair_acc(y_true, score, max_pairs, seed, allowed_pair=None):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    total = n * (n - 1) // 2
    if total <= max_pairs:
        i, j = np.triu_indices(n, k=1)
    else:
        i = rng.integers(0, n, max_pairs)
        j = rng.integers(0, n, max_pairs)
        keep = i != j
        i, j = i[keep], j[keep]

    keep = y_true[i] != y_true[j]
    if keep.sum() == 0:
        return None, 0

    i, j = i[keep], j[keep]
    correct = np.sign(y_true[i] - y_true[j]) == np.sign(score[i] - score[j])
    return float(100 * correct.mean()), int(len(i))


def class_pair_metrics(y_true, score, n_classes, max_pairs, seed):
    rows, adjacent = {}, []
    for a in range(n_classes):
        for b in range(a + 1, n_classes):
            acc, count = pair_acc(y_true, score, max_pairs,
                                  seed + 17 * a + b, allowed_pair=(a, b))
            rows[f"{a}-{b}"] = {"accuracy_percent": acc, "count": count}
            if b == a + 1 and acc is not None:
                adjacent.append(acc)
    return rows, (float(np.mean(adjacent)) if adjacent else None)


def compute_metrics(y_true, y_pred, score, args):
    pair_table, neighbor_macro = class_pair_metrics(
        y_true, score, args.n_classes, args.max_pairs, args.seed
    )
    overall_pair, _ = pair_acc(y_true, score, args.max_pairs, args.seed)
    return {
        "n_samples":                            int(len(y_true)),
        "exact_accuracy_percent":               float(100 * accuracy_score(y_true, y_pred)),
        "macro_f1_percent":                     float(100 * f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mae":                                  float(mean_absolute_error(y_true, y_pred)),
        "qwk":                                  float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "overall_pair_accuracy_percent":        overall_pair,
        "neighbor_pair_macro_accuracy_percent": neighbor_macro,
        "class_pair_ranking":                   pair_table,
        "confusion_matrix":                     confusion_matrix(y_true, y_pred, labels=list(range(args.n_classes))).tolist(),
    }
