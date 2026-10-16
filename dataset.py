import os
import re
from collections import Counter, defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class RelativeComparisonDataset(Dataset):
    """Training images arranged in grade folders (e.g. train/0, train/1, ...) plus a sparse
    pairwise comparison graph.

    The folder grades are used only to simulate the relative annotator: for an annotated pair
    (i, j) the label is r_ij = sign(g_i - g_j) in {+1, -1, 0} (Eq. 1). Images are compared with
    on average `pairs_per_image` random partners; unannotated pairs carry no supervision.
    """

    def __init__(self, data_dir, transform=None, pairs_per_image=10):
        self.data_dir = data_dir
        self.transform = transform
        self.pairs_per_image = pairs_per_image
        self._load_from_folders()

    def get_gt_grades(self) -> torch.Tensor:
        """Folder grades (used only for the known-prior oracle at evaluation)."""
        return torch.tensor(self.gt_grades, dtype=torch.long)

    def _load_from_folders(self):
        self.samples = []     # image paths
        self.img_ids = []     # global integer index of each image
        self.gt_grades = []   # folder grades

        def _extract_grade(folder_name):
            match = re.search(r"(\d+)", folder_name)
            return int(match.group(1)) if match else None

        folder_info = []
        for item in sorted(os.listdir(self.data_dir)):
            item_path = os.path.join(self.data_dir, item)
            if os.path.isdir(item_path):
                grade = _extract_grade(item)
                if grade is not None:
                    folder_info.append((grade, item, item_path))
        folder_info.sort(key=lambda x: x[0])

        print(f"[Dataset] Found {len(folder_info)} ordinal classes:")
        img_idx = 0
        for grade, folder_name, folder_path in folder_info:
            count = 0
            for img_name in sorted(os.listdir(folder_path)):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                    self.samples.append(os.path.join(folder_path, img_name))
                    self.gt_grades.append(grade)
                    self.img_ids.append(img_idx)
                    img_idx += 1
                    count += 1
            print(f"  Grade {grade} ({folder_name}): {count} images")

        n = len(self.samples)
        print(f"\n[Dataset] Total: {n} images")
        print(f"[Dataset] Grade distribution: {dict(sorted(Counter(self.gt_grades).items()))}")
        print(f"\n[Dataset] Building sparse pair graph (avg {self.pairs_per_image} pairs/image) …")
        self._build_connectivity_sparse(self.pairs_per_image)

    def _build_connectivity_sparse(self, pairs_per_image: int = 10):
        """self.pairs_dict[i] = list of images compared with image i (undirected)."""
        self.pairs_dict = defaultdict(set)
        n = len(self.samples)
        total = 0
        for i in range(n):
            k = int(np.random.poisson(pairs_per_image))
            k = min(k, n - 1)
            if k == 0:
                continue
            candidates = [j for j in range(n) if j != i]
            chosen = np.random.choice(candidates, size=k, replace=False)
            for j in chosen:
                if j not in self.pairs_dict[i]:
                    self.pairs_dict[i].add(j)
                    self.pairs_dict[j].add(i)
                    total += 1
        self.pairs_dict = {k: list(v) for k, v in self.pairs_dict.items()}

        max_possible = n * (n - 1) // 2
        coverage = 100 * total / max_possible
        print(f"[Dataset] Sparse graph: {total:,} edges | "
              f"coverage {coverage:.2f}% | avg {total / n:.1f} neighbours/image")

    def relative_label(self, global_i: int, global_j: int) -> int:
        """r_ij: +1 if image i is more severe than j, -1 if less severe, 0 if comparable."""
        li, lj = self.gt_grades[global_i], self.gt_grades[global_j]
        return 1 if li > lj else (-1 if li < lj else 0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.img_ids[idx]

    def get_pairs_for_batch(self, batch_ids):
        """Annotated pairs inside a batch.

        batch_ids : global image indices of the batch.
        Returns   : (pairs (P, 2) of local indices, relative labels (P,)) or (None, None).
        """
        id_to_local = {gid: local for local, gid in enumerate(batch_ids)}
        batch_set = set(batch_ids)
        pairs, labels, seen = [], [], set()

        for local_i, gid_i in enumerate(batch_ids):
            for gid_j in self.pairs_dict.get(gid_i, []):
                if gid_j not in batch_set:
                    continue
                local_j = id_to_local[gid_j]
                key = (min(local_i, local_j), max(local_i, local_j))
                if key in seen:
                    continue
                seen.add(key)

                rel = self.relative_label(gid_i, gid_j)
                if local_i < local_j:                  # emit the smaller local index first
                    pairs.append([local_i, local_j])
                    labels.append(float(rel))
                else:
                    pairs.append([local_j, local_i])
                    labels.append(float(-rel))

        if len(pairs) == 0:
            return None, None
        return torch.tensor(pairs, dtype=torch.long), torch.tensor(labels, dtype=torch.float32)
