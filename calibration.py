import numpy as np


def build_tie_pairs(dataset):
    """Tie pairs (annotated as comparable, r_ij == 0) of the training comparison graph.
    Returns an (M, 2) int array of global image indices."""
    ties = []
    for i, neigh in dataset.pairs_dict.items():
        for j in neigh:
            if j <= i:
                continue
            if dataset.relative_label(i, j) == 0:
                ties.append((i, j))
    return np.asarray(ties, dtype=np.int64).reshape(-1, 2)

def _boundary_cut_weights(order_pos, tie_pairs, n):
    """w[g] = number of tie pairs straddling gap g (between sorted index g-1 and g),
    g = 1..n-1. A single boundary at gap g cuts pair (a,b) iff pos_a < g <= pos_b."""
    w = np.zeros(n + 1, dtype=np.float64)        # diff array over gaps 1..n-1
    if tie_pairs.size == 0:
        return w[:n]
    pa = order_pos[tie_pairs[:, 0]]
    pb = order_pos[tie_pairs[:, 1]]
    lo = np.minimum(pa, pb)
    hi = np.maximum(pa, pb)
    # straddled gaps are (lo, hi]  ->  g in [lo+1, hi]
    np.add.at(w, lo + 1, 1.0)
    np.add.at(w, hi + 1, -1.0)
    w = np.cumsum(w)
    return w[:n]                                  # w[g] valid for g in 1..n-1

def tie_segmentation(scores, tie_pairs, n_classes, lam=1.0, min_frac=0.01):
    """Place K-1 thresholds on the 1-D severity axis by tie-constrained ordinal segmentation.

    scores     : (N,) severity scores (global-index order).
    tie_pairs  : (M, 2) global-index pairs that are graded equal (relative_label == 0).
    n_classes  : K (number of ordinal grades; a problem definition, not a label).
    lam        : weight on the tie-cut penalty (0 -> pure 1-D k-means natural breaks).
    min_frac   : minimum bin size as a fraction of N (>=2 points). Forbids the degenerate
                 collapse where a strong tie penalty parks a boundary past the data extreme
                 and wastes a bin on ~0 points.

    Returns thresholds : (K-1,) strictly increasing cut points on the score axis.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = scores.size
    K = int(n_classes)
    if K <= 1:
        return np.asarray([], dtype=np.float64)
    if n <= K:                                    # degenerate: cut between every point
        srt = np.sort(scores)
        return (srt[:-1] + srt[1:])[: K - 1] / 2.0 if n > 1 else np.zeros(K - 1)

    order = np.argsort(scores, kind="mergesort")  # global idx -> sorted position
    s_sorted = scores[order]
    order_pos = np.empty(n, dtype=np.int64)
    order_pos[order] = np.arange(n)
    w = _boundary_cut_weights(order_pos, np.asarray(tie_pairs, dtype=np.int64).reshape(-1, 2), n)

    # Prefix sums for O(1) within-segment SSE on the sorted axis.
    p1 = np.concatenate([[0.0], np.cumsum(s_sorted)])
    p2 = np.concatenate([[0.0], np.cumsum(s_sorted ** 2)])

    def sse(a, b):                                # SSE of sorted points [a, b)
        m = b - a
        if m <= 0:
            return 0.0
        seg1 = p1[b] - p1[a]
        seg2 = p2[b] - p2[a]
        return seg2 - seg1 * seg1 / m

    ms = max(2, int(min_frac * n))                # min points per segment
    if ms * K > n:                                # too tight for this N -> relax
        ms = max(1, n // K)

    INF = np.inf
    # dp[k][i] = min cost to split first i sorted points into k segments (each >= ms points).
    dp = np.full((K + 1, n + 1), INF)
    back = np.zeros((K + 1, n + 1), dtype=np.int64)
    dp[0][0] = 0.0
    for k in range(1, K + 1):
        # segment k spans [j, i); boundary closing segment k-1 is at gap j (cost lam*w[j]).
        for i in range(k * ms, n + 1):
            best, best_j = INF, -1
            for j in range((k - 1) * ms, i - ms + 1):
                prev = dp[k - 1][j]
                if prev == INF:
                    continue
                c = prev + sse(j, i) + (lam * w[j] if j > 0 else 0.0)
                if c < best:
                    best, best_j = c, j
            dp[k][i] = best
            back[k][i] = best_j

    # Backtrack the K-1 boundaries.
    cuts = []
    i = n
    for k in range(K, 0, -1):
        j = back[k][i]
        if k > 1:
            cuts.append(j)
        i = j
    cuts = sorted(cuts)
    # Threshold = midpoint between the two sorted scores straddling each boundary gap.
    thresholds = np.asarray([0.5 * (s_sorted[g - 1] + s_sorted[g]) for g in cuts], dtype=np.float64)
    return thresholds

def assign(scores, thresholds, n_classes):
    """Hard ordinal class = number of thresholds the score exceeds."""
    return np.clip(np.digitize(np.asarray(scores).reshape(-1), thresholds), 0, n_classes - 1)
