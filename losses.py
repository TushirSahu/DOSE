import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(t):
    return torch.where(t > 20, t, torch.log(torch.exp(t) - 1))


class RelativeCLOCLoss(nn.Module):
    """Single-margin variant (ablation, --single_margin): one shared margin for every negative."""

    def __init__(self, device, margin_init=1.0, learnable_margin=True, temperature=0.07):
        super().__init__()
        self.device = device
        self.temperature = temperature
        init = _inverse_softplus(torch.tensor([margin_init], device=device))
        self.margin = nn.Parameter(init) if learnable_margin else init

    def forward(self, embeddings, pairs, relative_labels, rank_distances=None):
        embeddings = F.normalize(embeddings, dim=1)
        cos_sim = torch.matmul(embeddings, embeddings.t()) / self.temperature
        n_samples = embeddings.size(0)

        ordering_matrix = torch.zeros(n_samples, n_samples, device=self.device)
        known_mask = torch.zeros(n_samples, n_samples, dtype=torch.bool, device=self.device)
        for (i, j), rel in zip(pairs, relative_labels):
            known_mask[i, j] = True
            known_mask[j, i] = True
            if rel > 0:                       # i more severe than j
                ordering_matrix[i, j] = 1
                ordering_matrix[j, i] = -1
            elif rel < 0:                     # j more severe than i
                ordering_matrix[i, j] = -1
                ordering_matrix[j, i] = 1

        margin = F.softplus(self.margin)
        total_loss = 0
        count = 0
        for i in range(n_samples):
            known_rels = known_mask[i]
            positives = (ordering_matrix[i] == 0) & known_rels
            positives[i] = False
            higher_than_i = (ordering_matrix[i] < 0) & known_rels
            lower_than_i = (ordering_matrix[i] > 0) & known_rels
            if not positives.any() or (not higher_than_i.any() and not lower_than_i.any()):
                continue

            for pos_idx in positives.nonzero(as_tuple=True)[0]:
                pos_sim = cos_sim[i, pos_idx]
                for neg_idx in higher_than_i.nonzero(as_tuple=True)[0]:
                    neg_sim = cos_sim[i, neg_idx]
                    total_loss += F.relu(-pos_sim + neg_sim + margin)
                    count += 1
                for neg_idx in lower_than_i.nonzero(as_tuple=True)[0]:
                    neg_sim = cos_sim[i, neg_idx]
                    total_loss += F.relu(pos_sim - neg_sim + margin)
                    count += 1

        if count == 0:
            return embeddings.sum() * 0.0
        return total_loss / count


class RelativeCLOCLoss_MultiMargin(nn.Module):
    """Dominance-aware multi-margin loss L_RMM (Eq. 8).

    The margin for a negative k grows with its ordinal distance d_ik:
    Delta_ik = sum_{t=1}^{d_ik} m_t, with m_t = softplus(theta_t) initialised in [0.5, 2.0].
    """

    def __init__(self, device, n_rank_levels=4, learnable_margin=True, temperature=0.07):
        super().__init__()
        self.device = device
        self.temperature = temperature
        self.max_rank_distance = n_rank_levels - 1
        init = _inverse_softplus(torch.linspace(0.5, 2.0, n_rank_levels - 1).to(device))
        self.margins = nn.Parameter(init) if learnable_margin else init

    def forward(self, embeddings, pairs, relative_labels, rank_distances=None):
        embeddings = F.normalize(embeddings, dim=1)
        cos_sim = torch.matmul(embeddings, embeddings.t()) / self.temperature
        n_samples = embeddings.size(0)

        i_idx = pairs[:, 0].long()
        j_idx = pairs[:, 1].long()
        sign = relative_labels.sign()
        rdist = (rank_distances.float() if rank_distances is not None
                 else torch.ones_like(sign)).clamp(min=1.0)

        ordering = torch.zeros(n_samples, n_samples, device=self.device)
        known = torch.zeros(n_samples, n_samples, dtype=torch.bool, device=self.device)
        rank_distance_matrix = torch.ones(n_samples, n_samples, device=self.device)
        known[i_idx, j_idx] = True
        known[j_idx, i_idx] = True
        ordering[i_idx, j_idx] = sign       # +1 when i is more severe than j
        ordering[j_idx, i_idx] = -sign
        rank_distance_matrix[i_idx, j_idx] = rdist
        rank_distance_matrix[j_idx, i_idx] = rdist

        pos_margins = F.softplus(self.margins)
        # class_positions[k] = cumulative margin for a pair k ordinal steps apart.
        class_positions = torch.cumsum(
            torch.cat([torch.zeros(1, device=self.device), pos_margins]), dim=0)
        max_step = len(pos_margins)

        total_loss = embeddings.new_zeros(())
        count = 0
        for i in range(n_samples):
            known_i = known[i]
            positives = (ordering[i] == 0) & known_i
            positives[i] = False
            negatives = (ordering[i] != 0) & known_i
            if not positives.any() or not negatives.any():
                continue

            pos_sim = cos_sim[i, positives]                                   # (P,)
            neg_sim = cos_sim[i, negatives]                                   # (Q,)
            steps = rank_distance_matrix[i, negatives].long().clamp(1, max_step)
            margins = class_positions[steps]                                  # (Q,)
            # Triplet hinge over all (positive, negative) combinations: pos_sim >= neg_sim + margin.
            diff = margins[None, :] + neg_sim[None, :] - pos_sim[:, None]      # (P, Q)
            total_loss = total_loss + F.relu(diff).mean()
            count += 1

        if count == 0:
            return embeddings.sum() * 0.0
        return total_loss / count


def compute_dominance_scores(pairs: torch.Tensor, relative_labels: torch.Tensor,
                             n_samples: int) -> torch.Tensor:
    """Copeland dominance (#wins - #losses) / #games, in [-1, 1]. Ablation for --dominance copeland."""
    device = pairs.device
    score = torch.zeros(n_samples, device=device)
    degree = torch.zeros(n_samples, device=device)

    rel = relative_labels.sign()
    keep = rel != 0
    if not keep.any():
        return score

    i, j, r = pairs[keep, 0].long(), pairs[keep, 1].long(), rel[keep]
    score.index_add_(0, i, r)
    score.index_add_(0, j, -r)
    degree.index_add_(0, i, torch.ones_like(r))
    degree.index_add_(0, j, torch.ones_like(r))

    valid = degree > 0
    dominance = torch.zeros_like(score)
    dominance[valid] = score[valid] / degree[valid]
    return dominance


def transitive_dominance(pairs, relative_labels, n_samples, iters: int = 30):
    """Global dominance q in [0, 1] via Rank Centrality (Eqs. 4-5): the stationary distribution
    of a random walk that steps toward the images that beat the current one, rank-normalised."""
    device = pairs.device
    rel = relative_labels.sign()
    keep = rel != 0
    if not keep.any():
        return torch.zeros(n_samples, device=device)

    i = pairs[keep, 0].long()
    j = pairs[keep, 1].long()
    r = rel[keep]
    winner = torch.where(r > 0, i, j)   # more-severe item
    loser = torch.where(r > 0, j, i)

    # A[w, l] = number of times w beats l.
    A = torch.zeros(n_samples, n_samples, device=device)
    A.index_put_((winner, loser), torch.ones_like(r, dtype=A.dtype), accumulate=True)

    totals = A + A.t()                                  # games played per ordered pair
    seen = totals.sum(dim=1) > 0
    if seen.sum() < 2:
        return torch.zeros(n_samples, device=device)

    d_max = (totals > 0).sum(dim=1).float().max().clamp_min(1.0)

    # Transition P[i, j] = fraction of games i loses to j (walk toward winners).
    P = torch.zeros_like(A)
    mask = totals > 0
    P[mask] = A.t()[mask] / totals[mask]                # P[i, j] = A[j, i] / totals[i, j]
    P = P / d_max
    P = P + torch.diag(1.0 - P.sum(dim=1))              # self-loops keep it stochastic

    # Power iteration for the stationary distribution (more visits == stronger).
    pi = torch.full((n_samples,), 1.0 / n_samples, device=device)
    Pt = P.t()
    for _ in range(iters):
        pi = Pt @ pi
        pi = pi / pi.sum().clamp_min(1e-8)

    q = torch.zeros(n_samples, device=device)
    idx = seen.nonzero(as_tuple=True)[0]
    order = torch.argsort(pi[idx])                      # ascending strength
    ranks = torch.empty(order.numel(), device=device)
    ranks[order] = torch.arange(order.numel(), device=device, dtype=ranks.dtype)
    if order.numel() > 1:
        ranks = ranks / (order.numel() - 1)             # percentile in [0, 1]
    q[idx] = ranks
    return q


def pairwise_ranking_loss(scores, pairs, relative_labels, margin: float = 0.5, loss_type: str = "hinge"):
    """L_rank on the annotated non-tied pairs. 'hinge': max(0, m - r*delta) (MarginRank);
    'logistic': log(1 + exp(-r*delta)) (RankNet, Eq. 9)."""
    r = relative_labels.sign()
    keep = r != 0
    if int(keep.sum()) == 0:
        return scores.sum() * 0.0
    i = pairs[keep, 0].long()
    j = pairs[keep, 1].long()
    rr = r[keep]
    diff = scores[i] - scores[j]
    if loss_type == "logistic":
        return F.softplus(-rr * diff).mean()
    return F.relu(margin - rr * diff).mean()


def dominance_pseudo_pair_loss(scores, dominance, margin: float = 0.5, tau: float = 0.1):
    """L_pp (Eq. 6): hinge on every batch pair whose dominance gap |q_i - q_j| exceeds tau."""
    B = scores.size(0)
    valid = dominance > 0
    if int(valid.sum()) < 2:
        return scores.sum() * 0.0
    dq = dominance[:, None] - dominance[None, :]          # (B,B)
    ds = scores[:, None] - scores[None, :]
    triu = torch.triu(torch.ones(B, B, dtype=torch.bool, device=scores.device), diagonal=1)
    mask = triu & (dq.abs() > tau) & valid[:, None] & valid[None, :]
    if int(mask.sum()) == 0:
        return scores.sum() * 0.0
    r = torch.sign(dq[mask]).detach()
    return F.relu(margin - r * ds[mask]).mean()
