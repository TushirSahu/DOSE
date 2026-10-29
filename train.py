"""DOSE training: learn an ordinal embedding z and a scalar severity score s from sparse
pairwise comparisons only (no absolute grades as targets).

    L = lambda_clip * L_RMM + lambda_rank * L_rank + lambda_pp * L_pp        (Eq. 10)
"""
import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import RelativeComparisonDataset
from losses import (
    RelativeCLOCLoss,
    RelativeCLOCLoss_MultiMargin,
    compute_dominance_scores,
    dominance_pseudo_pair_loss,
    pairwise_ranking_loss,
    transitive_dominance,
)
from model import DOSEModel


def collate(batch):
    images, img_ids = zip(*batch)
    return torch.stack(images), list(img_ids)


def train_transform(image_size, aug):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if aug == "strong":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.1, 0.02),
            transforms.ToTensor(), norm,
            transforms.RandomErasing(p=0.2),
        ])
    return transforms.Compose([                      # light (paper setting)
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(7),
        transforms.ColorJitter(0.08, 0.08, 0.05, 0.02),
        transforms.ToTensor(), norm,
    ])


def build_loader(args, rank, world_size, distributed):
    ds = RelativeComparisonDataset(
        data_dir=args.data_dir, transform=train_transform(args.image_size, args.aug),
        pairs_per_image=args.pairs_per_image,
    )
    sampler = (torch.utils.data.distributed.DistributedSampler(
        ds, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=(sampler is None),
                        sampler=sampler, num_workers=args.num_workers, collate_fn=collate,
                        pin_memory=torch.cuda.is_available(), drop_last=True)
    return ds, loader, sampler


def compute_global_dominance(dataset, mode="transitive", transitive_cap=None):
    """Dominance score q_i in [0, 1] for every training image, from the whole comparison graph."""
    n = len(dataset.samples)
    winners, losers = [], []
    for i, neigh in dataset.pairs_dict.items():
        gi = dataset.gt_grades[i]
        for j in neigh:
            if j <= i:
                continue
            gj = dataset.gt_grades[j]
            if gi == gj:
                continue
            (w, l) = (i, j) if gi > gj else (j, i)
            winners.append(w); losers.append(l)

    if not winners:
        return torch.zeros(n)
    pairs = torch.tensor([winners, losers], dtype=torch.long).t()
    rel = torch.ones(pairs.size(0))
    if mode == "transitive" and (transitive_cap is None or n <= transitive_cap):
        q = transitive_dominance(pairs, rel, n)
        print(f"[dominance] global transitive (Rank-Centrality) over {n} nodes, {len(winners)} edges")
    else:
        q = compute_dominance_scores(pairs, rel, n)               # Copeland (#wins-#losses)
        q = (q - q.min()) / (q.max() - q.min() + 1e-8)
        why = "copeland" if mode == "copeland" else f"fallback N>{transitive_cap}"
        print(f"[dominance] {why} over {n} nodes, {len(winners)} edges")
    return q


def dominance_gap_edges(dataset, q, n_classes):
    """Empirical-quantile edges that bucket a dominance gap |q_i - q_j| into an ordinal distance
    level in {1..K-1} (Eq. 7). Quantiles are taken over the observed non-tied pairs, so levels
    stay balanced under class imbalance. Returns (K-2,) edges (empty for K<=2)."""
    K = int(n_classes)
    q = q.numpy() if torch.is_tensor(q) else np.asarray(q)
    if K <= 2:
        return torch.empty(0)
    gaps = []
    for i, neigh in dataset.pairs_dict.items():
        for j in neigh:
            if j <= i or dataset.gt_grades[i] == dataset.gt_grades[j]:   # non-tied pairs only
                continue
            gaps.append(abs(float(q[i]) - float(q[j])))
    if not gaps:
        return torch.empty(0)
    edges = np.quantile(np.asarray(gaps), np.linspace(0.0, 1.0, K)[1:-1])   # K-2 inner quantiles
    return torch.tensor(edges, dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, device, epoch, args, criterion_clip,
                    q_global, gap_edges, rank):
    model.train()
    tot = dict(loss=0.0, clip=0.0, rank=0.0, pp=0.0)
    n_batches = 0
    q_global = q_global.to(device)
    gap_edges = gap_edges.to(device)
    it = tqdm(loader, desc=f"Epoch {epoch}") if rank == 0 else loader
    pair_rng = np.random.default_rng(args.seed + 1000 * epoch + rank)   # disjoint pairing rounds
    margin_params = [p for p in criterion_clip.parameters() if p.requires_grad]
    sync_margins = bool(margin_params) and dist.is_available() and dist.is_initialized()

    for images, img_ids in it:
        images = images.to(device, non_blocking=True)
        pairs, rel = loader.dataset.get_pairs_for_batch(img_ids, rounds=args.pair_rounds, rng=pair_rng)
        if pairs is None or len(pairs) == 0:
            continue
        pairs = pairs.to(device)
        rel = rel.to(device).float()
        gid = torch.as_tensor([int(g) for g in img_ids], device=device)

        z, s = model(images)

        # L_RMM on z: the ordinal distance of a pair is its dominance gap |q_i - q_j|
        # bucketed into {1..K-1}.
        gap = (q_global[gid[pairs[:, 0]]] - q_global[gid[pairs[:, 1]]]).abs()
        steps = (torch.bucketize(gap, gap_edges).float() + 1.0 if gap_edges.numel() > 0 else torch.ones_like(gap))
        l_clip = criterion_clip(z, pairs, rel, rank_distances=steps)

        # L_rank on the severity head over the annotated ordered pairs.
        l_rank = pairwise_ranking_loss(s, pairs, rel, margin=args.rank_margin, loss_type=args.rank_loss)

        # L_pp: dominance pseudo-pairs with a confident gap |q_i - q_j| > pp_tau.
        l_pp = s.new_zeros(())
        if args.use_dom_pairs:
            l_pp = dominance_pseudo_pair_loss(s, q_global[gid], margin=args.rank_margin, tau=args.pp_tau)

        loss = args.lambda_clip * l_clip + args.lambda_rank * l_rank + args.lambda_pp * l_pp
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if sync_margins:
            # The margins live in the loss, outside DDP, so average their gradients across GPUs
            # to keep one shared set of margins.
            world = dist.get_world_size()
            for p in margin_params:
                if p.grad is not None:
                    dist.all_reduce(p.grad)
                    p.grad /= world
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        for k, v in (("loss", loss), ("clip", l_clip), ("rank", l_rank), ("pp", l_pp)):
            tot[k] += float(v.detach())
        n_batches += 1
        if rank == 0 and hasattr(it, "set_postfix"):
            it.set_postfix(loss=f"{loss.item():.3f}")

    n = max(n_batches, 1)
    if rank == 0:
        print(f"Epoch {epoch}: loss={tot['loss']/n:.4f} (clip={tot['clip']/n:.4f} "
              f"rank={tot['rank']/n:.4f} pp={tot['pp']/n:.4f})")
    return tot["loss"] / n


def save_ckpt(path, model, optimizer, epoch, args, criterion=None):
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    ckpt = {"epoch": epoch, "model_state_dict": state,
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args)}
    if criterion is not None and hasattr(criterion, "margins"):
        ckpt["clip_margins"] = F.softplus(criterion.margins.detach()).cpu().numpy().tolist()
    torch.save(ckpt, path)


def run(rank, world_size, args, distributed):
    if distributed:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", args.ddp_port)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, loader, sampler = build_loader(args, rank, world_size, distributed)

    q_global = compute_global_dominance(dataset, mode=args.dominance, transitive_cap=args.transitive_cap)
    gap_edges = dominance_gap_edges(dataset, q_global, args.n_classes)
    if rank == 0:
        print(f"[dominance-margin] gap-bucket edges = {np.round(gap_edges.numpy(), 4).tolist()}")

    model = DOSEModel(backbone=args.backbone, projection_dim=args.projection_dim,
                      pretrained=args.pretrained).to(device)
    if distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    if args.multi_margin:
        criterion_clip = RelativeCLOCLoss_MultiMargin(device, n_rank_levels=args.n_classes,
                                                      learnable_margin=not args.fixed_margins)
    else:
        criterion_clip = RelativeCLOCLoss(device, learnable_margin=not args.fixed_margins)

    # Step margins m_t = softplus(theta_t) are learned jointly with the network (no weight decay).
    param_groups = [{"params": model.parameters()}]
    margin_params = list(criterion_clip.parameters())
    if margin_params:
        param_groups.append({"params": margin_params, "weight_decay": 0.0})
    optimizer = optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
                 if args.cosine_lr else None)

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        train_one_epoch(model, loader, optimizer, device, epoch, args, criterion_clip,
                        q_global, gap_edges, rank)
        if scheduler is not None:
            scheduler.step()
        if rank == 0 and (epoch + 1) % args.save_freq == 0:
            save_ckpt(os.path.join(args.output_dir, f"dose_epoch_{epoch:03d}.pth"),
                      model, optimizer, epoch, args, criterion_clip)

    if rank == 0:
        save_ckpt(os.path.join(args.output_dir, "dose_final.pth"),
                  model, optimizer, args.epochs - 1, args, criterion_clip)
        print(f"Done. Checkpoint → {os.path.join(args.output_dir, 'dose_final.pth')}")
    if distributed:
        dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser(description="DOSE: ordinal severity from sparse relative comparisons")
    p.add_argument("--data_dir", required=True, help="Training root with one folder per grade.")
    p.add_argument("--output_dir", default="./checkpoints")
    p.add_argument("--backbone", default="convnext_tiny")
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--projection_dim", type=int, default=128)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--aug", choices=["light", "strong"], default="light",
                   help="light: resize/flip/rotation/mild colour jitter (paper). "
                        "strong: random-resized-crop + colour jitter + random erasing.")
    p.add_argument("--n_classes", type=int, default=4)

    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=32, help="Per-GPU batch size.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--cosine_lr", action="store_true", default=True)
    p.add_argument("--no_cosine_lr", dest="cosine_lr", action="store_false")
    p.add_argument("--save_freq", type=int, default=25)
    p.add_argument("--pairs_per_image", type=int, default=170,
                   help="Average number of annotated partners per image (sets the pair budget).")

    p.add_argument("--rank_margin", type=float, default=0.5, help="Margin m of the hinge losses.")
    p.add_argument("--rank_loss", choices=["hinge", "logistic"], default="hinge",
                   help="Severity-head ranking loss: 'hinge' (MarginRank) or 'logistic' (RankNet).")
    p.add_argument("--lambda_clip", type=float, default=1.0, help="Weight on L_RMM.")
    p.add_argument("--lambda_rank", type=float, default=0.5, help="Weight on L_rank.")
    p.add_argument("--lambda_pp", type=float, default=0.5, help="Weight on L_pp.")
    p.add_argument("--use_dom_pairs", action="store_true", default=True,
                   help="Add dominance pseudo-pairs L_pp (on by default).")
    p.add_argument("--no_dom_pairs", dest="use_dom_pairs", action="store_false")
    p.add_argument("--pp_tau", type=float, default=0.1, help="Min dominance gap to form a pseudo-pair.")

    p.add_argument("--dominance", choices=["transitive", "copeland"], default="transitive",
                   help="Dominance source: Rank Centrality (proposed) or 1-hop Copeland (ablation).")
    p.add_argument("--transitive_cap", type=int, default=None,
                   help="Fall back to Copeland when N exceeds this (default: never).")

    p.add_argument("--distributed", action="store_true", default=True)
    p.add_argument("--no-distributed", dest="distributed", action="store_false")
    p.add_argument("--num_gpus", type=int, default=4)
    p.add_argument("--ddp_port", default="12366")
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--multi_margin", action="store_true", default=True)
    p.add_argument("--fixed_margins", action="store_true",
                   help="Keep the step margins at their initial values instead of learning them.")
    p.add_argument("--pair_rounds", type=int, default=8,
                   help="Pairwise-encoding rounds per batch; each round partitions the batch into "
                        "disjoint pairs. 0 = use every annotated pair in the batch.")
    p.add_argument("--single_margin", dest="multi_margin", action="store_false",
                   help="Ablation: one shared margin instead of the cumulative multi-margin.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.distributed and torch.cuda.device_count() > 1:
        world = min(args.num_gpus, torch.cuda.device_count())
        print(f"DDP over {world} GPUs")
        mp.spawn(run, args=(world, args, True), nprocs=world, join=True)
    else:
        run(0, 1, args, False)


if __name__ == "__main__":
    main()
