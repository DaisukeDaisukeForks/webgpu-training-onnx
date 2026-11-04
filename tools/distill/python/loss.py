"""Optional recovery loss for a pretrained, structurally pruned model."""
import torch
import torch.nn.functional as F
from compress import IDS

EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (15, 19), (19, 17), (19, 18), (17, 18), (16, 22), (22, 20), (22, 21), (20, 21)]


def teacher_targets(raw):
    selected = [v[:, IDS].detach() for v in raw]
    score = torch.minimum(selected[0].amax(-1), selected[1].amax(-1))
    mask = torch.where(score > .25, score.clamp(max=1), torch.zeros_like(score))
    return [(v * 5).softmax(-1) for v in selected], mask


def distillation_loss(logits, targets):
    axes, mask = targets
    loss = logits[0].new_zeros(())
    probabilities = [(v * 5).softmax(-1) for v in logits]
    for pred, target in zip(logits, axes):
        kl = (target * (target.clamp_min(1e-8).log() - (pred * 5).log_softmax(-1))).sum(-1)
        for lo, hi, scale in [(0, 17, 1), (17, 23, 2), (23, 30, 1), (30, 37, 1)]:
            w = mask[:, lo:hi]
            loss = loss + (kl[:, lo:hi] * w).sum() / w.sum().clamp_min(1) * (4 * scale / 15)
    def expectation(values):
        return torch.stack([(p * (torch.arange(p.shape[-1], device=p.device) / p.shape[-1])).sum(-1) for p in values], -1)
    positions, predicted = expectation(axes), expectation(probabilities)
    starts, ends = zip(*EDGES)
    weights = torch.minimum(mask[:, starts], mask[:, ends])
    delta = (predicted[:, ends] - predicted[:, starts]) - (positions[:, ends] - positions[:, starts])
    error = F.smooth_l1_loss(delta * delta.new_tensor([1, 1, 2]), torch.zeros_like(delta), beta=.02, reduction='none').mean(-1)
    return loss + 2 * (error * weights).sum() / weights.sum().clamp_min(1)
