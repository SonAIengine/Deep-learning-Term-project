"""Loss + accuracy helpers (eval at the '=' position only)."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def eval_split(model, x, y):
    logits = model(x)[:, -1, :]
    loss = F.cross_entropy(logits, y).item()
    acc = (logits.argmax(-1) == y).float().mean().item()
    return loss, acc


def train_loss(model, x, y):
    logits = model(x)[:, -1, :]
    return F.cross_entropy(logits, y)
