"""Modular arithmetic dataset for Step 1~2 (grokking).

Generates all p² (a, b) pairs and splits into train/test.
Token format: [a, b, '='] with vocab size p + 1 (numbers + '=' token at index p).
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.config import MODULAR_P, MODULAR_TRAIN_FRAC, SEED, DATASETS_DIR


def make_modular_data(p: int = MODULAR_P,
                      train_frac: float = MODULAR_TRAIN_FRAC,
                      seed: int = SEED):
    """Returns ((train_x, train_y), (test_x, test_y)).

    train_x shape: [N, 3] — tokens [a, b, '=' (index p)]
    train_y shape: [N]    — values in [0, p)
    """
    pairs = torch.tensor([(a, b) for a in range(p) for b in range(p)])
    labels = (pairs[:, 0] + pairs[:, 1]) % p
    eq = torch.full((pairs.size(0), 1), p)
    inputs = torch.cat([pairs, eq], dim=1)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(inputs.size(0), generator=g)
    n_train = int(train_frac * inputs.size(0))
    tr, te = perm[:n_train], perm[n_train:]
    return (inputs[tr], labels[tr]), (inputs[te], labels[te])


def save():
    os.makedirs(DATASETS_DIR, exist_ok=True)
    (xtr, ytr), (xte, yte) = make_modular_data()
    torch.save({"x": xtr, "y": ytr},
               os.path.join(DATASETS_DIR, "modular_train.pt"))
    torch.save({"x": xte, "y": yte},
               os.path.join(DATASETS_DIR, "modular_test.pt"))
    print(f"Saved: train={xtr.shape[0]}, test={xte.shape[0]}, "
          f"vocab={MODULAR_P + 1}")


if __name__ == "__main__":
    save()
