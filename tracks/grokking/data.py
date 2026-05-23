"""Load frozen modular_{train,test}.pt into device tensors."""
import os
import torch

from shared.config import DATASETS_DIR, DEVICE


def load_modular():
    """Returns ((train_x, train_y), (test_x, test_y)) on DEVICE."""
    tr = torch.load(os.path.join(DATASETS_DIR, "modular_train.pt"))
    te = torch.load(os.path.join(DATASETS_DIR, "modular_test.pt"))
    return ((tr["x"].to(DEVICE), tr["y"].to(DEVICE)),
            (te["x"].to(DEVICE), te["y"].to(DEVICE)))
