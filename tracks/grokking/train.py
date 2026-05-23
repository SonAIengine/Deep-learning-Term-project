"""Grokking baseline train loop. Full-batch AdamW, ~50K epochs.

Usage:
  python -m tracks.grokking.train --smoke         # 100 epochs, no ckpt
  python -m tracks.grokking.train                 # full 50K run
  python -m tracks.grokking.train --epochs 5000   # custom epoch count
"""
import argparse
import os
import random
import time

import numpy as np
import torch

from shared.config import RESULTS_DIR, SEED
from tracks.grokking import config as C
from tracks.grokking.data import load_modular
from tracks.grokking.eval import eval_split, train_loss
from tracks.grokking.model import build_model


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.NUM_EPOCHS)
    ap.add_argument("--smoke", action="store_true",
                    help="100 epochs, no checkpoints; pipe sanity only")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    epochs = 100 if args.smoke else args.epochs
    out_dir = args.out_dir or os.path.join(
        RESULTS_DIR, "grokking",
        ("smoke_" if args.smoke else "run_") + time.strftime("%Y%m%d_%H%M%S"))
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    if not args.smoke:
        os.makedirs(ckpt_dir, exist_ok=True)

    seed_all(SEED)
    (xtr, ytr), (xte, yte) = load_modular()
    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR,
                            betas=C.BETAS, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda step: min((step + 1) / C.WARMUP_STEPS, 1.0))

    log = {"epoch": [], "train_loss": [], "test_loss": [],
           "train_acc": [], "test_acc": []}

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = train_loss(model, xtr, ytr)
        loss.backward()
        opt.step()
        sched.step()

        if ep % C.LOG_EVERY == 0 or ep == epochs - 1:
            model.eval()
            tr_l, tr_a = eval_split(model, xtr, ytr)
            te_l, te_a = eval_split(model, xte, yte)
            log["epoch"].append(ep)
            log["train_loss"].append(tr_l)
            log["test_loss"].append(te_l)
            log["train_acc"].append(tr_a)
            log["test_acc"].append(te_a)
            print(f"ep {ep:6d}  train {tr_l:.4f}/{tr_a:.3f}  "
                  f"test {te_l:.4f}/{te_a:.3f}  "
                  f"elapsed {time.time()-t0:.1f}s", flush=True)

        if not args.smoke and (ep % C.CKPT_EVERY == 0 or ep == epochs - 1):
            torch.save({"epoch": ep, "model": model.state_dict()},
                       os.path.join(ckpt_dir, f"ckpt_{ep:06d}.pt"))

    np.savez(os.path.join(out_dir, "loss_curve.npz"),
             **{k: np.array(v) for k, v in log.items()})
    print(f"[done] out_dir={out_dir}  wall={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
