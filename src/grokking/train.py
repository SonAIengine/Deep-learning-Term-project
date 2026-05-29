"""Grokking training — modular addition, Nanda et al. 2023 setup.

Usage: python -m src.grokking.train [--steps N] [--no-wandb]

Logs train/test loss+acc to stdout, results/grokking_log.jsonl, and (optionally)
wandb. Checkpoints saved to results/grokking_ckpt_step{N}.pt at log boundaries
and final step.
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shared.config import MODULAR_P, SEED, DATASETS_DIR, RESULTS_DIR
from src.grokking.model import GrokkingTransformer


def load_data(device):
    tr = torch.load(os.path.join(DATASETS_DIR, "modular_train.pt"), weights_only=True)
    te = torch.load(os.path.join(DATASETS_DIR, "modular_test.pt"), weights_only=True)
    return (tr["x"].to(device), tr["y"].to(device),
            te["x"].to(device), te["y"].to(device))


@torch.no_grad()
def eval_split(model, x, y):
    logits = model.logits_at_last(x)
    loss = F.cross_entropy(logits, y).item()
    acc = (logits.argmax(dim=-1) == y).float().mean().item()
    return loss, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-mlp", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--tag", type=str, default="grokking")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    xtr, ytr, xte, yte = load_data(device)
    vocab = MODULAR_P + 1
    print(f"[data] train={xtr.shape[0]} test={xte.shape[0]} vocab={vocab} device={device}")

    model = GrokkingTransformer(
        vocab=vocab, n_ctx=xtr.shape[1],
        d_model=args.d_model, n_heads=args.n_heads,
        d_mlp=args.d_mlp, n_layers=args.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params:,}")

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98)
    )

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project="grokking-circuits",
                name=f"{args.tag}-p{MODULAR_P}-steps{args.steps}",
                config=vars(args) | {"p": MODULAR_P, "n_params": n_params},
            )
        except Exception as e:
            print(f"[wandb] disabled ({e})")
            use_wandb = False

    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_path = os.path.join(RESULTS_DIR, f"{args.tag}_log.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)
    log_f = open(log_path, "a", buffering=1)

    t0 = time.time()
    model.train()
    for step in range(args.steps + 1):
        opt.zero_grad(set_to_none=True)
        logits = model.logits_at_last(xtr)
        loss = F.cross_entropy(logits, ytr)
        loss.backward()
        opt.step()

        if step % args.log_every == 0 or step == args.steps:
            model.eval()
            tr_loss, tr_acc = eval_split(model, xtr, ytr)
            te_loss, te_acc = eval_split(model, xte, yte)
            model.train()
            elapsed = time.time() - t0
            rec = {
                "step": step, "elapsed_s": round(elapsed, 1),
                "train_loss": tr_loss, "train_acc": tr_acc,
                "test_loss": te_loss, "test_acc": te_acc,
            }
            log_f.write(json.dumps(rec) + "\n")
            print(f"[{step:>6}/{args.steps}] "
                  f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                  f"test loss={te_loss:.4f} acc={te_acc:.4f} | "
                  f"{elapsed:.1f}s")
            if use_wandb:
                wandb.log(rec, step=step)

        if step > 0 and (step % args.ckpt_every == 0 or step == args.steps):
            ck = os.path.join(RESULTS_DIR, f"{args.tag}_ckpt_step{step}.pt")
            torch.save({"step": step, "model": model.state_dict(),
                        "config": vars(args)}, ck)

    log_f.close()
    if use_wandb:
        wandb.finish()
    print(f"[done] total {time.time() - t0:.1f}s — log: {log_path}")


if __name__ == "__main__":
    main()
