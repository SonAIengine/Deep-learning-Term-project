"""Grokking circuit analysis — Step 2 of sonsj-proposal.

Runs:
  1. Fourier decomposition of W_E (token embedding) on final ckpt
  2. Per-head and per-MLP zero-ablation (test acc drop)
  3. Time-series across all checkpoints: Fourier mass + ablation effects
  4. Attention pattern visualization (saved as PNG grid)

Usage: python -m src.grokking.analysis [--tag grokking_full]
Outputs land in results/analysis/{tag}/.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shared.config import MODULAR_P, DATASETS_DIR, RESULTS_DIR
from src.grokking.model import GrokkingTransformer


def load_ckpt(path, device):
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = blob["config"]
    model = GrokkingTransformer(
        vocab=MODULAR_P + 1, n_ctx=3,
        d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        d_mlp=cfg["d_mlp"], n_layers=cfg["n_layers"],
    ).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, blob["step"]


def load_test_data(device):
    te = torch.load(os.path.join(DATASETS_DIR, "modular_test.pt"), weights_only=True)
    return te["x"].to(device), te["y"].to(device)


@torch.no_grad()
def test_accuracy(model, x, y):
    logits = model.logits_at_last(x)
    return (logits.argmax(dim=-1) == y).float().mean().item()


@torch.no_grad()
def fourier_mass_W_E(model):
    """Return mass of W_E[0:p] per Fourier frequency k=1..(p-1)//2.

    Nanda 2023: W_E rows for numerical tokens live in a sparse Fourier basis.
    Mass at frequency k = ||cos_k . W_E||^2 + ||sin_k . W_E||^2 summed over d_model.
    """
    p = MODULAR_P
    W = model.tok_emb.weight[:p].detach().cpu().numpy()       # [p, d_model]
    n = (p - 1) // 2
    a = np.arange(p)
    mass = np.zeros(n + 1)                                    # index by k=0..n
    for k in range(n + 1):
        cos_k = np.cos(2 * np.pi * k * a / p)
        sin_k = np.sin(2 * np.pi * k * a / p)
        c = cos_k @ W                                         # [d_model]
        s = sin_k @ W
        mass[k] = (c * c).sum() + (s * s).sum()
    return mass


@torch.no_grad()
def ablate_head(model, layer: int, head: int, x, y):
    """Zero out one attention head's output via a forward hook."""
    d_head = model.blocks[layer].attn.d_head
    n_heads = model.blocks[layer].attn.n_heads
    attn = model.blocks[layer].attn

    def hook(module, inputs, output):
        # output: [B, T, d_model], reshape to [B, T, H, Dh] to mask one head
        B, T, C = output.shape
        out = output.view(B, T, n_heads, d_head).clone()
        out[:, :, head, :] = 0
        return out.view(B, T, C)

    h = attn.proj.register_forward_hook(hook)
    try:
        acc = test_accuracy(model, x, y)
    finally:
        h.remove()
    return acc


@torch.no_grad()
def ablate_mlp(model, layer: int, x, y):
    mlp = model.blocks[layer].mlp
    def hook(module, inputs, output):
        return torch.zeros_like(output)
    h = mlp.fc2.register_forward_hook(hook)
    try:
        acc = test_accuracy(model, x, y)
    finally:
        h.remove()
    return acc


@torch.no_grad()
def attention_patterns(model, x):
    """Capture attention pattern for each layer from a batch of inputs."""
    p = MODULAR_P
    patterns = {}

    def make_hook(layer):
        def hook(module, inputs, output):
            # Recompute attn weights from qkv to log them (forward_hook can't intercept softmax easily)
            pass
        return hook

    # Easier: monkey-patch CausalSelfAttention.forward to log
    from src.grokking.model import CausalSelfAttention
    orig_forward = CausalSelfAttention.forward
    logs = {}

    def patched(self, xx):
        import math
        B, T, C = xx.shape
        qkv = self.qkv(xx).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(T, T, device=xx.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        logs.setdefault(id(self), []).append(attn.mean(dim=0).cpu().numpy())  # avg over batch
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)

    CausalSelfAttention.forward = patched
    try:
        _ = model.forward(x)
    finally:
        CausalSelfAttention.forward = orig_forward

    # Reorder logs by layer index
    by_layer = {}
    for i, block in enumerate(model.blocks):
        key = id(block.attn)
        if key in logs:
            by_layer[i] = logs[key][-1]                       # [H, T, T]
    return by_layer


def save_fourier_plot(mass, title, out_path):
    n = len(mass) - 1
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(range(1, n + 1), mass[1:])
    ax.set_xlabel("frequency k")
    ax.set_ylabel("||W_E projected onto (cos_k, sin_k)||²")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_attn_grid(patterns, title, out_path):
    n_layers = len(patterns)
    if n_layers == 0:
        return
    H = patterns[0].shape[0]
    fig, axes = plt.subplots(n_layers, H, figsize=(2 * H, 2 * n_layers), squeeze=False)
    for L in range(n_layers):
        for h in range(H):
            ax = axes[L][h]
            ax.imshow(patterns[L][h], vmin=0, vmax=1, cmap="viridis")
            ax.set_title(f"L{L}H{h}", fontsize=8)
            ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
            ax.set_xticklabels(["a", "b", "="], fontsize=7)
            ax.set_yticklabels(["a", "b", "="], fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_ckpt_timeseries(records, out_path):
    steps = [r["step"] for r in records]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    # Top-5 Fourier frequency mass over time
    mass_arr = np.array([r["fourier_mass"] for r in records])    # [T, K]
    top5 = np.argsort(mass_arr[-1])[-5:][::-1]
    for k in top5:
        axes[0].plot(steps, mass_arr[:, k], label=f"k={k}")
    axes[0].set_ylabel("Fourier mass (W_E)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Top-5 final-checkpoint frequencies — emergence over training")
    # test acc + worst single-component ablation drop
    test_acc = [r["test_acc"] for r in records]
    worst_ablate = [min(min(r["head_ablation"].values()), min(r["mlp_ablation"].values()))
                    for r in records]
    axes[1].plot(steps, test_acc, label="test acc (no ablation)", color="black")
    axes[1].plot(steps, worst_ablate, label="test acc after worst single ablation", color="red")
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("test accuracy")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Causal importance — biggest single-component drop")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="grokking_full")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(RESULTS_DIR, "analysis", args.tag)
    os.makedirs(out_dir, exist_ok=True)

    x, y = load_test_data(device)

    # discover checkpoints in step order
    ckpt_paths = glob.glob(os.path.join(RESULTS_DIR, f"{args.tag}_ckpt_step*.pt"))
    def step_of(p): return int(re.search(r"step(\d+)", p).group(1))
    ckpt_paths.sort(key=step_of)
    print(f"[scan] {len(ckpt_paths)} checkpoints found")

    records = []
    for path in ckpt_paths:
        model, step = load_ckpt(path, device)
        acc = test_accuracy(model, x, y)
        mass = fourier_mass_W_E(model)

        n_layers = len(model.blocks)
        n_heads = model.blocks[0].attn.n_heads
        head_drops = {f"L{L}H{h}": ablate_head(model, L, h, x, y)
                      for L in range(n_layers) for h in range(n_heads)}
        mlp_drops = {f"L{L}_mlp": ablate_mlp(model, L, x, y)
                     for L in range(n_layers)}

        rec = {
            "step": step,
            "test_acc": acc,
            "fourier_mass": mass.tolist(),
            "head_ablation": head_drops,
            "mlp_ablation": mlp_drops,
        }
        records.append(rec)
        print(f"  step={step:>6} test_acc={acc:.4f} "
              f"top_freq={int(np.argsort(mass)[::-1][0]):>3}  "
              f"head_min={min(head_drops.values()):.4f}  "
              f"mlp_min={min(mlp_drops.values()):.4f}")

    # save records
    with open(os.path.join(out_dir, "records.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # final-ckpt plots
    final = records[-1]
    save_fourier_plot(
        np.array(final["fourier_mass"]),
        f"Final ckpt W_E Fourier mass (step={final['step']}, test_acc={final['test_acc']:.4f})",
        os.path.join(out_dir, "fourier_final.png"),
    )
    # also save first/middle/last side by side
    for tag, rec in [("init", records[0]),
                     ("mid", records[len(records) // 2]),
                     ("final", records[-1])]:
        save_fourier_plot(
            np.array(rec["fourier_mass"]),
            f"{tag} (step={rec['step']}, test_acc={rec['test_acc']:.4f})",
            os.path.join(out_dir, f"fourier_{tag}.png"),
        )

    # final-ckpt attention pattern
    model, _ = load_ckpt(ckpt_paths[-1], device)
    pats = attention_patterns(model, x[:512])
    save_attn_grid(pats, f"Attention (avg over 512 test samples) — final ckpt",
                   os.path.join(out_dir, "attn_final.png"))

    save_ckpt_timeseries(records, os.path.join(out_dir, "timeseries.png"))

    # print final ablation ranking
    final = records[-1]
    all_ablate = {**final["head_ablation"], **final["mlp_ablation"]}
    ranked = sorted(all_ablate.items(), key=lambda kv: kv[1])
    print("\n[final ablation ranking — smallest acc = most important]")
    for name, acc in ranked:
        baseline = final["test_acc"]
        print(f"  {name:>10}: acc={acc:.4f}  drop={baseline - acc:+.4f}")

    print(f"\n[done] outputs → {out_dir}")


if __name__ == "__main__":
    main()
