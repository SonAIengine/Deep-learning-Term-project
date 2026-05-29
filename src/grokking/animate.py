"""Generate presentation-ready animations from grokking_full_log.jsonl and ckpts.

Outputs:
  - results/analysis/grokking_full/loss_curve.gif
  - results/analysis/grokking_full/attention_evolution.gif
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shared.config import RESULTS_DIR


def load_log(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def loss_curve_gif():
    log = load_log(os.path.join(RESULTS_DIR, "grokking_full_log.jsonl"))
    steps = np.array([r["step"] for r in log])
    train_loss = np.array([r["train_loss"] for r in log])
    test_loss = np.array([r["test_loss"] for r in log])
    train_acc = np.array([r["train_acc"] for r in log])
    test_acc = np.array([r["test_acc"] for r in log])

    # Subsample for animation speed (~120 frames)
    n_frames = 120
    stride = max(1, len(steps) // n_frames)
    idx_grid = list(range(0, len(steps), stride))
    if idx_grid[-1] != len(steps) - 1:
        idx_grid.append(len(steps) - 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Grokking: 2L Transformer on Modular Addition (p=113)",
                 fontsize=13, fontweight="bold")

    def init():
        for ax in (ax1, ax2):
            ax.clear()
        return []

    def update(frame_idx):
        i = idx_grid[frame_idx]
        ax1.clear(); ax2.clear()

        ax1.semilogy(steps[:i+1], train_loss[:i+1], color="tab:blue",
                     label="train loss", linewidth=2)
        ax1.semilogy(steps[:i+1], test_loss[:i+1], color="tab:red",
                     label="test loss", linewidth=2)
        ax1.set_xlim(0, steps[-1])
        ax1.set_ylim(1e-7, 10)
        ax1.set_xlabel("step")
        ax1.set_ylabel("loss (log scale)")
        ax1.set_title(f"Loss  ·  step = {steps[i]:>5}")
        ax1.legend(loc="upper right")
        ax1.grid(alpha=0.3)

        ax2.plot(steps[:i+1], train_acc[:i+1], color="tab:blue",
                 label="train acc", linewidth=2)
        ax2.plot(steps[:i+1], test_acc[:i+1], color="tab:red",
                 label="test acc", linewidth=2)
        ax2.set_xlim(0, steps[-1])
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_xlabel("step")
        ax2.set_ylabel("accuracy")
        ax2.set_title(f"Accuracy  ·  test = {test_acc[i]*100:.1f}%")
        ax2.legend(loc="lower right")
        ax2.grid(alpha=0.3)

        # annotate phase
        if test_acc[i] < 0.3:
            phase = "Phase 1: Memorization (train↑, test flat)"
            color = "tab:orange"
        elif test_acc[i] < 0.9:
            phase = "Phase 2: Grokking transition"
            color = "tab:purple"
        else:
            phase = "Phase 3: Generalized (test ≈ train)"
            color = "tab:green"
        fig.text(0.5, 0.93, phase, ha="center", fontsize=11,
                 color=color, fontweight="bold")
        return []

    print(f"[loss-gif] {len(idx_grid)} frames")
    anim = FuncAnimation(fig, update, frames=len(idx_grid), init_func=init,
                         blit=False, interval=80)
    out = os.path.join(RESULTS_DIR, "analysis/grokking_full/loss_curve.gif")
    anim.save(out, writer=PillowWriter(fps=15))
    plt.close(fig)
    print(f"  → {out}")


def attention_evolution_gif():
    """Build a GIF cycling through the 20 saved checkpoints showing attention patterns."""
    import torch
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from grokking.model import GrokkingTransformer

    p = 113
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_steps = list(range(2000, 40001, 2000))
    ckpts = []
    for s in ckpt_steps:
        path = os.path.join(RESULTS_DIR, f"grokking_full_ckpt_step{s}.pt")
        if os.path.exists(path):
            ckpts.append((s, path))
    print(f"[attn-gif] {len(ckpts)} checkpoints")
    if not ckpts:
        print("  no checkpoints found")
        return

    # use a single example input: (a=7, b=29) → token sequence [7, 29, =]
    a, b = 7, 29
    eq_tok = p          # token index for "="
    inp = torch.tensor([[a, b, eq_tok]], device=device)

    # capture attention patterns by monkey-patching forward
    state = {"attn_patterns": []}

    def capture_forward(self, x):
        import math as _math
        import torch.nn.functional as F
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / _math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        att = scores.softmax(dim=-1)
        state["attn_patterns"].append(att.detach().cpu().numpy()[0])  # [n_heads, T, T]
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)

    from grokking.model import CausalSelfAttention
    orig_fwd = CausalSelfAttention.forward
    CausalSelfAttention.forward = capture_forward

    all_patterns = []
    for step, path in ckpts:
        sd = torch.load(path, map_location=device, weights_only=False)
        state_dict = sd["model"]
        cfg = sd.get("config", {})
        # constructor expects: vocab, n_ctx, d_model, n_heads, d_mlp, n_layers
        kw = {"vocab": cfg.get("vocab", p + 1),
              "n_ctx": cfg.get("n_ctx", cfg.get("ctx", 3)),
              "d_model": cfg.get("d_model", 128),
              "n_heads": cfg.get("n_heads", 4),
              "d_mlp": cfg.get("d_mlp", 512),
              "n_layers": cfg.get("n_layers", 2)}
        model = GrokkingTransformer(**kw).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        state["attn_patterns"] = []
        with torch.no_grad():
            _ = model(inp)
        # state["attn_patterns"] has 2 entries (2 layers), each [n_heads, T, T]
        all_patterns.append((step, state["attn_patterns"]))

    CausalSelfAttention.forward = orig_fwd

    n_layers = len(all_patterns[0][1])
    n_heads = all_patterns[0][1][0].shape[0]
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(2.5 * n_heads, 2.5 * n_layers))
    if n_layers == 1:
        axes = np.array([axes])
    fig.suptitle(f"Attention pattern evolution during grokking  ·  input: ({a} + {b} mod {p})",
                 fontsize=12, fontweight="bold")

    ims = []
    for L in range(n_layers):
        for H in range(n_heads):
            ax = axes[L, H]
            im = ax.imshow(all_patterns[0][1][L][H], cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks([0, 1, 2]); ax.set_xticklabels([f"{a}", f"{b}", "="], fontsize=8)
            ax.set_yticks([0, 1, 2]); ax.set_yticklabels([f"{a}", f"{b}", "="], fontsize=8)
            ax.set_title(f"L{L}H{H}", fontsize=9)
            ims.append((L, H, im))

    step_text = fig.text(0.5, 0.93, "", ha="center", fontsize=11, fontweight="bold")

    def update(idx):
        step, patterns = all_patterns[idx]
        for L, H, im in ims:
            im.set_data(patterns[L][H])
        # phase annotation
        if step < 8000:
            phase = "memorization"
            color = "tab:orange"
        elif step < 16000:
            phase = "grokking transition"
            color = "tab:purple"
        else:
            phase = "generalized"
            color = "tab:green"
        step_text.set_text(f"step {step:>5}  ({phase})")
        step_text.set_color(color)
        return [im for _, _, im in ims]

    anim = FuncAnimation(fig, update, frames=len(all_patterns),
                         blit=False, interval=400)
    out = os.path.join(RESULTS_DIR, "analysis/grokking_full/attention_evolution.gif")
    anim.save(out, writer=PillowWriter(fps=4))
    plt.close(fig)
    print(f"  → {out}")


if __name__ == "__main__":
    print("Building loss curve GIF...")
    loss_curve_gif()
    print("\nBuilding attention evolution GIF...")
    attention_evolution_gif()
    print("\n[done]")
