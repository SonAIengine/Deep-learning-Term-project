"""Pull grokking run history from wandb cloud and render wandb-style charts.

Outputs to results/wandb/:
  - dashboard.png           — 4-panel (loss, acc, smoothed loss, smoothed acc)
  - loss_log.png            — log-scale loss curves
  - accuracy.png            — accuracy curves
  - grokking_gap.png        — train_acc − test_acc gap (highlights grokking)
  - system_metrics.png      — GPU/CPU/mem (if available)
  - run_info.json           — run metadata (config, runtime, system)
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shared.config import RESULTS_DIR


WANDB_ENTITY = "sonsj97-plateer"
WANDB_PROJECT = "grokking-circuits"
WANDB_RUN_ID = "n4bnqrak"

# wandb default chart style
WANDB_BLUE = "#5387dd"
WANDB_RED = "#dd5f53"
WANDB_GREEN = "#4caf50"
WANDB_GRAY = "#9aa0a6"


def ema(arr, alpha=0.95):
    out = np.zeros_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * out[i - 1] + (1 - alpha) * arr[i]
    return out


def wandb_style(ax):
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#fafbfc")


def main():
    import wandb
    api = wandb.Api()
    run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{WANDB_RUN_ID}")
    print(f"[load] {run.name}  state={run.state}  url={run.url}")

    out_dir = os.path.join(RESULTS_DIR, "wandb_charts")
    os.makedirs(out_dir, exist_ok=True)

    # ---- history (full sample, not page-limited) ----
    hist = run.history(samples=10000, pandas=True)
    print(f"[hist] {len(hist)} rows, cols: {list(hist.columns)}")
    hist.to_csv(os.path.join(out_dir, "history.csv"), index=False)

    steps = hist["_step"].to_numpy()
    metrics = {}
    for col in ["train_loss", "test_loss", "train_acc", "test_acc"]:
        if col in hist.columns:
            metrics[col] = hist[col].to_numpy()

    # ---- run info ----
    info = {
        "url": run.url,
        "name": run.name,
        "state": run.state,
        "config": dict(run.config),
        "summary": {k: v for k, v in run.summary.items() if not k.startswith("_")},
        "runtime_seconds": run.summary.get("_runtime"),
        "created_at": str(run.created_at),
        "system": {"gpu": run.metadata.get("gpu") if run.metadata else None,
                   "cpu_count": run.metadata.get("cpu_count") if run.metadata else None},
    }
    with open(os.path.join(out_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)
    print(f"[info] config: {info['config']}")
    print(f"[info] gpu: {info['system']['gpu']}")

    # ---- 1. 4-panel dashboard ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"wandb: {run.name}  ·  {run.url}",
                 fontsize=11, color=WANDB_GRAY)

    ax = axes[0, 0]
    ax.semilogy(steps, metrics["train_loss"], color=WANDB_BLUE,
                linewidth=1, alpha=0.4, label="raw")
    ax.semilogy(steps, ema(metrics["train_loss"]), color=WANDB_BLUE,
                linewidth=2, label="smoothed")
    ax.set_title("train/loss", fontweight="bold")
    ax.set_xlabel("Step"); ax.set_ylabel("loss")
    ax.legend(loc="upper right"); wandb_style(ax)

    ax = axes[0, 1]
    ax.semilogy(steps, metrics["test_loss"], color=WANDB_RED,
                linewidth=1, alpha=0.4, label="raw")
    ax.semilogy(steps, ema(metrics["test_loss"]), color=WANDB_RED,
                linewidth=2, label="smoothed")
    ax.set_title("test/loss", fontweight="bold")
    ax.set_xlabel("Step"); ax.set_ylabel("loss")
    ax.legend(loc="upper right"); wandb_style(ax)

    ax = axes[1, 0]
    ax.plot(steps, metrics["train_acc"], color=WANDB_BLUE,
            linewidth=1, alpha=0.4, label="raw")
    ax.plot(steps, ema(metrics["train_acc"], alpha=0.9),
            color=WANDB_BLUE, linewidth=2, label="smoothed")
    ax.set_title("train/acc", fontweight="bold")
    ax.set_xlabel("Step"); ax.set_ylabel("accuracy"); ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right"); wandb_style(ax)

    ax = axes[1, 1]
    ax.plot(steps, metrics["test_acc"], color=WANDB_RED,
            linewidth=1, alpha=0.4, label="raw")
    ax.plot(steps, ema(metrics["test_acc"], alpha=0.9),
            color=WANDB_RED, linewidth=2, label="smoothed")
    ax.set_title("test/acc", fontweight="bold")
    ax.set_xlabel("Step"); ax.set_ylabel("accuracy"); ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right"); wandb_style(ax)

    fig.tight_layout()
    out = os.path.join(out_dir, "dashboard.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")

    # ---- 2. loss log-scale combined ----
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.semilogy(steps, metrics["train_loss"], color=WANDB_BLUE,
                linewidth=1, alpha=0.35)
    ax.semilogy(steps, ema(metrics["train_loss"]), color=WANDB_BLUE,
                linewidth=2.5, label="train loss")
    ax.semilogy(steps, metrics["test_loss"], color=WANDB_RED,
                linewidth=1, alpha=0.35)
    ax.semilogy(steps, ema(metrics["test_loss"]), color=WANDB_RED,
                linewidth=2.5, label="test loss")
    ax.set_xlabel("Step"); ax.set_ylabel("loss (log)")
    ax.set_title("Grokking: train vs test loss (log scale)\n"
                 "memorization → generalization transition", fontweight="bold")
    ax.legend(); wandb_style(ax)
    out = os.path.join(out_dir, "loss_log.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")

    # ---- 3. accuracy combined ----
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(steps, metrics["train_acc"], color=WANDB_BLUE,
            linewidth=1, alpha=0.35)
    ax.plot(steps, ema(metrics["train_acc"], alpha=0.9),
            color=WANDB_BLUE, linewidth=2.5, label="train acc")
    ax.plot(steps, metrics["test_acc"], color=WANDB_RED,
            linewidth=1, alpha=0.35)
    ax.plot(steps, ema(metrics["test_acc"], alpha=0.9),
            color=WANDB_RED, linewidth=2.5, label="test acc")
    ax.axhline(1.0, color=WANDB_GRAY, linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_xlabel("Step"); ax.set_ylabel("accuracy"); ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Grokking: train vs test accuracy  ·  final test = {metrics['test_acc'][-1]*100:.2f}%",
                 fontweight="bold")
    ax.legend(); wandb_style(ax)
    out = os.path.join(out_dir, "accuracy.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")

    # ---- 4. grokking gap ----
    gap = metrics["train_acc"] - metrics["test_acc"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(steps, 0, ema(gap, alpha=0.9),
                    color=WANDB_GREEN, alpha=0.3)
    ax.plot(steps, gap, color=WANDB_GREEN, linewidth=1, alpha=0.4)
    ax.plot(steps, ema(gap, alpha=0.9), color=WANDB_GREEN,
            linewidth=2.5, label="train − test")
    ax.set_xlabel("Step"); ax.set_ylabel("acc gap")
    ax.set_title("Grokking gap (train_acc − test_acc): high → low = generalization happens",
                 fontweight="bold")
    ax.legend(); wandb_style(ax)
    out = os.path.join(out_dir, "grokking_gap.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")

    # ---- 5. system metrics (if available) ----
    sys_hist = run.history(stream="system", samples=10000, pandas=True)
    sys_cols = [c for c in sys_hist.columns if c.startswith("system.")]
    if sys_cols:
        sys_hist.to_csv(os.path.join(out_dir, "system_history.csv"), index=False)
        # pick interesting cols
        pick = []
        for c in sys_cols:
            if any(k in c for k in ["gpu.0.memoryAllocated", "gpu.0.gpu",
                                     "cpu", "memory."]):
                pick.append(c)
        pick = pick[:6]
        if pick:
            n = len(pick)
            fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(13, 3 * ((n + 1) // 2)))
            axes = np.array(axes).flatten()
            t = sys_hist["_runtime"].to_numpy()
            for i, col in enumerate(pick):
                v = sys_hist[col].to_numpy()
                axes[i].plot(t, v, color=WANDB_BLUE, linewidth=1.5)
                axes[i].set_title(col.replace("system.", ""), fontsize=10)
                axes[i].set_xlabel("runtime (s)"); wandb_style(axes[i])
            for i in range(len(pick), len(axes)):
                axes[i].axis("off")
            fig.suptitle("System metrics from wandb", fontsize=11, color=WANDB_GRAY)
            fig.tight_layout()
            out = os.path.join(out_dir, "system_metrics.png")
            fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"  → {out}")

    print(f"\n[done] all charts in {out_dir}")
    print(f"  wandb URL: {run.url}")


if __name__ == "__main__":
    main()
