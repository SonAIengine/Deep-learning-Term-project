"""Step 3 extra (D2): replication on GPT-2 medium (24L × 16H = 384 heads).

Same patching + necessity test as Pythia, but on a larger same-family model.
Question: does the late-layer concentration of GPT-2 small persist with scale,
or does the binding circuit move?
"""
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shared.config import DATASETS_DIR, RESULTS_DIR


def load_pairs(path):
    by_cf = defaultdict(dict)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by_cf[r["cf_id"]][r["role"]] = r
    return [{"cf_id": cf, "clean": d["clean"], "corrupt": d["corrupt"]}
            for cf, d in by_cf.items() if "clean" in d and "corrupt" in d]


def logit_diff(logits_last, answer_id, distractor_ids):
    ans = logits_last[answer_id]
    dist = logits_last[torch.tensor(distractor_ids, device=logits_last.device)].mean()
    return (ans - dist).item()


@torch.no_grad()
def main():
    from transformer_lens import HookedTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] GPT-2 medium on {device}")
    model = HookedTransformer.from_pretrained("gpt2-medium", device=device)
    model.eval()
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    print(f"  arch: {n_layers}L × {n_heads}H, d_model={model.cfg.d_model}, "
          f"total heads={n_layers*n_heads}")

    # GPT-2 family uses same tokenizer → answer_token_id from dataset is valid
    pairs = load_pairs(os.path.join(DATASETS_DIR, "var_binding_tier1.jsonl"))[:500]
    print(f"[data] {len(pairs)} pairs")

    effects = np.full((len(pairs), n_layers, n_heads), np.nan, dtype=np.float32)
    clean_lds, corr_lds = [], []
    import time
    t0 = time.time()
    for i, p in enumerate(pairs):
        ctok = model.to_tokens(p["clean"]["prompt"], prepend_bos=False).to(device)
        xtok = model.to_tokens(p["corrupt"]["prompt"], prepend_bos=False).to(device)
        if ctok.shape != xtok.shape:
            continue
        ans_id = p["clean"]["answer_token_id"]
        dist_ids = p["clean"]["distractor_answer_token_ids"]
        if not dist_ids:
            continue
        clean_logits = model(ctok)
        clean_ld = logit_diff(clean_logits[0, -1], ans_id, dist_ids)
        _, clean_cache = model.run_with_cache(ctok)
        corr_logits = model(xtok)
        corr_ld = logit_diff(corr_logits[0, -1], ans_id, dist_ids)
        clean_lds.append(clean_ld); corr_lds.append(corr_ld)
        gap = clean_ld - corr_ld
        if abs(gap) < 1e-4:
            continue
        for L in range(n_layers):
            clean_z = clean_cache[f"blocks.{L}.attn.hook_z"]
            for H in range(n_heads):
                def hook_fn(z, hook, _H=H, _cz=clean_z):
                    z[:, :, _H, :] = _cz[:, :, _H, :]
                    return z
                patched = model.run_with_hooks(
                    xtok, fwd_hooks=[(f"blocks.{L}.attn.hook_z", hook_fn)])
                p_ld = logit_diff(patched[0, -1], ans_id, dist_ids)
                effects[i, L, H] = (p_ld - corr_ld) / gap
        if (i + 1) % 25 == 0:
            print(f"  [{i+1:>3}/{len(pairs)}] {time.time()-t0:.0f}s")

    out_dir = os.path.join(RESULTS_DIR, "binding_gpt2med")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "head_effects.npy"), effects)

    mean_eff = np.nanmean(effects, axis=0)
    clean_mean = float(np.mean(clean_lds))
    corr_mean = float(np.mean(corr_lds))
    flat = [(float(mean_eff[L, H]), L, H) for L in range(n_layers) for H in range(n_heads)]
    flat.sort(key=lambda t: -t[0])
    print(f"\nbaselines: clean={clean_mean:+.4f} corrupt={corr_mean:+.4f} "
          f"gap={clean_mean-corr_mean:+.4f}")
    print("\n=== GPT-2 medium: top-15 binding heads ===")
    for m, L, H in flat[:15]:
        print(f"  L{L:>2}H{H:>2}  recovery={m:+.4f}")

    # layer profile
    layer_means = mean_eff.mean(axis=1)
    print("\n=== Per-layer mean recovery ===")
    for L in range(n_layers):
        bar = "█" * max(0, int(layer_means[L] * 200))
        print(f"  L{L:>2}: {layer_means[L]:+.4f}  {bar}")

    # ============ Necessity ============
    def run_remove(remove_heads):
        rem = defaultdict(set)
        for (L, H) in remove_heads:
            rem[L].add(H)
        def make_hook(L):
            rh = rem.get(L, set())
            def hk(z, hook):
                for H in rh:
                    z[:, :, H, :] = 0
                return z
            return hk
        hooks = [(f"blocks.{L}.attn.hook_z", make_hook(L)) for L in range(n_layers) if rem[L]]
        lds = []
        for p in pairs:
            ctok = model.to_tokens(p["clean"]["prompt"], prepend_bos=False).to(device)
            ans_id = p["clean"]["answer_token_id"]
            dist_ids = p["clean"]["distractor_answer_token_ids"]
            if not dist_ids:
                continue
            logits = model.run_with_hooks(ctok, fwd_hooks=hooks)
            lds.append(logit_diff(logits[0, -1], ans_id, dist_ids))
        return float(np.mean(lds))

    def run_empty_attn():
        def hk(z, hook):
            z[:, :, :, :] = 0
            return z
        hooks = [(f"blocks.{L}.attn.hook_z", hk) for L in range(n_layers)]
        lds = []
        for p in pairs:
            ctok = model.to_tokens(p["clean"]["prompt"], prepend_bos=False).to(device)
            ans_id = p["clean"]["answer_token_id"]
            dist_ids = p["clean"]["distractor_answer_token_ids"]
            if not dist_ids:
                continue
            logits = model.run_with_hooks(ctok, fwd_hooks=hooks)
            lds.append(logit_diff(logits[0, -1], ans_id, dist_ids))
        return float(np.mean(lds))

    print("\n[necessity] computing baselines...")
    empty_ld = run_empty_attn()
    attn_total = clean_mean - empty_ld
    print(f"  empty-attn ld = {empty_ld:+.4f}  attn total = {attn_total:+.4f}")

    medium_top10 = [(L, H) for _, L, H in flat[:10]]
    medium_top30 = [(L, H) for _, L, H in flat[:30]]   # ~IOI-26 size scaled
    all_heads = [(L, H) for L in range(n_layers) for H in range(n_heads)]
    necessity = {"empty_ld": empty_ld, "clean_ld": clean_mean,
                  "attn_total": attn_total, "sets": {}}
    print("\n[necessity] ablation comparison")
    print(f"  {'set':>14}  {'k':>3}  {'drop':>10}")
    for K, top_heads in [(3, medium_top10[:3]),
                         (5, medium_top10[:5]),
                         (10, medium_top10),
                         (30, medium_top30)]:
        m_top = run_remove(top_heads)
        d_top = clean_mean - m_top
        rdrops = []
        for s in range(5):
            rng = random.Random(s * 100 + K)
            rand = rng.sample(all_heads, K)
            m_r = run_remove(rand)
            rdrops.append(clean_mean - m_r)
        m_rd, s_rd = float(np.mean(rdrops)), float(np.std(rdrops))
        necessity["sets"][f"top{K}"] = {"k": K, "drop": d_top,
                                          "frac_of_attn": d_top / attn_total}
        necessity["sets"][f"random{K}"] = {"k": K, "mean_drop": m_rd, "std_drop": s_rd}
        print(f"  {'top'+str(K):>14}  {K:>3}  {d_top:+.4f}  "
              f"({d_top/attn_total:.0%} of attn)")
        print(f"  {'random'+str(K):>14}  {K:>3}  {m_rd:+.4f} ± {s_rd:.4f}")

    report = {
        "n_pairs": len(pairs), "clean_ld": clean_mean, "corrupt_ld": corr_mean,
        "n_layers": n_layers, "n_heads": n_heads,
        "top15": [{"layer": L, "head": H, "recovery": m} for m, L, H in flat[:15]],
        "layer_means": layer_means.tolist(),
        "necessity": necessity,
    }
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    vmax = float(np.nanmax(np.abs(mean_eff)))
    im = ax.imshow(mean_eff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for L in range(n_layers):
        for H in range(n_heads):
            v = mean_eff[L, H]
            if abs(v) > vmax * 0.3:
                ax.text(H, L, f"{v:+.2f}", ha="center", va="center", fontsize=5,
                        color="white" if abs(v) > vmax * 0.6 else "black")
    ax.set_xlabel("head"); ax.set_ylabel("layer")
    ax.set_title(f"GPT-2 medium ({n_layers}L × {n_heads}H): patching recovery  (n={len(pairs)})")
    plt.colorbar(im, ax=ax, label="recovery fraction")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gpt2med_heatmap.png"), dpi=120)
    plt.close(fig)

    # layer profile vs GPT-2 small vs Pythia
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(n_layers), layer_means, "-o", color="tab:blue",
            label="GPT-2 medium (24L)", linewidth=2)
    # overlay small & pythia for comparison if available
    for name, color, label, path in [
        ("gpt2-small", "tab:red", "GPT-2 small (12L)",
         os.path.join(RESULTS_DIR, "binding_patching/head_effects.npy")),
        ("pythia", "tab:green", "Pythia-160M (12L)",
         os.path.join(RESULTS_DIR, "binding_pythia/head_effects.npy")),
    ]:
        if os.path.exists(path):
            eff_other = np.load(path)
            lm = np.nanmean(eff_other, axis=0).mean(axis=1)
            xs = np.linspace(0, n_layers - 1, len(lm))   # rescale to medium's x-axis
            ax.plot(xs, lm, "--o", color=color, label=label, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("layer (scaled to GPT-2 medium 24 layers)")
    ax.set_ylabel("mean per-layer recovery")
    ax.set_title("Layer profile of binding heads across 3 models")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "layer_profile_comparison.png"), dpi=120)
    plt.close(fig)
    print(f"\n[done] {time.time()-t0:.0f}s → {out_dir}")


if __name__ == "__main__":
    main()
