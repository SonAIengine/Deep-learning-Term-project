"""Step 3 extra (D): cross-model replication on Pythia-160M.

Same patching + necessity experiment as on GPT-2 small, but on a different
architecture (rotary embeddings, parallel attn+MLP) and training data (Pile).
Same width (12L × 12H) so head coordinates can be directly compared.

Research questions:
  Q1. Does patching reveal a similarly concentrated set of binding heads?
  Q2. Are top heads in late layers (output-stage), as in GPT-2 small?
  Q3. Do the top-K heads overlap with GPT-2's top heads at the same coords?
  Q4. Necessity test — does ablating top-K hurt more than random K?
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


def retokenize_answer(model, answer_str):
    """Get Pythia token id for a single-character answer string.
    The cf data has token_ids from GPT-2 tokenizer; we need to redo for Pythia."""
    toks = model.to_tokens(answer_str, prepend_bos=False)
    if toks.shape[1] != 1:
        return None
    return int(toks[0, 0].item())


def retokenize_pair(model, pair):
    """Rebuild ans/distractor ids for Pythia."""
    clean = pair["clean"]
    # Answer strings — vb prompts have single-digit answers (0-9)
    # We assume token_str field exists or reconstruct from logits position
    # The dataset has 'answer_str' or 'answer_token' — but token_id is GPT-2 specific
    # Workaround: look at the prompt — last char before '=' is the digit assignment.
    # Actually the answer in this dataset is the *value* that the source var holds.
    # We get the answer string by looking at the dataset's 'answer' field if present.
    ans_str = clean.get("answer_str") or clean.get("answer")
    if ans_str is None:
        return None
    dist_strs = (clean.get("distractor_answers_str")
                 or clean.get("distractor_answers")
                 or clean.get("distractor_values"))
    if not dist_strs:
        return None
    # Try with leading space (GPT-NeoX tokenizer often needs it)
    ans_id = retokenize_answer(model, str(ans_str))
    if ans_id is None:
        ans_id = retokenize_answer(model, " " + str(ans_str))
    if ans_id is None:
        return None
    dist_ids = []
    for d in dist_strs:
        did = retokenize_answer(model, str(d)) or retokenize_answer(model, " " + str(d))
        if did is not None:
            dist_ids.append(did)
    if not dist_ids:
        return None
    return ans_id, dist_ids


@torch.no_grad()
def main():
    from transformer_lens import HookedTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] Pythia-160M on {device}")
    model = HookedTransformer.from_pretrained("EleutherAI/pythia-160m", device=device)
    model.eval()
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    print(f"  arch: {n_layers}L × {n_heads}H, d_model={model.cfg.d_model}")

    # peek dataset format
    pairs_all = load_pairs(os.path.join(DATASETS_DIR, "var_binding_tier1.jsonl"))
    print(f"[data] {len(pairs_all)} cf pairs total")
    print(f"  sample keys: {sorted(pairs_all[0]['clean'].keys())}")

    # Filter: matched token length under Pythia tokenizer + valid answer retokenization
    pairs_ok = []
    for p in pairs_all:
        try:
            clean_tok = model.to_tokens(p["clean"]["prompt"], prepend_bos=False)
            corr_tok = model.to_tokens(p["corrupt"]["prompt"], prepend_bos=False)
            if clean_tok.shape != corr_tok.shape:
                continue
            retok = retokenize_pair(model, p)
            if retok is None:
                continue
            ans_id, dist_ids = retok
            pairs_ok.append((p, clean_tok, corr_tok, ans_id, dist_ids))
        except Exception:
            continue
    print(f"  {len(pairs_ok)} pairs survive Pythia retokenization")
    pairs_ok = pairs_ok[:500]
    print(f"  using {len(pairs_ok)} for patching")

    if not pairs_ok:
        print("ERROR: no valid pairs after retokenization. Inspect dataset fields.")
        return

    # ============ Patching ============
    effects = np.full((len(pairs_ok), n_layers, n_heads), np.nan, dtype=np.float32)
    clean_lds, corr_lds = [], []
    import time
    t0 = time.time()
    for i, (p, ctok, xtok, ans_id, dist_ids) in enumerate(pairs_ok):
        ctok, xtok = ctok.to(device), xtok.to(device)
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
        if (i + 1) % 50 == 0:
            print(f"  [{i+1:>3}/{len(pairs_ok)}] {time.time()-t0:.0f}s")

    out_dir = os.path.join(RESULTS_DIR, "binding_pythia")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "head_effects.npy"), effects)

    mean_eff = np.nanmean(effects, axis=0)
    flat = [(float(mean_eff[L, H]), L, H) for L in range(n_layers) for H in range(n_heads)]
    flat.sort(key=lambda t: -t[0])
    print("\n=== Pythia-160M: top-10 binding heads ===")
    for m, L, H in flat[:10]:
        print(f"  L{L:>2}H{H:>2}  recovery={m:+.4f}")

    clean_mean = float(np.mean(clean_lds))
    corr_mean = float(np.mean(corr_lds))
    print(f"\nbaselines: clean_ld={clean_mean:+.4f}  corrupt_ld={corr_mean:+.4f}  "
          f"gap={clean_mean-corr_mean:+.4f}")

    # ============ Compare with GPT-2 top heads ============
    gpt2_top10 = [(10, 7), (10, 2), (6, 1), (3, 0), (10, 10),
                  (5, 0), (1, 11), (5, 8), (11, 10), (7, 11)]
    pythia_top10 = [(L, H) for _, L, H in flat[:10]]
    overlap = set(gpt2_top10) & set(pythia_top10)
    print(f"\n=== Cross-model top-10 overlap ===")
    print(f"  GPT-2 small top-10: {gpt2_top10}")
    print(f"  Pythia-160M top-10: {pythia_top10}")
    print(f"  Overlap (exact L,H coord): {len(overlap)}  → {sorted(overlap)}")
    print(f"  Random baseline: 10*10/144 ≈ 0.69")

    # Late-layer concentration
    layer_means = mean_eff.mean(axis=1)
    print(f"\n=== Per-layer mean recovery ===")
    for L in range(n_layers):
        print(f"  L{L:>2}: {layer_means[L]:+.4f}")

    # ============ Necessity test ============
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
        for (p, ctok, xtok, ans_id, dist_ids) in pairs_ok:
            ctok = ctok.to(device)
            logits = model.run_with_hooks(ctok, fwd_hooks=hooks)
            lds.append(logit_diff(logits[0, -1], ans_id, dist_ids))
        return float(np.mean(lds))

    def run_empty_attn():
        def make_hook(L):
            def hk(z, hook):
                z[:, :, :, :] = 0
                return z
            return hk
        hooks = [(f"blocks.{L}.attn.hook_z", make_hook(L)) for L in range(n_layers)]
        lds = []
        for (p, ctok, xtok, ans_id, dist_ids) in pairs_ok:
            ctok = ctok.to(device)
            logits = model.run_with_hooks(ctok, fwd_hooks=hooks)
            lds.append(logit_diff(logits[0, -1], ans_id, dist_ids))
        return float(np.mean(lds))

    print("\n[necessity] computing baselines...")
    empty_ld = run_empty_attn()
    print(f"  empty-attn ld = {empty_ld:+.4f}")
    print(f"  attention total contribution = {clean_mean - empty_ld:+.4f}")

    print("\n[necessity] ablating top-K vs random K (5 seeds)...")
    all_heads = [(L, H) for L in range(n_layers) for H in range(n_heads)]
    necessity = {"empty_ld": empty_ld, "clean_ld": clean_mean, "sets": {}}
    print(f"  {'set':>14}  {'k':>3}  {'ablated_ld':>10}  {'drop':>10}")
    for K in [3, 5, 10]:
        topK = pythia_top10[:K]
        m_top = run_remove(topK)
        d_top = clean_mean - m_top
        rdrops = []
        for s in range(5):
            rng = random.Random(s * 100 + K)
            rand = rng.sample(all_heads, K)
            m_r = run_remove(rand)
            rdrops.append(clean_mean - m_r)
        m_rd, s_rd = float(np.mean(rdrops)), float(np.std(rdrops))
        necessity["sets"][f"top{K}"] = {"k": K, "ablated_ld": m_top, "drop": d_top}
        necessity["sets"][f"random{K}"] = {"k": K, "mean_drop": m_rd, "std_drop": s_rd}
        print(f"  {'top'+str(K):>14}  {K:>3}  {m_top:+.4f}     {d_top:+.4f}")
        print(f"  {'random'+str(K):>14}  {K:>3}  {'—':>10}     {m_rd:+.4f} ± {s_rd:.4f}")

    # ============ Save & plot ============
    report = {
        "n_pairs": len(pairs_ok),
        "clean_ld": clean_mean,
        "corrupt_ld": corr_mean,
        "top10": [{"layer": L, "head": H, "recovery": m} for m, L, H in flat[:10]],
        "gpt2_top10": gpt2_top10,
        "overlap_with_gpt2_top10": sorted(list(overlap)),
        "overlap_count": len(overlap),
        "layer_means": layer_means.tolist(),
        "necessity": necessity,
    }
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    vmax = float(np.nanmax(np.abs(mean_eff)))
    im = ax.imshow(mean_eff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for L in range(n_layers):
        for H in range(n_heads):
            v = mean_eff[L, H]
            ax.text(H, L, f"{v:+.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > vmax * 0.5 else "black")
    # mark GPT-2 top-10 with green outline
    for (L, H) in gpt2_top10:
        ax.add_patch(plt.Rectangle((H - 0.5, L - 0.5), 1, 1, fill=False,
                                    edgecolor="lime", linewidth=1.5))
    ax.set_xlabel("head"); ax.set_ylabel("layer")
    ax.set_title(f"Pythia-160M: per-head patching recovery  (Tier 1, n={len(pairs_ok)})\n"
                 f"GPT-2 small top-10 outlined in green  |  overlap = {len(overlap)}/10")
    plt.colorbar(im, ax=ax, label="recovery fraction")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pythia_heatmap.png"), dpi=120)
    plt.close(fig)

    # necessity bar
    fig, ax = plt.subplots(figsize=(8, 5))
    Ks = [3, 5, 10]
    x = np.arange(len(Ks)); w = 0.35
    top_drops = [necessity["sets"][f"top{K}"]["drop"] for K in Ks]
    rand_drops = [necessity["sets"][f"random{K}"]["mean_drop"] for K in Ks]
    rand_stds = [necessity["sets"][f"random{K}"]["std_drop"] for K in Ks]
    ax.bar(x - w/2, top_drops, w, color="tab:red", label="ablate top-K (Pythia)")
    ax.bar(x + w/2, rand_drops, w, yerr=rand_stds, color="tab:gray",
           label="ablate random K (5 seeds)", capsize=4)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f"top{K}" for K in Ks])
    ax.set_ylabel("logit_diff drop")
    ax.set_title(f"Pythia-160M necessity test\n"
                 f"clean LD={clean_mean:+.3f}, empty-attn LD={empty_ld:+.3f}, "
                 f"attn total={clean_mean-empty_ld:+.3f}")
    for i, v in enumerate(top_drops):
        ax.text(i - w/2, v + 0.003, f"{v:+.3f}", ha="center", fontsize=9)
    for i, v in enumerate(rand_drops):
        ax.text(i + w/2, v + 0.003, f"{v:+.3f}", ha="center", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pythia_necessity.png"), dpi=120)
    plt.close(fig)
    print(f"\n[done] {time.time()-t0:.0f}s → {out_dir}")


if __name__ == "__main__":
    main()
