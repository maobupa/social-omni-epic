"""Run once to produce ablation_vs_quality.ipynb — then delete this file."""
import json, uuid, pathlib

def _id():
    return str(uuid.uuid4())[:8]

def code(src: str) -> dict:
    lines = src.lstrip("\n").rstrip()
    parts = lines.split("\n")
    source = [l + "\n" for l in parts[:-1]] + [parts[-1]]
    return {"cell_type": "code", "execution_count": None,
            "id": _id(), "metadata": {}, "outputs": [], "source": source}

def md(src: str) -> dict:
    lines = src.lstrip("\n").rstrip()
    parts = lines.split("\n")
    source = [l + "\n" for l in parts[:-1]] + [parts[-1]]
    return {"cell_type": "markdown", "id": _id(), "metadata": {}, "source": source}

# ─────────────────────────────────────────────────────────────────────────────
cells = []

# ── Title ────────────────────────────────────────────────────────────────────
cells.append(md(r"""
# VS Ablation: `SYSTEM_PROMPT` vs `VS_SYSTEM_PROMPT`

Generates **50 scenarios per condition** conditioned on the original 90 Sotopia seeds,
then judges quality with the **MoI gate** (novel AND learnable).

| Condition | Generator call |
|-----------|---------------|
| **No-VS** | `task_gen.generate_from_archive(examples)` — plain `SYSTEM_PROMPT` |
| **VS**    | `task_gen.generate_with_verbalized_sampling(examples, n_candidates=3)` — `VS_SYSTEM_PROMPT` |

**Part 1** — MoI pass rate, generation cost, intra-condition diversity
**Part 2** — Seed-pair dedup audit (context selection vs anchor selection)
"""))

# ── Cell 1: Config & imports ─────────────────────────────────────────────────
cells.append(code(r"""
import sys, os, json, time
from pathlib import Path

# Locate project root whether kernel is started from project/ or project/notebook/
project_root = Path.cwd()
if project_root.name == "notebook":
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from social_omni_epic.fm import FM
from social_omni_epic.seeds import load_sotopia_seeds
from social_omni_epic.task_generator import TaskGenerator
from social_omni_epic.model_of_interestingness import ModelOfInterestingness
from social_omni_epic.embedding_utils import get_similar_scenarios

# ── tuneable ──────────────────────────────────────────────────────────────────
MODEL           = "gpt-4o-mini"
N_TRIALS        = 50      # scenarios to generate per condition
N_CTX_EXAMPLES  = 3       # KNN context examples shown to the generator
VS_N_CANDIDATES = 3       # VS candidates per trial (3 reduces cost vs default 5)
SEEDS_PATH      = project_root / "data" / "sotopia_90_seeds.jsonl"
RESULTS_DIR     = Path("results_vs_ablation")
RANDOM_SEED     = 42
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR.mkdir(exist_ok=True)
np.random.seed(RANDOM_SEED)

fm       = FM(model=MODEL)
moi_gate = ModelOfInterestingness(fm, num_examples=5)
task_gen = TaskGenerator(fm, num_examples=N_CTX_EXAMPLES, max_retries=3)
print(f"FM model : {MODEL}")
print(f"Trials   : {N_TRIALS} per condition")
print(f"VS cands : {VS_N_CANDIDATES}")
"""))

# ── Cell 2: Load seeds + embed ────────────────────────────────────────────────
cells.append(code(r"""
EMBED_CACHE = RESULTS_DIR / "seed_embeddings.json"

seeds = load_sotopia_seeds(str(SEEDS_PATH), both_perspectives=True)
print(f"Loaded {len(seeds)} seed entries  ({len(seeds) // 2} source scenarios × 2 perspectives)")

if EMBED_CACHE.exists():
    print("Loading cached seed embeddings...")
    with open(EMBED_CACHE) as f:
        cached = json.load(f)
    emb_by_id = {e["id"]: e["embedding"] for e in cached}
    for s in seeds:
        s.embedding = emb_by_id.get(s.id)
    print(f"  {sum(1 for s in seeds if s.embedding)} embeddings restored")
else:
    print("Embedding seeds (~30 s for 180 entries)...")
    texts = [s.to_text_for_embedding() for s in seeds]
    all_embs: list = []
    for i in tqdm(range(0, len(texts), 50), desc="embed batches"):
        all_embs.extend(fm.get_embeddings(texts[i : i + 50]))
    for s, e in zip(seeds, all_embs):
        s.embedding = e
    with open(EMBED_CACHE, "w") as f:
        json.dump([{"id": s.id, "embedding": s.embedding} for s in seeds], f)
    print("  saved to cache")

seed_embs       = [s.embedding for s in seeds]
seed_source_ids = [s.source_scenario_id for s in seeds]
seed_agent_idxs = [s.target_agent_idx for s in seeds]
existing_types  = sorted({s.interaction_type for s in seeds if s.interaction_type})
print(f"\nSeed archive : {len(seeds)} entries")
print(f"Unique types : {len(existing_types)}")
"""))

# ── Cell 3: Context-selection helper ─────────────────────────────────────────
cells.append(code(r"""
def select_context_examples(trial_idx: int):
    """Pick a random seed anchor (deterministic per trial) then KNN neighbours.
    Both conditions use the same anchor per trial for a controlled comparison."""
    rng = np.random.RandomState(RANDOM_SEED + trial_idx)
    anchor_idx = int(rng.choice(len(seeds)))
    anchor_emb = seeds[anchor_idx].embedding
    idxs = get_similar_scenarios(
        anchor_emb, seed_embs, num_returns=N_CTX_EXAMPLES,
        source_ids=seed_source_ids,
        agent_idxs=seed_agent_idxs,
        preferred_agent_idx=seeds[anchor_idx].target_agent_idx,
    )
    return [seeds[i] for i in idxs], anchor_idx

# Quick sanity check
ex, anc = select_context_examples(0)
print(f"Trial-0 anchor : seed[{anc}]  id={seeds[anc].id[:40]}...")
print(f"Context types  : {[e.interaction_type for e in ex]}")
print(f"Context src_ids: {[e.source_scenario_id[:12] for e in ex]}")
"""))

# ── Cell 4: Generate No-VS (checkpointed) ────────────────────────────────────
cells.append(md("## Part 1 — Generation"))
cells.append(code(r"""
NO_VS_PATH = RESULTS_DIR / "generated_no_vs.jsonl"

if NO_VS_PATH.exists():
    print(f"Loading existing No-VS results from {NO_VS_PATH}")
    with open(NO_VS_PATH) as f:
        results_no_vs = [json.loads(l) for l in f if l.strip()]
    n_ok = sum(1 for r in results_no_vs if r["status"] == "ok")
    print(f"  {len(results_no_vs)} records  ({n_ok} ok)")
else:
    results_no_vs = []
    with open(NO_VS_PATH, "w") as fout:
        for trial in tqdm(range(N_TRIALS), desc="No-VS generation"):
            examples, anchor_idx = select_context_examples(trial)
            t0 = time.time()
            scenario = task_gen.generate_from_archive(examples, existing_types=existing_types)
            elapsed = time.time() - t0
            rec = {
                "trial":      trial,
                "anchor_idx": anchor_idx,
                "elapsed_s":  round(elapsed, 2),
                "scenario":   scenario.model_dump() if scenario else None,
                "status":     "ok" if scenario else "failed",
            }
            results_no_vs.append(rec)
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
    n_ok = sum(1 for r in results_no_vs if r["status"] == "ok")
    print(f"No-VS done: {n_ok}/{N_TRIALS} generated")
"""))

# ── Cell 5: Generate VS (checkpointed) ───────────────────────────────────────
cells.append(code(r"""
VS_PATH = RESULTS_DIR / "generated_vs.jsonl"

if VS_PATH.exists():
    print(f"Loading existing VS results from {VS_PATH}")
    with open(VS_PATH) as f:
        results_vs = [json.loads(l) for l in f if l.strip()]
    n_ok = sum(1 for r in results_vs if r["status"] == "ok")
    print(f"  {len(results_vs)} records  ({n_ok} ok)")
else:
    results_vs = []
    with open(VS_PATH, "w") as fout:
        for trial in tqdm(range(N_TRIALS), desc="VS generation"):
            examples, anchor_idx = select_context_examples(trial)
            t0 = time.time()
            scenario = task_gen.generate_with_verbalized_sampling(
                examples, existing_types=existing_types, n_candidates=VS_N_CANDIDATES
            )
            elapsed = time.time() - t0
            rec = {
                "trial":      trial,
                "anchor_idx": anchor_idx,
                "elapsed_s":  round(elapsed, 2),
                "scenario":   scenario.model_dump() if scenario else None,
                "status":     "ok" if scenario else "failed",
            }
            results_vs.append(rec)
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
    n_ok = sum(1 for r in results_vs if r["status"] == "ok")
    print(f"VS done: {n_ok}/{N_TRIALS} generated")
"""))

# ── Cell 6: Embed generated scenarios (checkpointed) ─────────────────────────
cells.append(code(r"""
from social_omni_epic.data_models import SocialScenario as _SS

GEN_EMBED_CACHE = RESULTS_DIR / "generated_embeddings.json"
gen_emb_cache: dict = {}

if GEN_EMBED_CACHE.exists():
    with open(GEN_EMBED_CACHE) as f:
        gen_emb_cache = json.load(f)
    print(f"Loaded {len(gen_emb_cache)} cached generated embeddings")

to_embed: dict[str, str] = {}
for cond, results in [("no_vs", results_no_vs), ("vs", results_vs)]:
    for r in results:
        if r["status"] == "ok":
            key = f"{cond}_{r['trial']}"
            if key not in gen_emb_cache:
                to_embed[key] = _SS(**r["scenario"]).to_text_for_embedding()

if to_embed:
    keys  = list(to_embed.keys())
    texts = [to_embed[k] for k in keys]
    print(f"Embedding {len(texts)} new scenarios...")
    new_embs: list = []
    for i in tqdm(range(0, len(texts), 50), desc="embed"):
        new_embs.extend(fm.get_embeddings(texts[i : i + 50]))
    for k, e in zip(keys, new_embs):
        gen_emb_cache[k] = e
    with open(GEN_EMBED_CACHE, "w") as f:
        json.dump(gen_emb_cache, f)
    print("Saved.")
else:
    print("All generated embeddings already cached.")
"""))

# ── Cell 7: MoI evaluation (checkpointed) ────────────────────────────────────
cells.append(code(r"""
MOI_CACHE = RESULTS_DIR / "moi_results.json"
moi_results: dict = {}

if MOI_CACHE.exists():
    with open(MOI_CACHE) as f:
        moi_results = json.load(f)
    print(f"Loaded {len(moi_results)} cached MoI results")

to_eval: dict = {}
for cond, results in [("no_vs", results_no_vs), ("vs", results_vs)]:
    for r in results:
        if r["status"] == "ok":
            key = f"{cond}_{r['trial']}"
            if key not in moi_results and key in gen_emb_cache:
                to_eval[key] = r

if to_eval:
    print(f"Running MoI on {len(to_eval)} scenarios (temperature=0.3)...")
    for key, r in tqdm(to_eval.items(), desc="MoI"):
        emb = gen_emb_cache[key]
        sim_idxs = get_similar_scenarios(
            emb, seed_embs, num_returns=5,
            source_ids=seed_source_ids,
            agent_idxs=seed_agent_idxs,
        )
        similar  = [seeds[i] for i in sim_idxs]
        scenario = _SS(**r["scenario"])
        scenario.embedding = emb
        passed, reason = moi_gate.evaluate(scenario, similar)
        moi_results[key] = {"passed": passed, "reason": reason}
    with open(MOI_CACHE, "w") as f:
        json.dump(moi_results, f, indent=2)
    print("Saved.")
else:
    print("All MoI results already cached.")
"""))

# ── Cell 8: Assemble results DataFrame ───────────────────────────────────────
cells.append(md("## Part 1 — Results"))
cells.append(code(r"""
COND_MAP = [
    ("No-VS  (SYSTEM_PROMPT)",  "no_vs", results_no_vs),
    ("VS  (VS_SYSTEM_PROMPT)",  "vs",    results_vs),
]

rows = []
for cond_label, cond_key, results in COND_MAP:
    for r in results:
        key = f"{cond_key}_{r['trial']}"
        mr  = moi_results.get(key, {})
        rows.append({
            "condition":        cond_label,
            "trial":            r["trial"],
            "elapsed_s":        r["elapsed_s"],
            "generated":        r["status"] == "ok",
            "moi_passed":       mr.get("passed"),
            "moi_reason":       mr.get("reason", ""),
            "interaction_type": (r["scenario"] or {}).get("interaction_type", ""),
            "scenario_snippet": ((r["scenario"] or {}).get("scenario", ""))[:130],
        })

df = pd.DataFrame(rows)

summary = (
    df.groupby("condition")
    .agg(
        n_trials      =("trial",      "count"),
        gen_ok_rate   =("generated",  "mean"),
        moi_pass_rate =("moi_passed", lambda x: x.dropna().mean()),
        avg_elapsed_s =("elapsed_s",  "mean"),
    )
    .reset_index()
)
for col, fmt in [("gen_ok_rate", "{:.1%}"), ("moi_pass_rate", "{:.1%}"), ("avg_elapsed_s", "{:.1f}s")]:
    summary[col] = summary[col].map(fmt.format)

print("=== Summary ===")
display(summary)
"""))

# ── Cell 9: Bar charts ────────────────────────────────────────────────────────
cells.append(code(r"""
COLORS      = ["#5B8DB8", "#E07B54"]
COND_LABELS = ["No VS", "With VS"]
COND_KEYS   = ["No-VS  (SYSTEM_PROMPT)", "VS  (VS_SYSTEM_PROMPT)"]

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# MoI pass rate
pass_rates = [df[df.condition == c]["moi_passed"].dropna().mean() * 100 for c in COND_KEYS]
axes[0].bar(COND_LABELS, pass_rates, color=COLORS, width=0.45, zorder=2)
for i, v in enumerate(pass_rates):
    axes[0].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
axes[0].set_ylim(0, 108)
axes[0].set_ylabel("MoI pass rate (%)")
axes[0].set_title("Quality (MoI gate)")
axes[0].grid(axis="y", alpha=0.3, zorder=1)

# Generation success rate
gen_rates = [df[df.condition == c]["generated"].mean() * 100 for c in COND_KEYS]
axes[1].bar(COND_LABELS, gen_rates, color=COLORS, width=0.45, zorder=2)
for i, v in enumerate(gen_rates):
    axes[1].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
axes[1].set_ylim(0, 108)
axes[1].set_ylabel("Generation success rate (%)")
axes[1].set_title("JSON parse success rate")
axes[1].grid(axis="y", alpha=0.3, zorder=1)

# Avg generation time
times = [df[df.condition == c]["elapsed_s"].mean() for c in COND_KEYS]
axes[2].bar(COND_LABELS, times, color=COLORS, width=0.45, zorder=2)
for i, v in enumerate(times):
    axes[2].text(i, v + 0.2, f"{v:.1f}s", ha="center", fontsize=11, fontweight="bold")
axes[2].set_ylabel("Avg time per scenario (s)")
axes[2].set_title("Generation cost")
axes[2].grid(axis="y", alpha=0.3, zorder=1)

plt.suptitle("VS Ablation — 50 trials × 2 conditions", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "vs_ablation_quality.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

# ── Cell 10: Diversity ────────────────────────────────────────────────────────
cells.append(code(r"""
def pairwise_avg_cosine_dist(keys: list) -> float:
    embs = np.array([gen_emb_cache[k] for k in keys if k in gen_emb_cache], dtype=float)
    if len(embs) < 2:
        return float("nan")
    normed = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    sim_mat = normed @ normed.T
    n = len(embs)
    upper = sim_mat[np.triu_indices(n, k=1)]
    return float(1.0 - upper.mean())   # cosine distance

keys_no_vs = [f"no_vs_{r['trial']}" for r in results_no_vs if r["status"] == "ok"]
keys_vs    = [f"vs_{r['trial']}"    for r in results_vs    if r["status"] == "ok"]

div_no_vs = pairwise_avg_cosine_dist(keys_no_vs)
div_vs    = pairwise_avg_cosine_dist(keys_vs)

print("Avg pairwise cosine distance (higher = more diverse within condition):")
print(f"  No-VS : {div_no_vs:.4f}")
print(f"  VS    : {div_vs:.4f}")

fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(COND_LABELS, [div_no_vs, div_vs], color=COLORS, width=0.4, zorder=2)
for i, v in enumerate([div_no_vs, div_vs]):
    ax.text(i, v + 0.0005, f"{v:.4f}", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Avg pairwise cosine distance")
ax.set_title("Intra-condition Embedding Diversity")
ax.set_ylim(0, max(div_no_vs, div_vs) * 1.2)
ax.grid(axis="y", alpha=0.3, zorder=1)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "vs_ablation_diversity.png", dpi=150)
plt.show()
"""))

# ── Cell 11: Sample passed/failed scenarios ───────────────────────────────────
cells.append(code(r"""
for cond_label, cond_key in [("No-VS", "No-VS  (SYSTEM_PROMPT)"), ("VS", "VS  (VS_SYSTEM_PROMPT)")]:
    subset = df[(df.condition == cond_key) & (df.moi_passed == True)]
    print(f"\n{'='*70}")
    print(f"PASSED MoI — {cond_label}  ({len(subset)} / {N_TRIALS})")
    print("="*70)
    for _, row in subset.head(3).iterrows():
        print(f"\n  [Trial {row['trial']}] {row['interaction_type']}")
        print(f"  Scenario: {row['scenario_snippet']}...")
        print(f"  MoI reason: {row['moi_reason'][:280]}")
"""))

# ── Part 2 header ─────────────────────────────────────────────────────────────
cells.append(md(r"""
---

## Part 2 — Seed-Pair Dedup Audit

Seeds are loaded with `both_perspectives=True` → 90 source scenarios × 2 = **180 entries**.
`{env_pk}_p0` and `{env_pk}_p1` share `source_scenario_id = env_pk` but differ in `target_agent_idx`.

Two distinct places where sibling pairs might cause redundancy:

| Where | Mechanism | Expected |
|-------|-----------|----------|
| **Context selection** (`get_similar_scenarios`) | `source_ids` dedup arg | Should **NOT** return both `_p0` and `_p1` in the same context window |
| **Anchor selection** (UCB1) | None — intentional | **CAN** select both; they are distinct learning tasks (different learner role) |
"""))

# ── Cell 12: Context-selection dedup audit ───────────────────────────────────
cells.append(code(r"""
# ── 2a: Context selection dedup ───────────────────────────────────────────────
from collections import Counter

violations = []
for trial in range(N_TRIALS):
    examples, _ = select_context_examples(trial)
    src_ids = [e.source_scenario_id for e in examples]
    dups = [sid for sid, cnt in Counter(src_ids).items() if cnt > 1]
    if dups:
        violations.append({"trial": trial, "dup_src_ids": dups})

print(f"Trials checked   : {N_TRIALS}")
print(f"Dedup violations : {len(violations)}  (expected 0)")
if violations:
    for v in violations[:5]:
        print(f"  trial {v['trial']}: dup source_ids = {v['dup_src_ids']}")
else:
    print()
    print("PASS  get_similar_scenarios correctly excludes sibling perspectives")
    print("      from context examples — source_ids dedup works as designed.")
"""))

# ── Cell 13: UCB1 anchor-selection behavior ───────────────────────────────────
cells.append(code(r"""
# ── 2b: UCB1 anchor selection — show that both perspectives CAN be selected ──
import tempfile, shutil
from social_omni_epic.archive import Archive

tmpdir = tempfile.mkdtemp()
try:
    seed_arch = Archive(checkpoint_dir=tmpdir)
    for s in seeds:
        seed_arch.add_successful(s)

    N_SIM = 20   # simulate 20 UCB1 anchor picks
    sel_history = []
    for step in range(N_SIM):
        idx  = seed_arch.ucb1_select()
        task = seed_arch.state.successful[idx]
        sel_history.append({
            "step":               step,
            "archive_idx":        idx,
            "id":                 task.id,
            "source_id_short":    task.source_scenario_id[:18] + "...",
            "target_agent_idx":   task.target_agent_idx,
        })
        seed_arch.record_selection(idx, step)
finally:
    shutil.rmtree(tmpdir)

df_sel = pd.DataFrame(sel_history)

# Flag rows whose source_id appeared more than once in the window
src_counts_sim = Counter(df_sel["source_id_short"])
df_sel["sibling_also_selected"] = df_sel["source_id_short"].map(lambda s: src_counts_sim[s] > 1)

print(f"First {N_SIM} UCB1 selections from {len(seeds)}-entry seed archive:\n")
print(df_sel[["step", "id", "target_agent_idx", "sibling_also_selected"]].to_string(index=False))

n_both = sum(1 for c in src_counts_sim.values() if c > 1)
print(f"\nSource scenarios with BOTH perspectives selected in {N_SIM} steps: {n_both}")
print()
print("Design note:")
print("  _p0 / _p1 are distinct learning tasks — UCB1 selecting both is intentional.")
print("  Prompt-level redundancy (both in context window) is already prevented by")
print("  get_similar_scenarios (see 2a above).")
"""))

# ─────────────────────────────────────────────────────────────────────────────
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out = pathlib.Path(__file__).parent / "ablation_vs_quality.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Written: {out}  ({out.stat().st_size // 1024} KB)")
