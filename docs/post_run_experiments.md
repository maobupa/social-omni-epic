# Post-Run Experiments & Analysis Plan

*Companion to `PROJECT_CANON.md` and `PAPER_DRAFT_1.md`. Queues the experiments and figures that
turn the gen90 run from "we built a system" into "we learned something." Nothing here is on the
critical path of the frozen Gen-90 run; it is the analysis + one ablation that follow it.*

---

## 1. The three-tier success logic (read this first — it resolves "what proves success?")

There is no single success metric. There are **three claims in a hierarchy**, each with its own
evidence, cost, and risk. Be explicit in the paper about which tier each figure supports.

| Tier | Claim | Evidence | Cost | Role in the paper |
|---|---|---|---|---|
| **1. Mechanism** | the curriculum self-organizes toward the frontier | **intrinsic** bank metrics (LP distribution, frontier-fraction over iteration, coverage, lineage stats) | free — already logged | **safest headline**; evaluation-independent |
| **2. Selection-value** | LP-weighted selection beats non-adaptive selection | **intrinsic comparison** of the Thompson bank vs a **random-anchor** bank | one extra gen run | the ablation that makes "self-organizing" *measured*, not asserted |
| **3. Transfer** | the bank's experience improves held-out performance | **downstream** ExpeL ICL: Generated90 vs Base90 vs Vanilla | expensive, noisy | end-to-end value; the aspiration |

**The methodological glue (this is the important move):** compare the cheap-to-make Thompson
**variants on Tier-1 intrinsic metrics** (high-resolution, directly shows what selection does to
bank composition). Use the expensive Tier-3 downstream eval **only for the headline** (Generated90
vs Base90). Then establish the **intrinsic→transfer link once** on that headline comparison —
Generated90 has more frontier/coverage than Base90 *and* transfers better — which **licenses
treating the intrinsic metric as a valid proxy** for the variant comparisons. Do **not**
downstream-eval every variant: a single run has too little statistical power to separate selection
variants through judge + partner noise.

---

## 2. The Thompson ablation menu (four orthogonal degrees of freedom)

Conditions are not ad hoc — they are cells of "hold everything fixed, vary one DOF." Breadth-vs-
depth is **not** a separate traversal; it is the *emergent consequence of DOF 2 (credit
assignment)*.

| DOF | Variants | Hypothesis tested | N=90 priority |
|---|---|---|---|
| **1. Selection rule** | Thompson · **random/uniform** · greedy · UCB | does value-weighted selection beat non-adaptive? | **DO (random)** |
| **2. Credit assignment** | parent-only (current) · child-also · lineage-discounted | is value at the *producer* or the *frontier node*? (breadth vs depth) | MEDIUM — predictable at N=90 |
| **3. Reward signal** | LP pseudo-votes (current) · binary frontier-hit · coverage-gain | is LP the right fitness? | LOW |
| **4. Granularity** | per-node (current) · niche-first two-level | does enforcing coverage help? | **DEFER (needs larger N)** |

**Why this prioritization at N=90:**
- **Random (DOF 1) is the one to run.** It is the foundational "does selection do anything?" test,
  cheap (a flag — see §4), and it converts the central "self-organizing curriculum" claim from
  asserted to measured.
- **Reward-child (DOF 2) is mostly predictable at N=90.** A depth-leaning policy on a 90-budget
  produces a few deep lineages with poor coverage, which almost certainly loses on coverage and
  transfer. Running it mainly *justifies the breadth-leaning choice* rather than revealing
  something new — so it belongs in Future Work, not the headline.
- **Niche-first (DOF 4) needs a larger bank** (the coverage-vs-depth tradeoff below). Defer.

---

## 3. The random-anchor ablation — exact protocol

**Run:** a second generation run, identical config and seeds, `anchor_selection=random` (§4).
Stop at the same 90 completed scenarios. Different `run_name` (e.g. `gen90_random`).

**Compare (intrinsic only — do NOT downstream-eval it):**
- frontier-fraction and its trend over iteration (Thompson should reach a higher steady-state
  frontier-fraction if selection helps);
- LP-yield (mean LP over completed scenarios);
- structural coverage (niche count / UMAP spread of the abstract `social_dynamic |
  target_perspective` embeddings);
- extraction yield (n success trajectories, n compare-pairs).

**Reading the result (both outcomes are publishable):**
- Thompson > random on frontier-yield/coverage → **validates** "self-organizing curriculum";
  selection is an active ingredient.
- Thompson ≈ random → **honest finding**: the generator + operators + LP-classification carry it;
  reframe the contribution from "self-organizing" toward "structured generation + LP measurement."
  Still a contribution; *not knowing* is the only bad outcome.

---

## 4. Figures & analysis the paper needs (beyond the headline transfer table)

Mapped to the tier each supports. F1–F5/T1 already scoped in `PAPER_DRAFT_1`; the **bold** items
are the additions this plan calls for.

| Fig | Content | Tier | Status |
|---|---|---|---|
| F1 | LP distribution — four discrete bars {0, .33, .67, 1.0} | 1 | drafted — the thesis figure (slope, not cliff) |
| F2 | classification (frontier/too_easy/beyond) over **curriculum iteration** | 1 | drafted — the self-organization figure |
| F3 | per-operator → resulting-band table (3×3) | 1 | drafted — the measured direction claim (aggregate only) |
| F4 | goal-trajectory case studies, one per band | 1 | drafted — bands as concrete behavior |
| F5 | parent–child embedding cosine (surface diversity) | 1 | drafted — fresh-surface works |
| **F6** | **structural coverage UMAP** — seeds + bank in abstract `social_dynamic\|perspective` space, colored by band | 1 | **add** — the QD "map not trophy case" figure; also the coverage axis for §3 |
| **F7** | **Thompson vs random** — frontier-yield + coverage bars (from §3) | 2 | **add iff random run done** — the selection-value ablation |
| **F8 (opt)** | **selection concentration** — selections-per-anchor / lineage tree colored by productivity | 1 | optional — shows Thompson concentrating on productive anchors |
| T1 | extraction yield (success trajs, compare-pairs, frontier-unsolved) | 1→3 bridge | drafted |
| T2 | **headline transfer** — Generated90 vs Base90 vs Vanilla, deltas-vs-Vanilla | 3 | the end-to-end claim (may read "in progress") |

Minimum honest workshop set: **F1, F2, F3, F5, T1** (Tier 1, all free) **+ T2** (Tier 3, headline).
Strongly recommended additions: **F6** (coverage) and **F7** (random ablation) — together they are
what let you *claim the loop and selection matter*, not just that a bank was produced.

---

## 5. Coverage-vs-depth at N=90 — acknowledge, don't apologize

At N=90 the budget forces a **coverage-vs-depth tradeoff**: you cannot have both long evolved
lineages *and* broad structural coverage with only 90 scenarios. The current reward-parent design
leans toward **breadth/coverage**, which is the *correct* choice for a small bank — diverse
frontier scenarios transfer better than one deeply-evolved lineage. State this explicitly; it
converts an apparent limitation into a motivated design decision and seeds Future Work:
- **Future work A:** depth-vs-breadth (DOF 2) at larger N, where deep lineages can coexist with
  coverage.
- **Future work B:** niche-aware two-level Thompson (DOF 4) at larger N, for explicit coverage
  guarantees.

---

## 6. Retrieval refinement (non-Thompson) — title-based exemplar KNN

Exemplar retrieval for the generation prompt currently keys on the **full-text** embedding
(`to_text_for_embedding`: scenario + goals), which mixes surface and structure and can retrieve
topic-matches rather than structural analogues. A cleaner alternative for the *exemplar* path
(not the diversity gate, which correctly needs surface) is to KNN on the abstract
`scenario_title` = `social_dynamic | target_perspective` embedding — archived scenarios already
carry titles, so this is feasible. **Principled refinement, modest expected impact** (exemplars
are secondary to the anchor, and fresh-surface already works) → post-run, not on the frozen run.

---

## 7. Priority order

1. **Tier-1 intrinsic analysis of the gen90 bank** (F1, F2, F3, F4, F5, F6, T1) — free, do first.
2. **Headline downstream eval** (T2: Generated90 vs Base90 vs Vanilla) — establishes the
   intrinsic→transfer link.
3. **Random-anchor ablation** (`gen90_random`, §3) → F7 — the one ablation that matters.
4. **Future work (paper text only):** reward-child (DOF 2), niche-first (DOF 4), title-retrieval
   (§6), larger-N depth+coverage.
