# Anchor Selection: Hierarchical Thompson Sampling

> **What this document covers:** how the curriculum runner picks which archive entry to generate the next scenario from, and why.

---

## The problem

The archive holds scenario anchors — scenarios that can be used as seeds to generate new training scenarios. At each iteration, we need to pick one anchor to generate from. The question is: which one?

We want to pick anchors that are **generatively productive** — anchors from which the two-loop calibration is likely to produce a scenario that bites (fails at least once) and then solves. Not all anchors are equally productive. Some social dynamics are rich enough to support calibrated difficulty; others tend to produce scenarios that are either too easy (discarded immediately) or unsolvable (agent never succeeds after K attempts).

The challenge: we don't know in advance which anchors are productive. We have to learn this from experience, while also making sure we don't fixate on a small number of good-looking anchors and ignore the rest.

---

## The mechanism: Thompson Sampling

Each anchor is modelled as having an unknown "productivity rate" — the probability that generating from it produces a solved-after-biting scenario. We don't know this rate upfront; we estimate it from experience.

**The intuition (no maths):** imagine each anchor has a reputation card that starts blank. Every time we generate from it:
- If the result is `solved_after_biting`: mark a ✓
- Otherwise: mark a ✗

Before each iteration, we look at every anchor's card and *make a guess* about how productive it is. The guess is random, but informed by the card:
- Blank card → wild guess, could be anything
- Mostly ✓s → guess clusters high
- Mostly ✗s → guess clusters low

We then pick whichever anchor guesses highest that round.

This naturally handles the exploration-exploitation tradeoff. Anchors with blank or sparse cards occasionally guess very high purely by chance, so they keep getting explored. Anchors with strong track records consistently guess high, so they get exploited. No hyperparameter controls this balance — it emerges from the uncertainty in the guesses.

**Technically:** each anchor maintains a Beta distribution over its productivity rate. At selection time, we sample once from each Beta and pick the argmax. The Beta is updated after each generation attempt.

---

## Hierarchical priors: why generated children aren't blank slates

Original SOTOPIA seeds start with a flat prior: Beta(1, 1), meaning total uncertainty — the seed could be productive or not.

Generated children (scenarios that entered the archive because they were `solved_after_biting`) start differently. Their parent anchor already has a track record. If the parent has produced 3 successful children out of 5 attempts, that's structural evidence: the social dynamic it represents supports difficulty-calibrated scenarios. A child generated from that parent inherits this social dynamic and its calibrated structure, so it is *a priori* more likely to be productive as an anchor itself.

We encode this by initialising the child's Beta prior to the parent's current posterior at the moment of the child's creation:

```
child.prior_alpha = parent.prior_alpha + parent.n_solved
child.prior_beta  = parent.prior_beta  + (parent.n_i - parent.n_solved)
```

The child then updates this inherited prior as its own evidence accumulates. This is called **empirical Bayes**: use observed data from the parent to set the child's prior, rather than assuming every entry is equally unknown.

**Why this matters at our scale:** with a target of N=60 solved scenarios spread across 90+ anchors, each anchor gets very few selection opportunities. A flat prior takes many trials to update meaningfully. The hierarchical prior front-loads structural knowledge from the parent's track record, which is especially important when evidence per anchor is sparse.

---

## What the Beta distribution looks like concretely

| Anchor state | Beta shape | Implication |
|---|---|---|
| Never selected (n_i=0), flat prior | Beta(1, 1) = uniform | Any productivity rate equally plausible |
| 0 successes out of 3 tries | Beta(1, 4) | Skewed low; unlikely to guess high |
| 1 success out of 3 tries | Beta(2, 3) | Moderate; wide uncertainty |
| 3 successes out of 4 tries | Beta(4, 2) | Skewed high; usually guesses high |
| Child of a 3/4 parent, never tried | Beta(4, 2) | Starts where parent left off |

---

## Archive policy by terminal state

| Terminal state | Effect on parent | Effect on archive |
|---|---|---|
| `solved_after_biting` | n_i ↑, n_solved ↑, n_children ↑ | Scenario added as new anchor; inherits parent's posterior as its prior |
| `failed` (bit, never solved) | n_i ↑, n_children ↑ | Scenario added to `failed_tasks` only; does not become an anchor |
| `discarded` (never bit) | n_i ↑ only | Not archived anywhere; n_children does NOT increment |

Note: discards do not increment `n_children` because a discarded scenario was too easy to be a valid curriculum example — it provides no signal about the parent's generative quality at the right difficulty level.

---

## How this replaces UCB1

The previous mechanism used:

```
score = C * sqrt(ln(N) / n_i) - D * n_children
```

This had three problems:
1. **No reward signal.** The formula only measures how often an anchor has been tried, not whether trying it produced useful outcomes. Productive and unproductive anchors were treated identically as long as they had the same n_i.
2. **Sequential bias.** All entries with n_i=0 scored infinity; numpy's argmax returns the first infinity, so all 90 original seeds were visited in strict JSONL index order before any generated child was ever selected.
3. **Two tunable constants (C, D)** with no principled way to set them.

Thompson Sampling eliminates all three problems: the reward signal (n_solved / n_i) is baked in, ties among unseen entries are broken by random sampling from Beta(1,1), and there are no constants to tune.

---

## Implementation locations

| File | Change |
|---|---|
| `social_omni_epic/data_models.py` | Added `n_solved`, `prior_alpha`, `prior_beta` to `SocialScenario` |
| `social_omni_epic/archive.py` | Added `thompson_select()`, `record_solved_child()`, `child_prior_from_parent()` |
| `scripts/run_phase2.py` | `_select_anchor_and_examples` calls `thompson_select`; solved branch sets child prior before archiving |
| `configs/social_omni_epic_phase2.yaml` | Removed `ucb1_C`, `ucb1_D` |
