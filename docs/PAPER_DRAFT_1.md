# Paper Skeleton — 6-page NeurIPS-style workshop paper (double-blind)

Source of truth: `PROJECT_CANON.md` (2026-06-12). All run statistics are `[N90: <stat>]`
placeholders. No institution/author/run names. "We" throughout. Restatement-test target: *"measure
the frontier with learning progress, evolve toward it, feed the experience back."*

---

## Title candidates (from canon §1.4)

1. **Measure the Frontier, Don't Define It: Learning Progress as the Selection Signal for an Open-Ended Social Curriculum**
2. **Learning Progress Is the Social Frontier: Weight-Free Curriculum Generation for LLM Social Skill**
3. **Don't Define Social Difficulty — Measure It: Learning-Progress-Calibrated Scenario Evolution with In-Context Transfer**

---

## Abstract (full prose, ~0.25 pp)

> Improving a large language model's social skill requires experience on tasks at its frontier of
> competence, but social difficulty is continuous and model-relative, so the right tasks cannot be
> specified in advance. We propose to measure the frontier rather than define it: a scenario sits
> at a model's social frontier exactly when the model fails it cold yet improves across in-context
> retries. This learning-progress (LP) signal does triple duty — it operationally defines the
> frontier, selects which scenarios an open-ended evolutionary curriculum should expand, and
> certifies which experience is worth keeping. Generated scenarios carry their difficulty in a
> hidden, theory-grounded partner key that guarantees solvability by construction, so LP measures
> skill acquisition rather than noise on impossible tasks. Running the loop to a bank of 90
> generated dyadic scenarios, intermediate LP values are populated ([N90: LP histogram]),
> evidence that social difficulty has a learnable gradient rather than the pass/fail cliff of
> logical tasks. Insights extracted from the bank (ExpeL) are deployed as inference-time memory —
> no weight updates — and evaluated on held-out SOTOPIA scenarios against the same extraction
> applied to the raw seeds and a no-memory baseline ([N90: eval deltas / evaluation in progress]).

---

## 1. Introduction (~1.0 pp)

**Purpose:** establish the missing-signal problem, the vanishing-frontier problem, and state the
one idea; end with exactly three coupled contributions.

- **P1. Social competence cannot be improved from text corpora alone, because the feedback signal — how a real interlocutor would have responded to a different move — is unobservable in static text; the experience must be manufactured.**
  - Pretraining sees outcomes of conversations that happened, never counterfactual responses to the learner's own choices.
  - Interactive simulation (SOTOPIA-style dyadic goal-conflict) supplies the missing signal [CITE: zhou2024sotopia].
  - Existing routes update weights (fine-tuning, RL) [CITE: wang2024sotopiapi]; we ask what is achievable with the weights frozen.

- **P2. The harder obstacle is that nobody knows which scenarios to manufacture: a capable LLM's logical difficulty is a cliff, but its social difficulty is a slope — and where the slope sits is model-relative and unobservable a priori.**
  - For math/logic, capable models either solve a task or no amount of in-context retrying helps (bimodal).
  - Social tasks admit partial credit and visible improvement across retries — a gradient worth climbing.
  - You cannot predict which scenario a given model finds hard; predicted-difficulty curricula collapse (our own first system did; §6 Discussion).

- **P3. Our answer: stop predicting; measure. A scenario is at the model's frontier iff the model fails it cold but improves when allowed to learn in-context — learning progress is simultaneously the definition of the frontier, the curriculum's selection signal, and the certificate on experience worth keeping.**
  - LP as curiosity signal: drive toward maximal learnability [CITE: oudeyer2007] [CITE: schmidhuber2010].
  - A hidden, theory-grounded *partner key* guarantees each scenario is solvable in principle, the precondition that makes LP measure skill rather than noise — an instrument, like a calibrated thermometer, not the discovery.
  - ExpeL in-context extraction [CITE: zhao2024expel] is the pre-existing transfer mechanism we plug in.

- **P4. Contributions (exactly three, mirroring the coupling — none alone is new; the thread is):**
  - (1) LP as the operational, model-relative definition of the social frontier — converting the field's ill-definedness embarrassment into the method.
  - (2) An open-ended evolutionary curriculum (archive + Thompson selection + directed mutation) that uses LP as its sole selection currency, self-extending toward the measured frontier.
  - (3) Weight-free transfer: LP-certified experience, distilled by ExpeL, improves held-out social performance over the same mechanism on a fixed seed distribution — the controlled test of the *curriculum*, not the ICL machinery.

---

## 2. Related work (~0.75 pp) — one paragraph per thread cluster

**Purpose:** position against four clusters; compress canon §3 table into prose.

- **P1. Open-ended curricula and learning progress as curiosity.** OMNI/OMNI-EPIC generate tasks "learnable and interestingly novel" with FM-judged interestingness [CITE: zhang2023omni] [CITE: faldor2025omniepic]; LP as intrinsic motivation [CITE: oudeyer2007] [CITE: schmidhuber2010].
  - We replace physical RL environments with social scenarios and aesthetic-only interestingness with frontier-ness *measured* by LP.

- **P2. UED and quality-diversity.** PAIRED/PLR/ACCEL/MAESTRO ask which environments are worth training on, answering with predicted regret [CITE: dennis2020paired] [CITE: jiang2021plr] [CITE: parkerholder2022accel] [CITE: samvelyan2023maestro]; POET/MAP-Elites/MCC give archive-as-map and the minimal criterion [CITE: wang2019poet] [CITE: mouret2015mapelites] [CITE: brant2017mcc]; Darwin-Gödel-style parent selection rewarded by descendants' progress [CITE: zhang2025dgm].
  - We answer the UED question with *measured* LP, no adversarial game; our minimal criterion is positive LP (failed cold ∧ improved).

- **P3. The SOTOPIA ecosystem and in-context experiential learning.** SOTOPIA seeds and Sotopia-Eval [CITE: zhou2024sotopia]; SOTOPIA-π fine-tunes via QLoRA [CITE: wang2024sotopiapi]; SOTOPIA-RL exists as an arXiv preprint (withdrawn from ICLR 2026) and is cited as a preprint only, never as a peer-reviewed named baseline [CITE: yu2025sotopiarl]. Reflexion gives within-episode verbal RL; ExpeL gives cross-task insight extraction [CITE: shinn2023reflexion] [CITE: zhao2024expel].
  - We are the weight-free, self-calibrating alternative; ExpeL is applied to a self-generated frontier curriculum rather than a fixed task set.

- **P4. Simulation-validity critiques and deployment grounding.** LLM social evaluation is gameable by pushover partners and lenient judges [CITE: misleading2025] [CITE: secret2024]; LLM social world models have real failure modes [CITE: zhou2025swm] [CITE: sap2022] [CITE: ullman2023] [CITE: hu2025]. Roleplay-DoH legitimizes simulated practice partners for human skill training [CITE: louie2024] [CITE: louie2026].
  - Our key check, cross-lab judge, and planned human sub-eval are direct mitigations; the defensible deployment is AI-mediated practice, not real-user chat.

---

## 3. Method (~1.75 pp) — order is mandatory: LP first

**Purpose:** define LP and the frontier criterion before any machinery; everything else is
instrumentation in service of that measurement.

- **P1. The frontier criterion and the LP measure.** A scenario is at the learner's frontier iff it fails attempt 1 (cold) and has LP > 0 across in-context retries — a two-sided minimal criterion, neither trivial nor impossible.
  - LP = improved_votes / total_votes ∈ [0,1]; key-blind cross-lab judge compares attempt 1 vs each later attempt in both presentation orders; order-swap disagreement collapses to no_difference (kills position bias).
  - Judge floor: if neither attempt makes progress toward the *stated objective*, tone/rapport differences do not count → no_difference. **Anti-hypothesis bias note:** this floor only makes frontier classification *rarer*, never more common. [canon Patch 11 — describe neutrally, no patch names in prose]
  - Three bands: too_easy (solved attempt 1), frontier (failed, LP>0), beyond_frontier (failed, LP=0); the archive keeps all three — a map, not a trophy case.

- **P2. The loop.** Thompson-select an anchor from the archive (Beta posteriors over "did my children learn?"); operator chosen by the anchor's band; generate a fresh-surface child with a hidden partner key; admission gates; a K=4 episode loop with within-episode Reflexion (attempt 1 always cold); measure LP; classify; charge the anchor's posterior with LP pseudo-votes; add the child as a new arm; repeat to 90 completed scenarios.
  - **Figure callout — F-loop (Method figure 1):** redraw canon §2 diagram. *Caption draft:* "One curriculum iteration. Learning progress, measured by a key-blind pairwise judge over the K-attempt episode, both classifies the child and updates the anchor's selection posterior; the bank of completed scenarios is the curriculum's output."
  - Stationarity by design: attempt 1 is always cold and nothing transfers across scenarios during the run, so LP is a stationary property of (scenario × learner × reflexion) and is comparable across the whole run. **No claim of learner improvement over the run is made anywhere** (forbidden claim 1).
  - Seeds: 90 human-authored SOTOPIA scenarios, calibrated in a Phase-0 pass to warm-start priors (soft asymmetric Beta priors only; seed self-LP never becomes votes).

- **P3. The partner key: the instrument that makes LP well-defined.** LP only measures skill acquisition if the task is solvable in principle; the hidden partner key guarantees solvability by construction.
  - Role-asymmetric schema: learner holds an outcome/constraint/shortcut goal triple (Dillard primary/secondary goals; Brown–Levinson face-threatening act) [CITE: dillard1989] [CITE: brown1987]; partner holds a natural-language goal plus the hidden key.
  - Key fields: key_mechanism ∈ {reactance, face_needs, validation_before_change, procedural_voice, reciprocity_disclosure} — a closed, auditable, swappable library with **no completeness claim** (forbidden claim 5); movement_conditions in sensor form ("something shifts when the learner does X"), never demand form; hardening_triggers; surface_misdirection (the discoverability dial); cost_coupling (satisfying the key must make the learner's own goal harder, never unreachable).
  - **The standardized-patient sentence (appears here only):** the keyed partner is a standardized patient — medical education settled decades ago that calibrated simulation beats uncalibrated reality *for training*, because the script controls what the trainee encounters and makes what they did verifiable; ground truth lives in the script, not the actor.
  - Key check (separate temperature-0, fail-closed judge, every attempt): passed iff ≥1 movement condition genuinely satisfied (semantically, not by mention) ∧ no unrepaired hardening trigger — where merely *offering* a proscribed move counts as tripping the trigger. **Anti-hypothesis bias note:** this rule only makes *solved* rarer; it guards against rewarding extraction-by-pushover. Conditions are a finite authored list, so a genuinely effective unlisted lever scores as a false negative (conservative; see Limitations).

- **P4. Admission gates (one paragraph; Methods detail).** Single-axis ownership, fail-closed: coherence (key valid, never leaks, cost real); surface novelty (deterministic no-clone check + embedding diversity vs the whole archive, threshold 0.92); worth (MOI as a *ranker* over a 3-candidate batch — a ranker cannot saturate to "yes"); learnability is never gated by prediction — it is measured by LP downstream.

- **P5. Mutation operators (one paragraph; Methods detail).** Direction-setters only, under a universal fresh-surface mandate (new names, setting, stakes; keyless-seed parents get a key invented from scratch): escalate (parent too_easy → harder), relax (parent beyond_frontier → easier along the diagnosed dimension), lateral (parent frontier → same difficulty, different mechanism). No slot-preservation contracts: operators set direction; **LP verifies where the child landed**, and a mis-aimed mutation simply charges its anchor.

- **P6. Two success labels, stated explicitly.** Curriculum-internal *solved* = GOAL≥7 ∧ REL≥0 ∧ key_check_passed (integrity guard: on a keyed scenario, GOAL≥7 with no condition met usually means the partner broke discipline). Extraction and evaluation success = GOAL≥7 ∧ REL≥0 **only**, keeping Generated90 and Base90 trajectory labels identical and keeping the finite-condition false-negative risk out of every reported result.

---

## 4. Experimental setup (~0.75 pp)

**Purpose:** canon §7 in spirit — the headline experiment tests the loop, not the key.

- **P1. Claim under test and conditions.** *An open-ended, LP-calibrated social curriculum yields in-context experience that improves a frozen LLM's held-out social performance more than the same ICL mechanism applied to a fixed seed distribution — no weight updates.* Conditions: (1) Vanilla (no memory); (2) ExpeL-Base90 (insights from the raw 90 seeds); (3) ExpeL-Generated90 (insights from our 90-scenario bank). Random90 is a separate workstream; SOTOPIA-π is a reference point for what fine-tuning achieves, **not a comparison target** (forbidden claim 4).
  - **Falsifiability statement (required):** the experiment tests the *loop*, not the key. If ExpeL-Generated90 does not exceed ExpeL-Base90 on deltas-vs-Vanilla over held-out scenarios, the claim that LP-calibrated curricula produce more transferable experience than a fixed distribution is falsified — regardless of how well-calibrated the bank's internal LP distribution looks.

- **P2. Protocol invariants.** One frozen partner engine across conditions; eval partners are vanilla — no key, no memory — because the claim is transfer to *standard* tasks; cross-lab judge (different lab than the learner, breaking the self-evaluation monoculture); results reported as **deltas vs. Vanilla**, cancelling judge leniency and partner agreeableness in one subtraction; contamination check (ID + embedding overlap, seeds/bank vs eval set) [N90: overlap stats]; per-condition compute accounting (the curriculum's budget is a strict superset of Base90's — stated, not hidden).

---

## 5. Results (~1.0 pp) — F1→F5 + T1, each with one claim sentence

**Purpose:** the curriculum measurements carry the section; the held-out eval slot reads
"evaluation in progress" if numbers are unavailable (planned framing).

- **F1 — LP distribution (the thesis figure).** Claim: intermediate LP bars are populated — social difficulty has a learnable gradient, not a cliff.
  - *Caption draft:* "Learning-progress distribution over the [N90: n_frontier + n_beyond] generated scenarios that failed attempt 1. LP is discrete with four values {0, 0.33, 0.67, 1.0} (K=4 attempts → 3 comparisons × 2 order-swapped votes); four labeled bars, not a density. Populated middle bars indicate a learnable gradient, in contrast to the solved/unsolved cliff capable LLMs exhibit on logical tasks. [If Base90 LP available: side-by-side, framed as 'populated under harder, keyed conditions' — never 'ours is higher.']"

- **F2 — Classification over curriculum iteration.** Claim: curriculum adaptation — beyond-heavy early (escalation off trivially easy seeds), the relax cycle converting lineages back toward the frontier.
  - *Caption draft:* "Cumulative band counts vs. **curriculum iteration** (x-axis label exactly this, never 'time'); frontier-solved events as dots. The figure shows the curriculum adapting; because attempt 1 is always cold and nothing transfers across scenarios, it does not and cannot show learner improvement." (forbidden claim 1 enforced in caption)

- **F3 — Per-operator direction table (3×3).** Claim: the measured form of the direction claim — escalate skews harder than lateral, relax easier — **supported at the aggregate level only** (forbidden claim 2).
  - *Caption draft:* "Operator × resulting-band counts [N90: per_operator_classification_counts]. Surfaces vary freely under mutation, so direction is an aggregate property of operators, not a per-pair causal claim."

- **F4 — Goal-trajectory case studies.** Claim: the bands are concrete conversational behavior.
  - *Caption draft:* "Per-attempt GOAL for one scenario per band. The frontier exhibit ([N90: donor-pledge lineage, GOAL 2→3→6→6]) plateaus within a single approach — the within-episode ceiling that motivates cross-scenario insight extraction."

- **F5 — Surface diversity.** Claim: the fresh-surface mandate works; no clone lobe.
  - *Caption draft:* "Parent–child embedding cosine similarity [N90: parent_child_cosine histogram], optionally per operator. Absence of mass near the 0.92 admission threshold indicates mutations produce genuinely new scenarios, not near-duplicates."

- **T1 — Extraction yield (bridge to the eval).** Claim: the curriculum produced usable raw material.
  - *Caption draft:* "Extraction yield: [N90: n_success_tasks] success trajectories, [N90: n_compare_pairs] compare pairs, [N90: n_frontier_unsolved] frontier-but-unsolved scenarios. The curriculum produced [N90: n_compare_pairs] compare pairs for ExpeL extraction."

- **P-final. Held-out evaluation.** [N90: 3-condition deltas-vs-Vanilla table] — if unavailable at submission: "evaluation in progress; the curriculum measurements above are the run-derived, evaluation-independent evidence."
  - Caption hygiene throughout: note the judge-version boundary where relevant (reproducibility).

---

## 6. Discussion & limitations (~0.5 pp)

**Purpose:** name the sharpest threat ourselves; present the two internal findings as findings,
not apologies; close with the five verbatim limitation sentences.

- **P1. The generic-prompt ablation, named openly (canon §8.6).** A hand-written prompt encoding the five mechanism descriptions might match ExpeL-Generated90 on held-out eval; if so, the curriculum's value reduces to prior override (suppressing RLHF agreeableness) rather than skill acquisition through experience.
  - Defense-in-framing: the contribution is automated *curriculum design* (which scenarios to train on), not discovery of social psychology; transfer to unkeyed held-out partners is the non-tautological step; the ablation is named as the immediate next experiment.

- **P2. The LP-vs-success gap, as a finding (canon §8.3).** Frontier-by-LP scenarios can improve without ever reaching GOAL≥7 ([N90: n_frontier_unsolved]) — curriculum-valuable but extraction-poor; if frontier were dominated by near-zero-goal near-misses, that is a generator calibration finding, not a bug.

- **P3. The escalate-overshoot finding (canon §8.4).** Escalation off too_easy parents frequently overshoots to beyond_frontier ([N90: escalate band counts]) — partly the genuinely high keyed success bar, partly a step-size effect of trivially easy seeds; the lineage cycle self-corrects (beyond → relax walks back; a depth-2 relax child landed frontier). We deliberately did not let the generator *predict* "just above the frontier" — that is the prediction relapse this architecture exists to avoid; future fixes are selection-side.

- **P4. Limitations — the five sentences, verbatim (canon §8.7), written with the same confidence as the results:**
  1. "Because mutation operators vary the scenario surface freely, individual parent–child pairs do not support ceteris-paribus attribution of difficulty changes to specific key edits; the direction claim is therefore supported at the aggregate level (per-operator classification distributions, Table T3/F3) rather than per-pair."
  2. "Our difficulty and worth constructs are theory-stipulated and LLM-judged; human validation of both is the necessary next step." (forbidden claim 3 honored)
  3. "The condition-based key check can only credit solution paths the generator enumerated; a learner satisfying the underlying mechanism through a novel, unlisted move is scored as a false negative. This conservative bias understates success rates and never inflates them — and it is confined to curriculum-internal bookkeeping (§5, two success labels): extraction and evaluation use GOAL≥7 ∧ REL≥0 only."
  4. "Results are from a single curriculum run; multi-seed variance runs were deferred (Henderson et al., 2018)." [CITE: henderson2018]
  5. "A critical ablation we have not yet run is whether a hand-authored prompt encoding the five mechanism descriptions matches the curriculum-trained agent's held-out performance; if so, the curriculum's value would reduce to prior override rather than skill acquisition through experience."

---

## 7. Conclusion (~0.25 pp max)

- **P1. One paragraph: restate the one idea and its coupling.** Social difficulty cannot be specified in advance, but it can be measured: learning progress operationally defines the frontier, drives an open-ended curriculum toward it, and certifies the experience that transfers back at inference time — one currency for difficulty, selection, and transfer, with the model's weights untouched.

---

## Bibliography (BibTeX-style; `% VERIFY` = entry not independently verified)

```bibtex
@article{oudeyer2007,
  author={Oudeyer, Pierre-Yves and Kaplan, Fr{\'e}d{\'e}ric and Hafner, Verena V.},
  title={Intrinsic Motivation Systems for Autonomous Mental Development},
  journal={IEEE Transactions on Evolutionary Computation}, year={2007}}

@article{schmidhuber2010,
  author={Schmidhuber, J{\"u}rgen},
  title={Formal Theory of Creativity, Fun, and Intrinsic Motivation (1990--2010)},
  journal={IEEE Transactions on Autonomous Mental Development}, year={2010}}

@inproceedings{zhang2023omni,  % VERIFY year/venue
  author={Zhang, Jenny and Lehman, Joel and Stanley, Kenneth O. and Clune, Jeff},
  title={OMNI: Open-endedness via Models of Human Notions of Interestingness},
  year={2024}, note={ICLR}}  % VERIFY

@article{faldor2025omniepic,  % VERIFY venue/year
  author={Faldor, Maxence and Zhang, Jenny and Cully, Antoine and Clune, Jeff},
  title={OMNI-EPIC: Open-endedness via Models of Human Notions of Interestingness with Environments Programmed in Code},
  year={2025}}

@article{wang2019poet,  % VERIFY venue
  author={Wang, Rui and Lehman, Joel and Clune, Jeff and Stanley, Kenneth O.},
  title={Paired Open-Ended Trailblazer (POET): Endlessly Generating Increasingly Complex and Diverse Learning Environments and Their Solutions},
  journal={arXiv preprint}, year={2019}}

@article{mouret2015mapelites,
  author={Mouret, Jean-Baptiste and Clune, Jeff},
  title={Illuminating Search Spaces by Mapping Elites},
  journal={arXiv preprint arXiv:1504.04909}, year={2015}}

@article{lehman2011novelty,  % VERIFY exact venue
  author={Lehman, Joel and Stanley, Kenneth O.},
  title={Abandoning Objectives: Evolution Through the Search for Novelty Alone},
  journal={Evolutionary Computation}, year={2011}}

@inproceedings{brant2017mcc,  % VERIFY
  author={Brant, Jonathan C. and Stanley, Kenneth O.},
  title={Minimal Criterion Coevolution: A New Approach to Open-Ended Search},
  booktitle={GECCO}, year={2017}}

@inproceedings{dennis2020paired,
  author={Dennis, Michael and others},
  title={Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design},
  booktitle={NeurIPS}, year={2020}}

@inproceedings{jiang2021plr,  % VERIFY
  author={Jiang, Minqi and Grefenstette, Edward and Rockt{\"a}schel, Tim},
  title={Prioritized Level Replay},
  booktitle={ICML}, year={2021}}

@inproceedings{parkerholder2022accel,  % VERIFY title/venue
  author={Parker-Holder, Jack and others},
  title={Evolving Curricula with Regret-Based Environment Design},
  booktitle={ICML}, year={2022}}

@inproceedings{samvelyan2023maestro,  % VERIFY
  author={Samvelyan, Mikayel and others},
  title={MAESTRO: Open-Ended Environment Design for Multi-Agent Reinforcement Learning},
  booktitle={ICLR}, year={2023}}

@article{zhang2025dgm,  % VERIFY — canon names only "Darwin-G{\"o}del-style self-improvement"
  author={TODO},
  title={Darwin G{\"o}del Machine: Open-Ended Evolution of Self-Improving Agents},
  year={2025}}

@inproceedings{zhou2024sotopia,
  author={Zhou, Xuhui and others},
  title={SOTOPIA: Interactive Evaluation for Social Intelligence in Language Agents},
  booktitle={ICLR}, year={2024}}

@inproceedings{wang2024sotopiapi,  % VERIFY venue
  author={Wang, Ruiyi and others},
  title={SOTOPIA-$\pi$: Interactive Learning of Socially Intelligent Language Agents},
  booktitle={ACL}, year={2024}}

@misc{yu2025sotopiarl,
  author={Yu, TODO and others},  % VERIFY authors
  title={SOTOPIA-RL: TODO full title},
  note={arXiv preprint; withdrawn from ICLR 2026 — cite as preprint only, never as a
        peer-reviewed named baseline (canon \S3)}, year={2025}}  % VERIFY

@inproceedings{shinn2023reflexion,
  author={Shinn, Noah and others},
  title={Reflexion: Language Agents with Verbal Reinforcement Learning},
  booktitle={NeurIPS}, year={2023}}

@inproceedings{zhao2024expel,
  author={Zhao, Andrew and others},
  title={ExpeL: LLM Agents Are Experiential Learners},
  booktitle={AAAI}, year={2024}}

@article{zhou2025swm,  % VERIFY — canon: "Zhou, Sap et al. 2025 (Social World Models)"
  author={Zhou, Xuhui and Sap, Maarten and others},
  title={Social World Models (TODO full title)}, year={2025}}

@inproceedings{sap2022,  % VERIFY title
  author={Sap, Maarten and others},
  title={Neural Theory-of-Mind? On the Limits of Social Intelligence in Large LMs},
  booktitle={EMNLP}, year={2022}}

@article{ullman2023,  % VERIFY
  author={Ullman, Tomer},
  title={Large Language Models Fail on Trivial Alterations to Theory-of-Mind Tasks},
  journal={arXiv preprint}, year={2023}}

@article{hu2025,  % VERIFY — canon: "Hu/Sosa/Ullman 2025"
  author={Hu, TODO and Sosa, TODO and Ullman, Tomer},
  title={TODO}, year={2025}}

@book{brown1987,
  author={Brown, Penelope and Levinson, Stephen C.},
  title={Politeness: Some Universals in Language Usage},
  publisher={Cambridge University Press}, year={1987}}

@article{dillard1989,  % VERIFY co-authors
  author={Dillard, James Price and Segrin, Chris and Harden, Janie M.},
  title={Primary and Secondary Goals in the Production of Interpersonal Influence Messages},
  journal={Communication Monographs}, year={1989}}

@article{misleading2025,  % VERIFY authors/venue
  author={TODO},
  title={The Misleading Success of Simulating Social Intelligence in LLMs},
  year={2025}}

@inproceedings{secret2024,  % VERIFY — likely Mireshghallah et al., ConfAIde
  author={Mireshghallah, Niloofar and others},
  title={Can LLMs Keep a Secret? Testing Privacy Implications of Language Models via Contextual Integrity Theory},
  booktitle={ICLR}, year={2024}}

@inproceedings{louie2024,  % VERIFY venue
  author={Louie, Ryan and others},
  title={Roleplay-doh: Enabling Domain-Experts to Create LLM-Simulated Patients via Eliciting and Adhering to Principles},
  booktitle={EMNLP}, year={2024}}

@article{louie2026,  % VERIFY — canon: "Louie et al. 2026 (counselor upskilling)"
  author={Louie, Ryan and others},
  title={TODO (counselor upskilling)}, year={2026}}

@inproceedings{henderson2018,
  author={Henderson, Peter and others},
  title={Deep Reinforcement Learning that Matters},
  booktitle={AAAI}, year={2018}}
```

---

## Pre-return checklist (all satisfied in this skeleton)

- [x] Falsifiability statement (§4 P1): tests the loop; named falsifying outcome.
- [x] Anti-hypothesis bias notes at the LP floor (§3 P1) and key check (§3 P3).
- [x] Every run statistic as `[N90: <stat>]`.
- [x] Double-blind: no names, no run identifiers, "we" only; patch numbers flagged as not-for-prose.
- [x] F1 = four discrete LP bars; F2 x-axis = "curriculum iteration."
- [x] Standardized-patient sentence once, in Method §3 P3 only.
- [x] Forbidden claims 1–6: enforced at F2 caption, F3 caption, limitation 2, §4 P1 (π reference-only), §3 P3 (no completeness claim), bibliography note on SOTOPIA-RL.
- [x] Restatement test: the skeleton's spine is LP-measurement → evolution → transfer; key/gates/operators appear only inside Method as instrumentation.