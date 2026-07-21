# Contrastive Analysis: What Makes SOTOPIA Scenarios Hard for Vanilla gpt-5-mini?

**Method:** For each category with mixed pass/fail outcomes, the LLM was shown up to 5 failed and 3 passed episodes side-by-side and asked to identify structural properties that explain the difference.

## Key Insight

Vanilla LLMs are competent at isolated, single‑turn, low‑risk social moves but systematically fail when interactions require multi‑turn strategy, explicit trade‑offs across multiple social objectives, or nuanced modeling of an interlocutor’s resistance, emotions, or hidden information. The core weakness is not language fluency but limited sequential social planning, opponent/partner modeling, and pragmatic calibration under multiple constraints.

## Difficulty Taxonomy

### Multi-turn strategic planning  `[high]`

**Definition:** Requires sequencing adaptive conversational moves over several turns (phased concessions, scaffolding, follow‑ups, repair) rather than a single utterance; success depends on turn-by-turn contingency and long‑horizon coherence.

**Why hard for vanilla:** Vanilla LLMs tend to produce locally coherent single responses but lack explicit planning and a memory/strategy over multiple turns; they repeat tactics, fail to schedule concessions, and cannot reliably anticipate downstream effects of early moves.

**Categories:** normbank, social_chemistry, social_iqa, hand-craft, deal-or-no-deal, original_creation, social_dynamics, inspired_by_prompt, persuation_for_good, generated, user_generated, family_dynamics, original_content, custom_scenario, original_prompt, craigslist_bargains, mutual_friends, inspired_prompt

### Multi-objective & simultaneous-constraint optimization  `[high]`

**Definition:** Requires satisfying several independent and sometimes competing soft or hard constraints at once (instrumental outcome + relationship preservation + secrecy + technical criteria).

**Why hard for vanilla:** Models struggle to represent and trade off multiple objectives explicitly; they often optimize for a single salient goal and fail to generate compromises or layered proposals that hedge across constraints.

**Categories:** normbank, social_chemistry, social_iqa, hand-craft, original_creation, deal-or-no-deal, persuation_for_good, original_content, user_generated, family_dynamics, original_prompt, custom_scenario

### Partner rigidity / low cooperativeness  `[high]`

**Definition:** Targets who hold firm preferences, principled objections, or explicit non-negotiable positions, requiring tailored persuasion, reframing, or multi-step credibility-building.

**Why hard for vanilla:** LLMs under-model interlocutor resistance and tend to default to generic persuasion or direct requests rather than adaptive reframing, graded concessions, or evidence-building that address principled refusal.

**Categories:** normbank, social_chemistry, social_iqa, deal-or-no-deal, craigslist_bargains, original_creation, social_dynamics, persuation_for_good, generated, user_generated, inspired_prompt, original_content, original_prompt

### High emotional / safety stakes and trust repair  `[high]`

**Definition:** Situations where missteps cause large, possibly lasting emotional harm or physical risk, and where repair, validation, de‑escalation, or explicit safety planning is required.

**Why hard for vanilla:** Models often underuse explicit empathic scaffolding, de‑escalation language, and safety‑first contingencies; they may give direct advice that threatens rapport or miss cues that require pacing and repair.

**Categories:** social_chemistry, social_iqa, social_dynamics, family_dynamics, inspired_by_prompt, generated, user_generated, original_prompt, custom_scenario

### Information elicitation & hypothesis‑testing  `[high]`

**Definition:** Requires targeted probing, integrating negative evidence, planning decision trees (diagnostic questions), and updating hypotheses about hidden motives or attributes.

**Why hard for vanilla:** LLMs tend to either assume missing details or ask undiagnostic questions; they lack systematic short decision‑tree construction and reliable integration of negative responses over turns.

**Categories:** mutual_friends, craigslist_bargains, family_dynamics, generated, inspired_by_prompt, original_creation, hand-craft, custom_scenario

### Concealment, controlled signaling & indirect persuasion  `[medium]`

**Definition:** Requires hiding intent or modulating signals (indirect phrasing, hints, plausible deniability) while still eliciting cooperating moves or information without alerting the partner.

**Why hard for vanilla:** Models are prone to explicit, literal wording; they struggle to produce naturalistic indirect speech acts, plausible pretexts, or subtle hedges that preserve concealment and rapport.

**Categories:** normbank, hand-craft, custom_scenario, craigslist_bargains, original_creation, generated

### Naturalistic pragmatics & low‑pressure phrasing  `[medium]`

**Definition:** Requires giving conversationally plausible, non‑interrogative, low‑pressure language (casual elicitation, face‑saving wording, polite closures) that preserves believability and rapport.

**Why hard for vanilla:** The model often produces terser, more transactional or overly explicit language that breaks natural cadence and can come off as coercive, robotic, or implausible in everyday dialogue.

**Categories:** hand-craft, mutual_friends, craigslist_bargains, user_generated, generated, original_creation

### Combinatorial & domain‑specific reasoning  `[medium]`

**Definition:** Requires arithmetic, allocation search, technical fixes, or concrete implementable trade-offs (splitting items, scheduling, technical compromises) rather than abstract persuasion.

**Why hard for vanilla:** Vanilla LLMs make calculation errors, fail to search combinatorial spaces, and produce vague or impractical domain solutions instead of concrete, operational plans or creative side‑payments.

**Categories:** deal-or-no-deal, original_content, persuation_for_good, craigslist_bargains, original_creation

### Ambiguous success criteria & subjective goals  `[medium]`

**Definition:** When the reward or acceptability of outcomes is soft/subjective (not sounding controlling, preserving warmth) and multiple acceptable endpoints exist, requiring calibration and meta-social judgment.

**Why hard for vanilla:** Models lack a reliable internal metric for subjective rapport outcomes and therefore cannot calibrate tone or tradeoffs to maximize subtle social goals; they default to heuristics that may miss stakeholder subtleties.

**Categories:** normbank, social_chemistry, social_dynamics, original_creation, user_generated, generated

## Model Failure Profile

**Handles well:**
- Single‑turn direct requests with aligned incentives
- Low‑stakes informational answers and factual explanations
- Brief polite expressions (thanks, condolences) or scheduling with clear constraints
- One‑shot transactional offers where success is immediately verifiable

**Systematically fails:**
- Extended multi‑turn negotiations requiring phased concessions or staged persuasion
- Scenarios with competing objectives (instrumental outcome vs relationship preservation) and many simultaneous constraints
- Interactions with rigid or principled partners that need reframing, evidence, or credibility‑building
- Tasks requiring covert intent, subtle pragmatic phrasing, hypothesis testing, or combinatorial/domain‑specific tradeoffs

## Implication for Curriculum / Training

Prioritize multi‑turn, interactive training that teaches planning over dialogues (scaffolding, concession schedules, repair moves), paired with modules for partner modeling (resistance, values, hidden motives) and pragmatic hedging (indirect requests, rapport‑preserving language). Also include domain and combinatorial problem solving (allocation, technical fixes) and safety/empathy drills so the model learns to balance instrumental goals with emotional and ethical constraints.

---

## Per-Category Findings

### normbank  (n=51, pass=67%)

**Hard scenario signature:** A hard scenario pits the agent's immediate objective directly against a partner's entrenched preference or requires hiding motives/emotions while preserving rapport, with success depending on multi-turn negotiation and satisfying several soft constraints simultaneously.

**Easy scenario signature:** An easy scenario has aligned incentives or a single clear request with low emotional risk, permitting short, direct moves that produce immediate and unambiguous feedback or mutual benefit.

**What the model lacks:** The model struggles with multi-constraint social optimization: balancing instrumental goals against relational costs, concealing or modulating internal states, and sustaining a negotiated pathway across multiple turns. It also lacks fine-grained pragmatic and emotional calibration needed for ambiguous or high-stakes interpersonal tradeoffs.

**Difficulty factors:**

- **High goal–relationship tradeoff** — Scenarios where the agent's objective directly conflicts with the partner's preference or risks harming the relationship require subtle balancing—pushing too hard achieves the goal but damages rapport, while being too conciliatory fails the goal.
  - *In failed cases:* Insisting on a thriller vs friend preferences, convincing a friend to stick to a strict diet, and persuading someone to prefer traditional pizza toppings all create direct tension between winning the argument and preserving the friendship.
  - *In passed cases:* Passed scenarios (flirting, asking a roommate to lower noise, making a date special) either align the agent's short-term goal with relationship building or have clearly prosocial framing, so pursuing the goal typically benefits the relationship rather than endangering it.
- **Multiple simultaneous constraints / conditional success** — Tasks that require satisfying several constraints at once (e.g., persuade, avoid sounding controlling, and keep a secret) are harder because success depends on hitting several soft conditions rather than a single clear action.
  - *In failed cases:* Several fails required convincing without appearing controlling (diet), creating distance while hiding romantic feelings (comforting-but-distant), or persuading while respecting a colleague's craft (Keurig vs manual coffee).
  - *In passed cases:* Passed cases typically have a single, clear actionable goal (flirt, request quiet, create a special date) with fewer competing constraints to juggle simultaneously.
- **High emotional stakes / need to conceal internal states** — When the agent must manage intense emotions or conceal internal motives, the interaction needs nuanced emotional signaling and careful pacing to avoid leaks or social missteps.
  - *In failed cases:* Episodes involving hiding romantic feelings while distancing or responding to a crying friend require careful emotional management; failing to modulate empathy vs distance is costly.
  - *In passed cases:* Passed scenarios either openly display positive emotion (flirting, romantic date) or involve a pragmatic request with low emotional risk (noise complaint), reducing the need for concealment.
- **Partner rigidity / strong entrenched preferences** — Hard cases involve an interlocutor with strong, principled preferences that are unlikely to be swayed by a single message; persuasion requires gradual framing, evidence, or trade-offs.
  - *In failed cases:* The coffee purist and the diet-committed or pizza-traditionalist partners are presented as ideologically committed, so changing their mind needs more sustained rationale or compromise.
  - *In passed cases:* Passed interactions feature partners with weaker resistance or clear incentives to comply (roommate who needs quiet for a meeting, someone receptive on a first meet), so one or two well-placed utterances suffice.
- **Need for sustained back-and-forth and closure** — Scenarios that require multi-turn negotiation, iterative concessions, or an explicit wrap-up are harder because they demand planning over several conversational moves and maintaining engagement until resolution.
  - *In failed cases:* Failed items often required negotiation and a mutually acceptable compromise (coffee machine, diet, pizza) or avoiding abrupt exits; leaving before closure was noted as a social error.
  - *In passed cases:* Passed cases typically succeed with a short sequence: a clear opening plus a direct request or an easily interpretable romantic move that produces immediate feedback.
- **Ambiguous success criteria / soft, subjective goals** — When success depends on subjective judgments (not coming off as controlling, preserving friendship warmth while persuading), evaluating and optimizing behavior is less straightforward and harder to execute.
  - *In failed cases:* Goals like 'don't be too controlling' or 'distance without revealing feelings' are inherently vague and require fine-grained tonal control, which the scenarios demanded.
  - *In passed cases:* Passed scenarios have concrete, observable outcomes (lower noise, a successful flirt, making a date memorable) that are easier to pursue with clear actions.

### social_chemistry  (n=35, pass=69%)

**Hard scenario signature:** A hard scenario pits the learner's goal against a partner's clear preference or a fragile relationship, requires multiple conditional agreements or iterative persuasion, and carries emotional/reputational risk if handled poorly.

**Easy scenario signature:** An easy scenario is a single, low‑risk request with aligned incentives or clear authority/constraints, producing straightforward compliance or a simple expressive move with minimal negotiation.

**What the model lacks:** The model struggles with multi‑turn negotiation strategies and calibrating emotional tone when stakes are high: it does not reliably sequence concessions, anticipate or defuse strong partner reactions, or trade short‑term gains for long‑term relationship maintenance. It also underweights the need for explicit empathy and repair language when delivering disclosures or pressing on contentious points.

**Difficulty factors:**

- **high emotional risk / reputational stakes** — Scenarios in which the learner's action threatens trust, intimacy, or a person's self‑image are harder because the partner's reaction can be strong, variable, and long‑lasting, requiring careful tone, timing, and repair strategies.
  - *In failed cases:* Failed 3 (admit affair) requires delivering damaging personal information; Failed 1 and 5 (pressing for more game time / more classes) risk irritating or alienating a close friend/sibling.
  - *In passed cases:* Passed cases involve low‑risk requests or clear institutional/transactional contexts (FaceTime texting complaint is a single emotional complaint but low risk; samples/garage sale are low‑stakes or rule-driven).
- **goal conflict / opposing incentives** — When the partner's preferences or constraints directly conflict with the learner's goal, success requires negotiation, tradeoffs, or convincing someone who benefits from refusal.
  - *In failed cases:* Failed 2 (weekend trip) and Failed 1 (friend tired/early commitment) feature explicit conflicting preferences or schedules that the partner defends.
  - *In passed cases:* Passed 3 (manager told to clear samples) and Passed 2 (selling at a garage sale) present aligned incentives or clear authority/urgency that make compliance straightforward.
- **multi‑condition / multi‑step success criteria** — Scenarios requiring multiple things to happen (concessions, future promises, price negotiation, scheduling coordination) increase complexity because each condition is a potential failure point.
  - *In failed cases:* Failed 4 (sell play rights) needs agreement on price, terms, and fit; Failed 1 needs a five‑minute concession plus follow‑through to keep the friend agreeable.
  - *In passed cases:* Passed cases generally require a single compliant action (give away samples, accept a sale, or make a single statement about feelings).
- **sustained back‑and‑forth persuasion** — Scenarios that demand iterative persuasion, managing repeated objections, or long negotiations are harder because they require adaptive strategies and phased concessions.
  - *In failed cases:* Failed 2 shows repetitive pushing and diminishing relationship returns; Failed 4 implies bargaining over numbers and licensing terms.
  - *In passed cases:* Passed scenarios typically involve one clear request or a straightforward compliance moment with minimal pushback.
- **relational complexity and fragility** — New or sensitive relationships (new romance, close family) magnify the consequences of missteps and require more nuanced empathy and calibration of disclosures.
  - *In failed cases:* Failed 3 is a new romantic relationship where honesty can create major doubt; Failed 5 involves a sibling relationship where pressure can be perceived as controlling.
  - *In passed cases:* Passed examples are either transactional (garage sale, samples) or an established couple addressing a minor annoyance, reducing fragility.
- **ambiguity in acceptable outcomes** — When success has many subjective definitions (fair price, convincing without harming relationship, 'challenging' a friend academically), it's harder because the learner must both achieve the goal and navigate relational norms.
  - *In failed cases:* Failed 4 (fair price) and Failed 5 (what counts as 'challenging yourself' academically) lack a single clear metric for success.
  - *In passed cases:* Passed tasks have clear, objective outcomes (give away samples, complete sale, express a feeling about a clear behavior).

### social_iqa  (n=32, pass=69%)

**Hard scenario signature:** A hard scenario combines moral/ethical tension or illicit goals with an uncooperative partner and multiple simultaneous constraints (e.g., reveal X but hide Y), requiring adaptive, varied persuasion over many turns while protecting the relationship and controlling partner inferences.

**Easy scenario signature:** An easy scenario involves a benign, cooperative goal with clear mutual benefits or a single simple condition; success can be achieved with one or two natural, transparent moves and minimal risk to the relationship.

**What the model lacks:** The model struggles with multi-step persuasive planning and adaptive conversational tactics: it tends to repeat the same move rather than vary strategy, and it fails to maintain multiple constraints (e.g., keep anonymity while disclosing content). It also has difficulty managing high-stakes ethical trade-offs and preserving relational capital after risky requests.

**Difficulty factors:**

- **High ethical/moral stakes** — Scenarios that require accomplishing goals which conflict with moral or legal norms (confessing to change a death sentence, injuring someone, or sustaining an anonymous wrongdoing) raise internal constraints that limit legitimate persuasive moves and force trade-offs that are hard to navigate without explicit policy/ethical reasoning.
  - *In failed cases:* Failed 4 (persuade to confess that could alter execution) and Failed 5 (injure a third person) directly involve ethically fraught aims; Failed 1 also requires revealing a secret while preserving anonymity, an ethically delicate maneuver.
  - *In passed cases:* Passed cases (share blanket, drive/show car, share bed) involve benign, socially acceptable goals with no moral/legal conflict, so the agent can pursue straightforward cooperative strategies.
- **Goal conflict / partner unwillingness (low cooperativeness)** — When the partner is resistant, rigid, or morally opposed, success requires extended, tailored persuasion and contingency handling rather than a single direct request.
  - *In failed cases:* Failed 4 explicitly involves an unwilling inmate; Failed 3 and 1 show partners who either accept then need follow‑up care or who require subtle cues to avoid revealing identity; Failed 2 involved a friend withholding info and needing negotiation tactics.
  - *In passed cases:* Passed scenarios assume or quickly establish cooperative alignment (friend expects to let you drive, mutual coldness encourages blanket-sharing, existing intimacy on vacation makes bed-sharing natural), so a single ask or simple benefit framing suffices.
- **Multi-condition / compound constraints** — Goals that require satisfying multiple simultaneous constraints (achieve X while not revealing Y; persuade without coercion; change an outcome that depends on others’ beliefs) amplify difficulty because each turn must preserve several invariants.
  - *In failed cases:* Failed 1 required revealing content but preserving anonymity; Failed 4 required eliciting a confession without violating conscience-based resistance and while avoiding immediate harm; Failed 2 attempted to extract an answer without breaching friendship norms.
  - *In passed cases:* Passed tasks generally have a single primary condition (get to drive, share blanket, sleep together) and straightforward incentives or mutual benefit, reducing planning complexity.
- **Need for sustained, varied persuasion across turns** — Successful resolution often requires a sequence of adaptive moves (empathy, reframing, bargaining, concessions) rather than repetition of the same tactic; this demands turn-by-turn strategy and conversational variation.
  - *In failed cases:* Failed 1 showed repetitive storytelling that looped rather than varied tactics; Failed 4 needed adaptive empathy and alternative deals; Failed 2 required escalating/varied incentives but sometimes defaulted to the same offers.
  - *In passed cases:* Passed scenarios were solvable with one or two natural moves (offer the blanket, invite to drive, cuddle to improve sleep) so sustained persuasion was unnecessary.
- **Relational risk and repair sensitivity** — Scenarios where a misstep (bluntness, abrupt exit, revealing secret) severely damages the relationship require careful turn management and repair strategies; mistakes have outsized relational costs.
  - *In failed cases:* Failed 3 featured an abrupt departure after a successful invite that damaged rapport; Failed 1 risked exposing the secret and thus trust; Failed 5 required channeling hostility without destroying the alliance.
  - *In passed cases:* Passed cases either strengthen rapport by default (sharing warmth, intimacy) or do not present actions with high relational collateral, so small errors are less catastrophic.
- **Requirement to manage identity/opacity (deception or anonymity)** — Tasks that ask the agent to manipulate what the partner believes about the agent’s identity or source of information (stay anonymous while revealing) increase complexity because the agent must hedge statements and control inferences.
  - *In failed cases:* Failed 1 explicitly required revealing a secret while hiding that it was the agent's secret and Failed 2 involved coaxing knowledge without exposing motives; mismanagement led to admissions or suspicion.
  - *In passed cases:* Passed scenarios did not require hiding authorship, lying, or complex inference control — straightforward transparency was acceptable.

### hand-craft  (n=13, pass=38%)

**Hard scenario signature:** A hard scenario demands covert or low-pressure persuasion across multiple turns, requires satisfying several simultaneous constraints (emotional preservation, secrecy, concrete material outcomes), and hinges on nuanced, naturalistic phrasing to avoid damaging rapport.

**Easy scenario signature:** An easy scenario features a clear, single-turn or short strategic choice with aligned incentives or explicit permission to be direct, low need for concealment, and few simultaneous success conditions to track.

**What the model lacks:** The model struggles with subtle, multi-turn pragmatic planning: it cannot reliably balance information-seeking with maintaining naturalness or secrecy, nor manage conditional bargaining that preserves relationship capital. It underweights long-horizon conversational strategy (how early moves affect later trust and plausibility).

**Difficulty factors:**

- **covert intent / need for secrecy** — Scenarios that require the learner to hide their true purpose (e.g., buying a surprise gift, not signalling a fundraising goal) demand indirect, plausible-seeming questions and careful conversational framing. This increases difficulty because every informative move must also preserve the concealment.
  - *In failed cases:* Failed 2 required asking many personal preference questions without revealing that a gift was being bought; other fails involved not revealing sensitive personal information while pursuing a goal.
  - *In passed cases:* Passed scenarios involved openly stated intentions or obvious motives (share driving, confess/defect dilemma, set boundaries) so the agent could act directly without covert framing.
- **multi-step conditional bargaining** — Goals that require offering tradeoffs, staged commitments, or sequenced incentives (e.g., borrow money with negotiated repayment/benefit, persuade a friend using material incentives) need planning across multiple turns and conditional promises, which increases cognitive and conversational complexity.
  - *In failed cases:* Failed 1 and Failed 4 involve negotiating money or material benefits and require contingent offers and follow-through; the heirloom search (Failed 5) required coordinating schedules and searching steps without disrupting harmony.
  - *In passed cases:* Passed cases had single, simple, salient asks (switch driving, choose to confess/keep silent, state boundaries) with fewer contingent moves and less need for multi-turn bargaining.
- **high relational/ reputational risk** — When the goal threatens the relationship or requires vulnerability (asking for a large loan, criticizing family behavior, searching a sentimental item), the agent must balance assertiveness with repair strategies and maintain trust, which complicates phrasing and timing.
  - *In failed cases:* Failed 1 (large financial request), Failed 5 (sentimental heirloom) and Failed 4 (persuading a sibling) all required preserving relationship while pushing for a potentially awkward request.
  - *In passed cases:* In Passed 2 and Passed 1 the social cost of the direct ask is limited or calculable (driving swap is routine; prisoner's dilemma is a strategic one-shot with known payoffs), reducing need for delicate reputation management.
- **need for naturalistic, low-pressure information gathering** — Some tasks require eliciting many specific facts while sounding casual and non-interrogative; unnatural rapid-fire questioning breaks believability and undermines success.
  - *In failed cases:* Failed 2's judge noted an unnatural barrage of detailed questions about birthday logistics; Failed 3 and 4 also show over-specified, professional-style information seeking that feels out of place.
  - *In passed cases:* Passed scenarios typically allow a single straightforward disclosure or request (e.g., 'I'm tired, please drive'), so the agent need not sustain casual information-gathering.
- **multiple success conditions / high info requirements** — Scenarios that require satisfying several independent constraints to count as success (exact money amount, secrecy, emotional preservation, schedule coordination) raise the combinatorial difficulty: the agent must track and satisfy many criteria simultaneously.
  - *In failed cases:* Failed episodes often list extra_info or strategy_hints (need $3000 but secure partial amount; hide intent while getting specific gift details; offer financial incentives while keeping relationship intact) meaning several simultaneous constraints.
  - *In passed cases:* Passed episodes have clearer, single-dimensional success metrics (minimize jail years, take turns driving, set a boundary) that are easier to optimize.

### deal-or-no-deal  (n=10, pass=20%)

**Hard scenario signature:** A hard scenario is one where a few scarce items carry disproportionate value, both players' preferences strongly conflict, the partner is uncooperative or anchored, and achieving a deal requires many adaptive rounds or off-item concessions.

**Easy scenario signature:** An easy scenario has distributed or compatible valuations (or an explicit fairness constraint), few items with simple weights, and can be resolved with a small number of clear, one-shot offers that don't demand extended rapport management.

**What the model lacks:** The model struggles to balance multiple social objectives (maximizing points while preserving rapport) and to sustain adaptive, multi-turn bargaining strategies (persistence, calibrated concessions, creative side-payments). It also shows weaknesses in searching the combinatorial allocation space and avoiding premature withdrawal when a negotiated trade is possible.

**Difficulty factors:**

- **high goal conflict / zero-sum preferences** — Scenarios where both parties highly value the same scarce items (or where one or two items dominate total utility) create strong direct competition and few Pareto-improving splits, making agreement harder.
  - *In failed cases:* Several failed cases pivoted around single high-value items (e.g., a highly valued ball/orange) or both players wanting the same items, producing tough trade-offs and no obvious low-cost concessions.
  - *In passed cases:* Passed cases had more evenly distributed or compatible valuations (or an explicit fairness goal), so simple splits or small concessions produced acceptable outcomes.
- **multi-objective tension (points vs. relationship)** — When success requires balancing material gain with social norms (maintaining rapport, not abruptly leaving), the scenario demands softer tactics and reputational management in addition to numerical optimization.
  - *In failed cases:* Judges flagged abrupt exits and damaged relationships in multiple failed episodes — the negotiation required keeping the partner engaged while trading value, but the interaction collapsed when the agent focused on point-maximization or withdrew.
  - *In passed cases:* At least one passed scenario explicitly added a fairness constraint and another implied cooperative strategy, aligning the agent's incentives with socially acceptable behavior and reducing the trade-off.
- **need for sustained, iterative bargaining and creative side-payments** — Hard cases require many back-and-forth turns and off-item compensation (favors, chores, goods) to reach agreement; they demand adaptive concession schedules and creativity beyond a single proposal.
  - *In failed cases:* Failed episodes show negotiations that needed multiple rounds and creative offers; the agent often left early or failed to iteratively adjust offers to achieve a deal.
  - *In passed cases:* Passed episodes resolved with few exchanges or with straightforward, one-shot divisions that did not require extended bargaining or novel side-payments.
- **partner rigidity / low cooperativeness** — If the partner is inflexible (rare concessions, anchored demands), the space for win-win trades shrinks and the negotiation requires persistent persuasion or alternative leverage.
  - *In failed cases:* Judges described partners as rigid in several failures; the scenarios penalized insufficient persistence and flexibility, causing breakdowns.
  - *In passed cases:* Passed cases either involved more cooperative partners (or hints that allowed alignment), so the agent could reach agreement without prolonged persuasion.
- **combinatorial arithmetic and allocation complexity** — Larger item counts, uneven item multiplicities, or disparate point weights increase the search space for optimal splits and raise the chance of calculation errors or suboptimal offers.
  - *In failed cases:* Some failed scenarios had more items (12 total) or uneven valuations and judges noted arithmetic errors and suboptimal resulting allocations.
  - *In passed cases:* Passed scenarios tended to have smaller totals and simpler, more uniform valuations, reducing allocation complexity and arithmetic burden.

### craigslist_bargains  (n=10, pass=50%)

**Hard scenario signature:** A 'hard' scenario has a large price gap with an explicitly inflexible seller or many logistical constraints, where seller motivation is hidden and success requires sustained probing plus creative non-monetary or bundled offers.

**Easy scenario signature:** An 'easy' scenario has a small effective price gap or clear seller urgency/flexibility and few logistical constraints, so a short, direct offer or small concession can close the deal.

**What the model lacks:** The model struggles to convert multi-turn information into tailored leverage: it fails to probe efficiently for seller motivation, to synthesize discovered constraints into creative trade proposals, and to escalate persuasion when faced with a hard floor. It also underuses relational moves (rapport, targeted incentives) that can unlock otherwise rigid sellers.

**Difficulty factors:**

- **seller rigidity / explicit non-negotiability** — Scenarios where the seller states the price is fixed or otherwise signals a hard floor leave little room for simple lowball offers and require discovering or creating alternative concessions.
  - *In failed cases:* Several failed cases include explicit 'price is non-negotiable' language or sellers with a discovered hard floor very close to the asking price (Tile Mate, Galaxy S8, Antique table), so straightforward bargaining fails.
  - *In passed cases:* Passed cases show sellers who either signal flexibility or have implicit motives to move the item (BMW, TV stand, Antique chair), enabling a single well-timed offer or small concession to succeed.
- **large relative price gap coupled with little seller urgency** — When the buyer target is far below the listed price and the seller shows no urgency, the required concession from the seller is large and unlikely without strong leverage or persuasion.
  - *In failed cases:* Failed episodes often had substantial percentage gaps (e.g., Galaxy S8 and 47" TV) and judges noted sellers' unwillingness to accept non-cash or big markdowns.
  - *In passed cases:* Passed episodes had smaller effective gaps or seller signals of willingness to accept modest discounts (TV stand, Antique chair), or clear motivation to sell (BMW), reducing required concessions.
- **informational asymmetry about seller motivation** — Hard scenarios hide why the seller is selling or whether they need a quick sale; resolving that requires multi-turn probing to find leverage points.
  - *In failed cases:* Failed cases required several probes to reveal the seller's true bottom or motivations and often the model either didn't find actionable motivation or the seller genuinely had no urgency.
  - *In passed cases:* Passed interactions either contained explicit motivation cues or the negotiation succeeded without needing deep probing, so the agent could close the deal quickly.
- **multiple transactional constraints** — When listings include constraints (cash-only, pick-up-only, no trade-ins, missing parts), the negotiation space is narrower and the buyer must negotiate around logistics in addition to price.
  - *In failed cases:* Failed episodes specified cash-only, pick-up logistics, missing handle, or no trade-ins, limiting the set of feasible offers and raising complexity.
  - *In passed cases:* Passed listings were simpler logistically or the constraints did not block price negotiation, so the buyer could focus mainly on price and persuasion.
- **need for creative non-monetary trades or bundled offers** — Some sellers might accept services, swaps, or other creative sweeteners; recognizing and proposing those requires inference about seller preferences and a willingness to trade non-cash value.
  - *In failed cases:* Judges noted buyers attempted creative offers but sellers either had hard floors or weren't interested; success required better targeting or discovery of what the seller valued.
  - *In passed cases:* Passed scenarios didn't require or benefited less from creative non-monetary proposals — simple monetary concessions sufficed.
- **sustained multi-turn probing vs. single-ask resolution** — Scenarios that demand iterative concessions, information-gathering, and rapport-building are harder for strategies that rely on one or two direct offers.
  - *In failed cases:* Failures often needed many turns to elicit seller bottom lines and motivations; the model's tactics didn't convert that information into a viable offer before the seller held firm.
  - *In passed cases:* Passed cases were resolvable with few exchanges or the seller quickly revealed a concession, so the agent's simpler tactics worked.

### original_creation  (n=39, pass=87%)

**Hard scenario signature:** A hard scenario combines an emotionally sensitive relationship, an interlocutor with clear resistance or entrenched preference, and a goal that requires multiple negotiated steps or tradeoffs while preserving the relationship.

**Easy scenario signature:** An easy scenario has a single, concrete request or disclosure in a context where the partner is likely to be amenable or where urgency/clarity reduces the need for extended negotiation or complex face-work.

**What the model lacks:** The model struggles to coordinate multi-step, contingent strategies that balance empathy with instrumental persuasion: i.e., it fails to plan sequences of concessions, follow-up commitments, and face-saving moves. It also weakly models partner resistance and history, so it either over-pressures or abandons negotiations instead of using graded, contingent tactics.

**Difficulty factors:**

- **Mixed / dual goals** — Scenarios that require pursuing an instrumental objective while also managing an emotional or relational goal (e.g., giving support while pushing a policy) are harder because they demand balancing empathy and persuasion simultaneously.
  - *In failed cases:* Failed cases often combined aims: offering recovery support while pushing safety protocols (ranch injury), being supportive yet trying to change behavior (siblings cleaning, blackout companion), or coaxing someone to join while preserving comfort (dance party).
  - *In passed cases:* Passed scenes tended to have one dominant, coherent goal — a single confession, a single request for help, or an expression of appreciation paired with one delicate admission — so the agent could focus on one communicative strategy.
- **Partner rigidity / conflicting preferences** — When the partner has a stable, stated preference or resistance (prefers solitude, is relaxed about tasks, has social anxiety), success requires tailored concessions and persistent, contingency-aware persuasion.
  - *In failed cases:* Several fails featured explicit resistance: neighbor preferring darkness, travel companion preferring to wait, injured rider resistant to safety changes, sibling not motivated — each requires overcoming entrenched preferences.
  - *In passed cases:* Passed scenarios involved partners who were either amenable (neighbor with mechanical skills, friends happy to reconnect) or in contexts where compliance is straightforward (urgent help), so less resistance had to be overcome.
- **Need for sustained multi-turn negotiation** — Scenarios that demand a series of small agreements, negotiated tradeoffs, or follow-through are harder because they require planning, contingent responses, and persistence across turns.
  - *In failed cases:* Cleaning-division and airport-preparation scenes required iterated bargaining and concrete task allocation; the ranch scene required maintaining rapport while introducing policy changes over time.
  - *In passed cases:* Passed cases mostly needed a single well-phrased ask or a limited disclosure (confession, quick request for a ride), which can be resolved within few turns.
- **High emotional vulnerability / face management** — When the partner or agent is emotionally exposed (fear of rejection, recovery pain, social anxiety), the interaction requires careful empathy and calibrated risk-taking to avoid harming the relationship.
  - *In failed cases:* Conflicts about safety after injury, persuading someone uncomfortable in the dark, and encouraging a socially anxious friend to dance all required sensitive face-saving strategies and reassurance — mishandling these causes breakdown.
  - *In passed cases:* Although some passed scenes were emotionally charged, the setting made vulnerability manageable (private intimate moment for a confession, friends expecting reconnection), reducing the complexity of face work.
- **Multiple conditional steps / coordination complexity** — Scenarios that hinge on several conditions being met (both parties completing separate tasks, coordinating times/roles, or changing shared norms) raise the bar because success depends on several interlocking commitments.
  - *In failed cases:* Airport passports+prints, sibling division of chores, and implementing ranch safety protocols require multiple actions and coordination, any of which can derail the outcome.
  - *In passed cases:* Passed scenarios tended to require a single immediate action or response (give emotional disclosure, accept a ride/help), minimizing coordination failure points.
- **Relational complexity and power dynamics** — When pre-existing relationships have asymmetric expectations (siblings with history, acquaintances vs. close friends) or the outcome could alter the relationship, the agent must navigate history, norms, and authority carefully.
  - *In failed cases:* Siblings negotiating chores involve long-standing norms and entrenched roles; acquaintances in a blackout or ranch colleagues involve differing levels of obligation and trust that complicate persuasion.
  - *In passed cases:* Passed interactions either occurred between long-trusting friends in a supportive context or between a person and a neighbor with a clear, narrow role (mechanic), reducing ambiguous social costs.

### mutual_friends  (n=10, pass=70%)

**Hard scenario signature:** A hard scenario supplies only weak, generic, or mismatched attributes about friends and the partner gives mostly negative or vague responses, forcing multiple elimination steps and cross-attribute inference under a constraint against directly naming people.

**Easy scenario signature:** An easy scenario provides at least one clear, distinctive attribute shared between parties (aligned school/hobby/other cue) or allows a small number of high-yield questions that directly reveal a mutual connection without long elimination chains.

**What the model lacks:** The model lacks robust multi-step hypothesis-testing strategies: planning short sequences of diagnostic questions, integrating negative evidence, and converting sparse or cross-attribute clues into discriminative inferences. It also struggles to adapt questioning style when the partner provides vague or exclusionary answers.

**Difficulty factors:**

- **need_for_multi-step_elimination** — Scenarios that require the agent to eliminate multiple non-matches through a sequence of targeted questions (rather than a single diagnostic question) are harder because success depends on planning a short decision tree and remembering prior negative answers.
  - *In failed cases:* Judge notes in the failed cases show the learner mostly ruled out candidates or learned only non-matches (e.g., learned details allowing them to rule people out), indicating the task required multiple elimination steps.
  - *In passed cases:* Passed episodes appear to allow a single or very small number of high-yield questions (or more directly diagnostic cues) that reveal the mutual connection without long elimination sequences.
- **low_attribute_diagnosticity** — When the known attributes of friends are generic or incomplete (partial school/hobby strings, common activities) they fail to uniquely identify a mutual friend, making inference from partner responses fragile.
  - *In failed cases:* The agent's friend entries in failed cases are partially truncated and generic (e.g., partial hobby or school fields), offering few unique cues to map to the other person's descriptions.
  - *In passed cases:* Passed cases show friend attributes that are more distinctive or align better with likely conversational cues (clear school or hobby tags), making matches easier from fewer signals.
- **attribute_mismatch_between_parties** — Harder scenarios occur when the attributes the agent knows about their friends (e.g., school, hobby) are different from the attributes the partner is likely to mention (e.g., workplace, charities), forcing cross-attribute inference rather than direct matching.
  - *In failed cases:* Judge comments indicate learners had to reconcile different kinds of info and mostly learned non-matching attributes (work/university vs hobbies), implying a mismatch in attribute types shared by partners.
  - *In passed cases:* In passed cases the salient attributes available to the agent and those the partner reveals appear to align, so direct matching is possible with minimal cross-attribute inference.
- **partner_vagueness_or_negative_information** — Scenarios where the other person predominantly provides vague replies or negations (what their friends are not/do not do) make it hard to converge on a positive identification of a mutual friend.
  - *In failed cases:* Failed transcripts show the learner mainly received negative information or learned exclusions rather than positive identifying details, leaving the mutual friend unconfirmed.
  - *In passed cases:* Passed interactions imply the partner offered at least some positive or identifying detail early, reducing ambiguity and the need for many follow-ups.
- **low_social_reward_for_directness** — When the scenario implicitly constrains direct naming (social constraints or instruction to not just list names), agents must use indirect elicitation; tasks that require subtle indirect strategies are harder than those allowing straightforward verification.
  - *In failed cases:* Failed episodes emphasize not simply listing names and required more skillful elicitation, but the scenario supplies sparse clues, amplifying difficulty.
  - *In passed cases:* Passed scenarios permitted conversational moves or provided cues that made indirect identification straightforward without violating the 'no listing names' constraint.

### social_dynamics  (n=10, pass=70%)

**Hard scenario signature:** A hard scenario pits the agent's goal against another person's core values or autonomy, involves high emotional stakes (possible trust loss or relationship termination), and requires balancing multiple constraints via sustained, adaptive dialogue.

**Easy scenario signature:** An easy scenario has low emotional cost, a single clear objective, and a tolerant or neutral partner—success can typically be achieved with a brief, polite, and transparent utterance.

**What the model lacks:** The model struggles with multi-objective social reasoning: balancing persuasion with respect for identity, and sequencing reparative moves when trust is damaged. It also underutilizes sustained, empathy-driven strategies (explicit validation, iterative concessions) required to de-escalate high-stakes or values-based conflicts.

**Difficulty factors:**

- **value_conflict_and_moral_identity** — Scenarios where the other person's choices are rooted in firm values or moral identity are harder because successful interaction requires persuading or accommodating without threatening that identity. Actions seen as violating those values trigger resistance that a single benign pitch cannot overcome.
  - *In failed cases:* Failed 1 asks an environmentally conscious friend to join a car road trip that conflicts with their transit preferences; Failed 3 centers on a friendship split due to fundamental value divergence.
  - *In passed cases:* Passed scenarios involve practical, social, or preference-based issues (minor incident, scheduling/conflict on dates, polite curiosity about tattoos) rather than challenges to core moral identity.
- **high_emotional_stakes_and_finality** — When outcomes have strong emotional consequences (trust loss, relationship termination), the interaction must manage grief, defensiveness, and long-term repercussions. These situations require careful pacing, reassurance, and sometimes multiple reparative moves.
  - *In failed cases:* Failed 2 involves a boundary violation that undermines autonomy and trust; Failed 3 explicitly ends a long friendship, a high-stakes, final move with heavy emotional impact.
  - *In passed cases:* Passed cases are low-to-moderate stakes (preserving harmony, choosing between dates, a brief compliment) where immediate harm is minimal and reversibility is high.
- **need_for_trust_repair_and_boundary_management** — Scenarios that require restoring trust or acknowledging an overstep demand explicit apologies, validation, and negotiated future boundaries; simple persuasion or explanation is insufficient.
  - *In failed cases:* Failed 2 centers on repairing a boundary breach (asking parents without consent) and earning back autonomy and respect; Failed 1 also required offering compromises that respect Agent2's principles rather than overriding them.
  - *In passed cases:* Passed interactions do not hinge on undoing a specific breach of trust or re-establishing autonomy, so fewer delicate repair moves are necessary.
- **multi-constraint_goals_and_tradeoffs** — Hard scenarios impose several simultaneous constraints (convince the other, avoid harming the relationship, respect identity, achieve a practical outcome), so acceptable responses must balance competing goals rather than satisfy a single clear objective.
  - *In failed cases:* Failed 1 must persuade without violating environmental values and maintain rapport; Failed 2 must both apologize and salvage the invitation; Failed 3 must explain a breakup while minimizing unnecessary harm.
  - *In passed cases:* Passed cases typically have one primary objective (preserve group harmony, explain predicament about dates, ask about tattoos) and fewer conflicting constraints.
- **partner_rigidity_and_sensitivity** — When the interlocutor is portrayed as especially rigid or sensitive (firm beliefs, strict parents, high boundary vigilance), there is lower tolerance for mistakes and a narrower range of acceptable appeals, making success harder.
  - *In failed cases:* Failed 1's friend is strongly environmentally conscious; Failed 2's friend values independence and has strict parents; Failed 3 involves long-standing relational history that heightens sensitivity.
  - *In passed cases:* Passed scenarios feature more malleable or neutral partners (community member to be reassured, colleague open to discussion, stranger at a gallery).
- **requires_sustained_back-and-forth_and_nuanced_empathy** — Difficult scenarios typically require extended dialogue, iterative repair, explicit validation, and adaptive tone over multiple turns rather than a single request or compliment.
  - *In failed cases:* Failed 2 and 3 need multi-turn trust repair and emotional calibration; Failed 1 may require negotiating alternatives and responding to principled objections across turns.
  - *In passed cases:* Passed interactions are often resolvable with a concise, single-turn reassurance, clear explanation, or a brief, low-risk question that doesn't demand prolonged emotional labor.

### inspired_by_prompt  (n=15, pass=80%)

**Hard scenario signature:** A hard scenario features elevated physical or emotional risk, strong partner rigidity or institutional constraints, and requires multi-turn probing plus conditional trade-offs (safety options, phased agreements, or alternative access) to reach a solution.

**Easy scenario signature:** An easy scenario is low-risk with a single clear lever (motivation, simple compromise or reassurance) and a partner who is amenable to brief encouragement or one-off concessions.

**What the model lacks:** The model struggles with multi-step persuasion strategies: it does not reliably elicit hidden constraints, construct conditional proposals, or execute graded reassurance/safety plans over several turns. It also underuses explicit de-escalation and contingency framing needed when stakes, fear, or institutional rules restrict acceptable moves.

**Difficulty factors:**

- **Safety / high emotional stakes** — Scenarios that involve potential physical danger or strong emotional fears raise stakes and constrain acceptable persuasive moves; they require careful de-escalation, reassurance, and safety planning rather than simple encouragement.
  - *In failed cases:* Failed cases include an upside-down exercise that triggers a fear of heights, and a collectors' dispute that recalls a previous gun brandishing—both require safety-first responses and cautious negotiation.
  - *In passed cases:* Passed scenarios (friendly 5K competition, neighbor noise, karaoke) are low-physical-risk and emotionally lower-stakes, so straightforward encouragement or compromise is appropriate.
- **Multiple conditional constraints / complex trade-offs** — Successful resolution depends on satisfying several interlocking conditions (institutional rules, time pressures, ownership terms) so effective responses must propose and coordinate multi-option solutions.
  - *In failed cases:* The library case involves checkout policy, a deadline, and alternative access; the collectors case expects buyout/shared custody/trade options; the upside-down exercise may require stepwise exposure, spotting, and exit plans.
  - *In passed cases:* Passed scenarios typically have a single clear lever (motivation, agreed quiet times, fun framing) and can be resolved with one or two simple offers or concessions.
- **Partner rigidity or fear-driven refusal** — When the other person holds a principled, fear-based, or institutionally backed stance, persuasion requires more probing, validation, and tailored concessions rather than generic encouragement.
  - *In failed cases:* Agent2s in failed episodes are resistant because of fear (heights), firm constraints (library policy/need for whole book), or adversarial stance (collector staking claim), making a single ask unlikely to work.
  - *In passed cases:* In passed episodes the partner is more malleable (motivated by fun/competition or open to time-based compromises), so lighter persuasive tactics succeed.
- **Need for sustained multi-turn scaffolding** — Hard cases require a sequence of small steps, trust-building, or probing questions over multiple turns (e.g., graded exposure, negotiating phased ownership), not a single persuasive utterance.
  - *In failed cases:* The upside-down exercise calls for modeling, practice, and safety checks across turns; the collectors' dispute and library negotiation both require iterative bargaining and contingency-making.
  - *In passed cases:* Passed scenarios can usually be resolved with a few reassuring lines, explicit compromises, or one-off encouragement that do not demand long scaffolding.
- **Ambiguous or missing partner information** — Scenarios where success depends on unspoken motives or specific constraints force the agent to elicit information first; failure occurs when those probes or tailored responses are not produced.
  - *In failed cases:* Failed episodes hide critical specifics (exact nature of the fear of heights, precise reasons the other needs the book, terms acceptable for settlement) that must be uncovered to craft viable offers.
  - *In passed cases:* Passed episodes contain clear, surface-level motivations (wanting to improve, keep music down, have fun) so the agent can respond effectively without deep elicitation.

### persuation_for_good  (n=10, pass=80%)

**Hard scenario signature:** A hard scenario pairs a persuasive ask with conflicting incentives or formal constraints (quality standards, donation rules), requires satisfying multiple independent conditions or technical evidence, and demands iterative negotiation with a risk‑averse or professional partner.

**Easy scenario signature:** An easy scenario is a single, low‑cost ask to a sympathetic friend or donor with aligned incentives, where an emotional appeal or brief rationale suffices and no technical approvals or trade-offs are needed.

**What the model lacks:** The model struggles with planning multi-step persuasion strategies that simultaneously satisfy competing constraints and with producing credible technical or procedural justifications for skeptical, risk‑averse interlocutors. It also underutilizes staged negotiation: proposing trade-offs, securing small commitments, and progressively building credibility.

**Difficulty factors:**

- **conflicting goals / incentives** — Scenarios where the interlocutor has an objective that directly conflicts with the learner's ask (e.g., maintain product quality or comply with event rules) are harder because the target must give up or trade off something they value.
  - *In failed cases:* In the supplier-cutting case the partner prioritizes quality control and lab/COA standards that oppose cost-cutting; in the gala case there was an explicit donation constraint (minimums/requirements) requiring a workaround.
  - *In passed cases:* Passed cases involve friends predisposed to give or low-stakes donation asks where the target's incentives are already aligned with donating, so little or no tradeoff is required.
- **multiple conditional requirements** — When success requires satisfying several independent conditions (technical approvals, quality tests, or logistical constraints) persuasion becomes a multi-step problem rather than a single ask.
  - *In failed cases:* The business scenario required meeting lab stability, COAs, and sensory panels; the gala conversation required finding a workaround to specific donation rules and accommodating donor limits.
  - *In passed cases:* Passed dialogues typically need only a simple yes/no donation decision without extra approvals or procedural hurdles.
- **partner rigidity and risk-aversion** — Targets who are cautious, professional, or bound by standards require evidence, assurances, or negotiated trade-offs making persuasion slower and more contingent.
  - *In failed cases:* The partner in the company review demanded technical proof before changing suppliers; the gala partner had fixed donation limits and social/procedural expectations that reduced flexibility.
  - *In passed cases:* Friends at charity events are portrayed as generous and emotionally receptive, so they are less risk-averse and easier to move with appeals.
- **high domain/technical knowledge demand** — Persuasion that depends on technical details or specialized evidence needs the persuader to present credible facts or expertise, increasing difficulty if the agent lacks domain signals.
  - *In failed cases:* The supplier argument hinged on lab testing, COAs, and quality-control criteria—details that require authoritative evidence to overcome objections.
  - *In passed cases:* Passed scenarios rely on general moral/emotional appeals about children or reputable charities that don't demand specialist proof.
- **relational complexity and formality** — Professional or unfamiliar relationships introduce norms, reputational risk, and need for formal justification, whereas close friendships allow more direct emotional appeals and social flexibility.
  - *In failed cases:* Both failed cases are more formal/professional (business partners, individuals at a gala with rules), increasing social costs for pushy tactics and requiring tactful negotiation.
  - *In passed cases:* Passed scenes are informal friend-to-friend interactions where candid emotional appeals and direct asks are socially appropriate and effective.
- **need for sustained back-and-forth and problem solving** — Harder scenarios require iterative negotiation, proposing alternatives, and adapting to revealed constraints instead of a single persuasive pitch.
  - *In failed cases:* The failed episodes involved discovering constraints mid-conversation (donation limits, quality standards) and needed iterative solution-building (workarounds, compromise on supplier choices).
  - *In passed cases:* Passed episodes typically succeed with a single emotional framing or simple request and minimal negotiation.

### generated  (n=7, pass=71%)

**Hard scenario signature:** A hard scenario pits the agent against a partner with entrenched hesitations or misinformation, requires simultaneous emotional validation and practical/logistical planning, and demands iterative, trust-building persuasion rather than a single informational reply.

**Easy scenario signature:** An easy scenario features cooperative partners, shared goals or clear social norms, and a single actionable objective (comforting, a concrete procedural ask, or low‑threat persuasion) that can be advanced with one or two targeted, direct moves.

**What the model lacks:** The model lacks adaptive interpersonal calibration: it tends to default to planning or factual presentation rather than first establishing rapport, clarifying the partner's emotional stance, and sequencing trust-building steps. It also struggles to manage entrenched beliefs through motivational interviewing–style moves (validate, elicit, tailor evidence) across multiple turns.

**Difficulty factors:**

- **role / perspective clarity** — Scenarios that require the speaker to clearly adopt a first-person, empathic stance toward a single partner (speak directly to their feelings and decisions) are harder when the situation encourages third‑party planning or depersonalized advice.
  - *In failed cases:* In the pet adoption failure the agent's responses read like planning for a third party rather than speaking directly to Agent1's hesitations; in the vaccine case the agent leaned toward delivering citations and facts rather than conversationally addressing the friend's concerns.
  - *In passed cases:* Passed cases featured clearer interpersonal framing (siblings honoring a brother, neighbors debating pet freedom, colleagues at work) so the model could naturally adopt a direct, relational stance and maintain role clarity.
- **partner rigidity / entrenched beliefs** — When the conversational partner holds entrenched or misinformation-driven beliefs, success requires careful epistemic strategies (trust-building, calibrated challenges, motivational interviewing) rather than a single corrective message.
  - *In failed cases:* The vaccine scenario involved a skeptical partner influenced by misinformation, requiring delicate challenge of beliefs; the adoption scenario included strong personal hesitations about capacity and suitability that function like entrenched barriers.
  - *In passed cases:* Passed interactions involved partners who were more open or aligned (neighbors with negotiable views, siblings seeking comfort, colleagues with shared institutional incentives), reducing the need to overcome deep resistance.
- **emotional personalization and high stakes** — Scenarios that hinge on intimate fears, guilt, or identity (e.g., loneliness, guilt about a death, skepticism tied to identity) demand empathic, validating responses and careful pacing; these are harder than low‑stakes informational asks.
  - *In failed cases:* The adopter's loneliness and fear of failing a pet and the friend's vaccine mistrust are personally meaningful and identity-linked, requiring affective validation which the agent neglected in favor of planning or citation.
  - *In passed cases:* Although one passed case involved grief, the structure made supportive responses straightforward (shared ritual, explicit permission to grieve); other passed cases were framed around practical or professional norms rather than fragile personal identity.
- **number of success conditions / practical constraints** — Failures are more likely when success depends on satisfying many conditions (emotional reassurance, logistics, finances, long‑term commitment) rather than a single conversational goal.
  - *In failed cases:* The adoption scenario requires addressing emotional readiness, budgeting, shelter logistics, and long-term care commitments; the vaccine case requires restoring trust, correcting misinformation, and sometimes addressing systemic concerns about safety.
  - *In passed cases:* Passed scenarios often required one clear outcome (accepting more pet freedom, honoring a memory, following hygiene protocols) with fewer simultaneous constraints.
- **need for sustained iterative persuasion** — Scenarios that realistically require multiple conversational turns and gradual attitude change are harder for a single-turn dialog model because they demand strategy sequencing and monitoring of changing cues.
  - *In failed cases:* Both failed scenarios implicitly require a sequence of trust‑building, empathy, tailored evidence, and follow‑up — not just a one-shot explanation or plan.
  - *In passed cases:* Passed cases could often be advanced meaningfully in a single interaction by offering concrete steps, emotional validation, or appealing to shared norms, reducing the need for long sequences.
- **ambiguity in conversational goal (support vs planning vs education)** — When the scenario's goal is multifaceted or ambiguous (is the task emotional support, practical planning, or factual education?), the model must choose an approach; mis-selection makes success unlikely.
  - *In failed cases:* The judge noted confusion in the pet case between planning and direct support; the vaccine case required blending empathy with evidence but the agent focused mainly on citation-driven education.
  - *In passed cases:* Passed scenarios had clearer single-minded goals (comfort, persuasion toward protocol, or practical compromise) so the appropriate strategy was more obvious.

### user_generated  (n=11, pass=82%)

**Hard scenario signature:** A hard scenario pits the learner's goal against a firmly held personal value or autonomy in an ongoing relationship, requires satisfying multiple constraints at once (outcome + relationship preservation), and needs iterative, emotionally sensitive bargaining.

**Easy scenario signature:** An easy scenario has low-resistance interlocutors or shared norms, a single primary objective, and can be resolved with a clear, direct appeal or one-off compromise without risking long-term relationship harm.

**What the model lacks:** The model struggles with multi-step, emotionally attuned negotiation strategies: it tends to rely on direct persuasion rather than incremental concessions, face-saving language, and validation that reduce resistance. It also underutilizes relationship-preserving moves (explicit boundary acknowledgements, staged compromises) needed when autonomy or identity is at stake.

**Difficulty factors:**

- **Partner rigidity / resistance** — Scenarios where the interlocutor has a strongly held opposing preference or value require careful, multi-step persuasion and boundary management; simple arguments or one-off offers are unlikely to succeed.
  - *In failed cases:* In the roommate case the partner prefers staying in and cooking; in the parent/adult-child case the child strongly values independence. Both present clear, entrenched opposition to the learner's goal.
  - *In passed cases:* Passed scenarios featured lower or more negotiable resistance (a friend worried about concentration, a club competition, or shared values like academic integrity) so a single persuasive move or appeal to norms sufficed.
- **Relational and identity stakes** — When the goal implicates personal autonomy, parental authority, or the ongoing quality of a close relationship, success requires preserving trust and face — adding constraints beyond mere agreement.
  - *In failed cases:* The parent/child scenario directly threatens the child's autonomy and identity; the roommate scenario risks damaging everyday rapport by pushing when one is exhausted, so the learner must negotiate both outcome and relationship.
  - *In passed cases:* Passed cases involved transient or low-identity stakes (a short celebration, a single competition, or enforcing shared academic norms) where persuasive moves don't threaten core identity or long-term trust as much.
- **Multiple success conditions / trade-offs** — Hard scenarios require achieving the target behavior while simultaneously satisfying secondary constraints (preserve autonomy, avoid guilt, offer compensation), increasing planning complexity.
  - *In failed cases:* To get the roommate to go out the learner must overcome exhaustion, offer compensation or alternative, and preserve goodwill; to influence the adult child the learner must give advice without coercion and maintain a respectful relationship.
  - *In passed cases:* Passed scenarios typically had a single primary condition (perform a task, get permission for a one-time event, endorse reporting) and fewer simultaneous constraints to balance.
- **Need for sustained back-and-forth negotiation** — Scenarios that require iterative bargaining, attentive concessions, and adaptive emotional calibration are harder than those solvable with a short persuasive appeal.
  - *In failed cases:* Both failed episodes imply ongoing negotiation (roommates live together; parent-child is an ongoing relationship) and likely need repeated reassurance and staged compromises rather than a single prompt.
  - *In passed cases:* Passed episodes could often be resolved with a clear, immediate rationale or a small one-off compromise rather than a prolonged interaction.
- **Emotional intensity / sensitivity** — When topics touch on deep emotions (autonomy, parental role, exhaustion and care) the interlocutor can react unpredictably; successful strategies must de-escalate, validate feelings, and propose face-saving alternatives.
  - *In failed cases:* The parent/child exchange involves sensitive family authority and independence; the roommate case involves personal exhaustion and perceived obligation — both require strong empathic framing and careful wording.
  - *In passed cases:* Passed scenarios involved lighter emotions or normative appeals (competition, celebration, academic integrity), so arguments could be more direct and less risk of emotional escalation.

### family_dynamics  (n=8, pass=75%)

**Hard scenario signature:** A hard family_dynamics scenario combines hidden or shameful information, high emotional cost, competing objectives (discover truth while avoiding hurt), and third-party pressures, requiring multi-turn trust-building and contingency strategies.

**Easy scenario signature:** An easy scenario focuses on an observable behavior or single emotional complaint between two people, with low hidden information and clear, direct conversational moves (express concern, request change, apologize) that can succeed in a short interaction.

**What the model lacks:** The model struggles with strategies for phased, low-risk information elicitation and with balancing multiple simultaneous social goals (truth-seeking, emotional containment, relationship preservation). It also underutilizes contingency planning for likely refusals and the incremental trust-building required across turns.

**Difficulty factors:**

- **hidden-information asymmetry** — Scenarios where the interlocutor has withheld important facts or where the learner lacks knowledge that substantially changes the meaning of events require careful, phased elicitation and make conversational success contingent on uncovering concealed details.
  - *In failed cases:* Failed 1: Agent2 had not told the other biological parent about the adoption — a critical fact unknown to Agent1 that required sensitive discovery. Failed 2: Agent1 did not know where Agent2 stood on romantic interest beyond social cues, so success depended on eliciting private preference without pressure.
  - *In passed cases:* Passed cases involved addressing observable behaviors (dismissive tone, lack of empathy, suspected surveillance) or straightforward concerns where evidence or feelings were immediate and the conversational move was a direct confrontation or request for change rather than unearthing a concealed past.
- **high emotional stakes and risk of harm** — When the topic touches deep identity, abandonment, or potential romantic rejection, the cost of misstepping is large; the learner must both extract sensitive information and avoid inflicting emotional harm, increasing the number of delicate trade-offs.
  - *In failed cases:* Failed 1: Adoption and parental secrecy are identity- and abandonment-linked issues with high emotional risks. Failed 2: Proposing a romantic relationship risks losing a friendship and causes personal embarrassment or hurt.
  - *In passed cases:* Passed interactions focused on behavior correction or expressing hurt about a concrete incident, where the emotional goal was bounded (repair, set boundary) and the immediate risk of catastrophic relational damage was lower.
- **competing goals / multi-constraint optimization** — Scenarios that require achieving multiple, potentially conflicting objectives (e.g., learn full truth + avoid distress + preserve relationship) are harder because they force trade-offs and sequencing decisions rather than a single persuasive or declarative act.
  - *In failed cases:* Failed 1 required full disclosure while minimizing agent2's distress and building a relationship; Failed 2 demanded persuading Agent2 to date while preserving the existing friendship and managing a parent’s expectations.
  - *In passed cases:* Passed tasks generally had a single primary aim (express hurt, confront surveillance, request respectful speech) where success could be achieved with one well-calibrated conversational move and did not require juggling multiple, opposing constraints simultaneously.
- **partner rigidity and predictable refusal** — When the interlocutor is likely to refuse, be defensive, or hold a firm prior (romantic disinterest, secret-keeping), the scenario becomes harder because the learner must plan contingencies, manage face-threats, and preserve rapport after rejection.
  - *In failed cases:* Failed 2 involved a partner likely to decline romantic advances (resulting in preservation of friendship but failure on the romantic goal). Failed 1 involved a parent who may become defensive or ashamed about past secrecy.
  - *In passed cases:* In passed episodes partners were more likely to accept being corrected or to engage in cooperative problem-solving (e.g., apologize, explain, or agree to change behavior), reducing need for complex contingency handling.
- **need for sustained trust-building / multi-turn scaffolding** — Hard cases require gradual disclosure, empathy, and follow-up commitments over multiple turns (or meetings) rather than a single request; success depends on pacing, reassurance, and incremental bargaining.
  - *In failed cases:* Failed 1 demanded establishing enough trust to reveal painful past events and negotiate future relationship-building — not solvable with one direct question. Failed 2 required ongoing boundary management between family pressure and interpersonal chemistry.
  - *In passed cases:* Passed scenarios were resolvable within a short interaction: stating expectations, confronting a behavior, or requesting concern for wellbeing — actions that can produce clear outcomes quickly without extended scaffolding.
- **relational complexity and third-party dynamics** — Scenarios involving third parties (other biological parent, the mother pressuring matchmaking) or broader family-system implications create more moving parts to manage and additional loyalties that constrain acceptable moves.
  - *In failed cases:* Failed 1 implicated a second biological parent whose knowledge/state mattered to the truth and future relationship. Failed 2 involved the mother's expectations as an external pressure that complicated Agent1’s options.
  - *In passed cases:* Passed episodes mainly centered on dyadic interactions between Agent1 and Agent2 with limited third-party entanglement, making the set of permissible and effective conversational strategies narrower and easier to navigate.

### inspired_prompt  (n=10, pass=80%)

**Hard scenario signature:** A hard scenario asks the agent to persuade a partner to take an action that conflicts with the partner's ethics or exposes them to reputational risk, requires multiple independent assurances or evidence, and demands sustained negotiation rather than a single request.

**Easy scenario signature:** An easy scenario involves low-stakes requests or opinion exchanges where compliance or agreement does not threaten the partner's principles, can be achieved with one clear argument or a simple supportive move, and needs little follow-up.

**What the model lacks:** The model struggles with multi-step persuasive strategies: anticipating principled objections, sequencing trust-building actions (evidence, concessions, safeguards), and offering credible mitigation for ethical/reputational risks. It also underuses reframing techniques that align requests with the partner's values and neglects to propose concrete procedural guarantees that would make risky asks acceptable.

**Difficulty factors:**

- **High goal conflict / ethical disagreement** — Scenarios are harder when the learner's request directly conflicts with the partner's moral stance or professional ethics, because the partner may refuse on principle rather than lack of persuasion.
  - *In failed cases:* Both failed cases ask Agent2 to do something that touches ethics: endorse a confrontational anti-bullying tactic (potentially harmful or divisive) and write a positive recommendation despite ethical doubts about the applicant.
  - *In passed cases:* Passed cases involve low moral conflict (arguing a policy view, asking for a small grooming change, or offering support for distress) where the partner can change opinion or comply without violating core ethics.
- **High personal / reputational stakes** — Requests that could harm the partner's reputation or professional integrity raise stakes and make brief persuasive moves insufficient; partners need strong, credible assurances before risking consequences.
  - *In failed cases:* Requesting a strong recommendation letter and endorsing confrontational tactics both expose Agent2 to reputational or ethical consequences, so Agent2 requires substantial justification.
  - *In passed cases:* Passed scenarios ask for minimal personal risk (reduce a distracting habit) or revolve around discussion/support rather than risking the responder's standing.
- **Multiple conditions for success (evidence, mitigation, and trust-building)** — Hard scenarios require the requester to satisfy several independent concerns (ethical justification, concrete evidence, follow-up plans, safeguards), not just make a single ask.
  - *In failed cases:* The recommender needs specific examples, contextualization, and assurances; the campaign collaborator needs evidence the confrontational approach is effective and won’t backfire—multiple threads to address.
  - *In passed cases:* Passed cases typically succeed with a single clear move (make a reasoned argument, gently request a behavior change, or express concern and offer help) without lengthy mitigation steps.
- **Partner rigidity / principled resistance** — When the partner holds firm principles or professional standards, persuasion requires reframing their values or offering trade-offs, which is more complex than appealing to preferences.
  - *In failed cases:* Agent2 in both fails is portrayed as ethically constrained—unsure about endorsing confrontational tactics or writing a letter after an incident—so they are less malleable to simple persuasion.
  - *In passed cases:* In passed examples the partner is either open to debate (policy discussion), responsive to social norms (minimizing a distracting habit), or receptive to support (someone who appears distressed).
- **Need for sustained back-and-forth negotiation** — Scenarios that require iterated concessions, conditional offers, and Q&A make a one-shot persuasive utterance unlikely to succeed; the dialogue must manage objections over multiple turns.
  - *In failed cases:* Both failed episodes imply follow-up: providing specific documentation, offering safeguards, negotiating campaign content—none are resolvable with a single persuasive statement.
  - *In passed cases:* Passed episodes are solvable with immediate, low-bandwidth interaction (state a position, make a polite request, or show concern) and typically don't demand extended bargaining.

### original_content  (n=11, pass=91%)

**Hard scenario signature:** A hard scenario pits two repeatedly enacted goals against each other, requires satisfying multiple constraints (instrumental outcome + relationship preservation), and demands concrete, sustained coordination or technical compromises to resolve.

**Easy scenario signature:** An easy scenario has a single clear social objective that can be achieved with one or a few tactful conversational moves, involves a receptive or pliable partner, and does not require long-term behavioral change or technical solutions.

**What the model lacks:** The model struggles with multi-issue negotiation: it doesn't consistently generate concrete, implementable trade-offs or plans for sustained coordination and it underestimates partner resistance tied to habits/identity. It also weakly integrates domain-specific fixes into social proposals.

**Difficulty factors:**

- **High goal conflict** — Scenarios where the learner's desired outcome directly competes with the partner's habitual or valued behavior create harder trade-offs and require genuine negotiation rather than a one-off request.
  - *In failed cases:* Agent1 wants a quiet, productive dawn session while Agent2's preferred fishing method (loud modern gadgets) directly undermines that goal, so interests are in direct opposition.
  - *In passed cases:* Passed cases either involved persuading about parenting style, encouraging confidence, or discreetly warning about a wardrobe issue—goals that can be framed cooperatively or achieved without forcing the partner to abandon an important, repeated practice.
- **Multiple and competing success conditions** — When success requires satisfying more than one constraint (e.g., peace + productivity + relational harmony), planning and trade-offs are needed; single-action solutions are unlikely to meet all conditions simultaneously.
  - *In failed cases:* The learner must secure both a peaceful environment and a productive fishing outcome, while preserving the relationship and accommodating the partner’s preferences.
  - *In passed cases:* Passed scenarios tend to have a single clear objective (convince, encourage, or inform discreetly) that can be achieved with one or a small number of conversational moves.
- **Need for sustained coordination or behavior change** — Scenarios that require the partner to change habits across future interactions (scheduling, altering equipment use) demand commitments, follow-up, and concrete proposals, not just empathic statements.
  - *In failed cases:* Resolving the disturbance likely requires repeated accommodations (e.g., changing times, equipment settings, spatial agreements) and coordination across visits.
  - *In passed cases:* Passed cases typically require a single intervention (a persuasive framing, reassurance, or a discreet alert) with little or no ongoing coordination.
- **Partner rigidity / entrenched preference** — When the partner has a strong identity-linked preference or a technical habit, they are less likely to concede; success requires tailored incentives or specific compromises rather than generic appeals.
  - *In failed cases:* Agent2’s use of modern, loud gadgets appears tied to their fishing identity and equipment choices, making simple requests less likely to succeed without concrete alternatives.
  - *In passed cases:* In the passed scenarios the partner is portrayed as more pliable or motivated by social/emotional cues (e.g., protecting family ties, boosting a friend’s confidence, avoiding embarrassment).
- **Need for domain-specific solutions** — Harder scenarios demand practical, concrete proposals (technical or logistic fixes) that reconcile competing aims; lacking those, dialogue can stall.
  - *In failed cases:* A workable resolution requires proposing options like scheduling, quieter settings, or spatial separation—specific technical or logistical solutions.
  - *In passed cases:* Passed cases can rely on general social strategies (empathy, reassurance, tact) rather than technical problem-solving.

### custom_scenario  (n=5, pass=80%)

**Hard scenario signature:** A hard scenario asks the agent to induce another person to perform a potentially risky or burdensome act while simultaneously avoiding certain speech acts (no explicit ask) and preserving a constrained relationship/professional norm, requiring subtle multi-turn persuasion and safety-aware framing.

**Easy scenario signature:** An easy scenario involves expressing emotions, providing hints, or making socially permissible direct requests with low safety risk and few simultaneous constraints, so a single clear conversational move or simple hinting suffices.

**What the model lacks:** The model struggles to generate tactful, indirect persuasive moves that respect imposed prohibitions and safety concerns; it tends to fall back to explicit requests and abrupt endings instead of multi-step scaffolding, risk mitigation, and polite closure. It also fails to creatively reframe the situation to produce acceptable alternatives.

**Difficulty factors:**

- **Prohibition on explicit speech act** — The goal specifically forbids making an explicit request (e.g., 'please crawl under the shelf'), forcing the agent to use indirect persuasion or implication rather than the straightforward strategy of asking.
  - *In failed cases:* Agent1 had to get Agent2 to retrieve an item but could not explicitly ask them to crawl under the shelf, a constraint the model violated by making a direct request.
  - *In passed cases:* Passed scenarios either allowed direct expressions or were achievably guided with benign indirect cues (e.g., hinting at where to look) without requiring avoidance of a clearly defined forbidden utterance.
- **Safety / physical risk and ethical sensitivity** — The target action involves a potentially risky physical task for the partner (crawling under a heavy shelf), which raises ethical/safety concerns and demands cautious wording and risk mitigation strategies.
  - *In failed cases:* The failed case required persuading someone to perform a potentially dangerous action while also acknowledging Agent1's own injury and not imposing, increasing complexity.
  - *In passed cases:* Passed scenarios involved low physical risk (finding an item at a market, discussing a gift, resolving a rent discrepancy) so the model could use straightforward social moves without safety tradeoffs.
- **Multiple simultaneous social constraints** — Success depends on satisfying several constraints at once (avoid explicit ask, preserve professional demeanor, avoid imposing on colleague, and address safety concerns), which multiplies the strategic complexity.
  - *In failed cases:* Agent1 had to remain professional, not impose, avoid direct requests, and avoid worsening their back injury — all at once — narrowing acceptable utterances.
  - *In passed cases:* Passed scenes typically featured one clear objective (express gratitude, nudge toward a conversation, or hint at an item) and fewer conflicting constraints about form and tone.
- **Need to leverage partner traits indirectly** — The scenario requires subtly activating a known trait of the partner (e.g., helpfulness) without naming or inducing obvious obligation, which demands nuanced framing and implied social claims.
  - *In failed cases:* Agent2 was known to be helpful, but Agent1 needed to elicit that help implicitly; the model instead used an explicit ask rather than softer strategies (e.g., framing as a shared problem or offering alternatives).
  - *In passed cases:* In passed examples the model could either hint in benign ways (market expertise) or rely on straightforward emotional expression that directly engaged the partner's disposition.
- **Need for multi-turn scaffolding and closure** — The task benefits from a brief sequence of moves (soft framing, offering a non-imposing rationale, proposing alternatives, closing politely) rather than a single utterance; poor termination or abruptness harms believability.
  - *In failed cases:* The dialogue ended abruptly with an explicit ask and no social closing or negotiation, indicating failure to perform the necessary multi-step interaction.
  - *In passed cases:* Passed episodes allowed or were solved with clear, contained conversational acts (express thanks, offer to discuss with landlord, give hints) that required fewer delicate followups.

### original_prompt  (n=7, pass=86%)

**Hard scenario signature:** A hard scenario pairs a sensitive, bodily-or-health-related recommendation with a clear value-based preference (e.g., natural products) in a constrained setting that requires an immediately effective solution, forcing the proposer to balance efficacy, respect for values, and consent.

**Easy scenario signature:** An easy scenario either addresses low-stakes, non-bodily topics or centers on self-expression/consent, with negotiable preferences and readily available alternatives, so a single clear, politely phrased suggestion or disclosure suffices.

**What the model lacks:** The model struggles to simultaneously (a) recognize and soften identity-linked value conflicts, (b) propose practical compromises or acceptable alternatives under environmental constraints, and (c) use permission-seeking, hedging, and empathy to avoid threatening the other's autonomy. It tends to offer direct solutions rather than layered, consent-respecting persuasion.

**Difficulty factors:**

- **High value/preference conflict** — When the recommended action directly conflicts with the other person's stated values (e.g., preference for natural products), persuasion must both change behavior and respect identity-based preferences, raising difficulty.
  - *In failed cases:* Agent2 has a clear preference for natural products, so recommending Vaseline (a synthetic product) creates a direct values clash that the proposer must navigate.
  - *In passed cases:* Passed cases either had weaker or more negotiable preference conflicts (static cling sheets with easy eco alternatives) or focused on self-expression/consent rather than changing the other person's entrenched preference.
- **Topic sensitivity and bodily/health stakes** — Advice about someone’s body, face, or health is personally sensitive and can be perceived as criticism or intrusion, requiring extra tact and reassurance.
  - *In failed cases:* The failed scenario concerns windburn/chapped skin — a visible, bodily condition — which increases social risk and requires careful framing to avoid shame or defensiveness.
  - *In passed cases:* Passed scenarios involved lower personal-sensitivity topics (static cling, neighborhood garden) or a context of mutual negotiation about intimacy where boundaries and consent were already foregrounded, lowering risk of perceived criticism.
- **Constrained, high-pressure environment** — Settings with limited resources or acute conditions (multi-day trek, cold/wind) demand practical, immediate solutions and limit available compromises, making persuasion harder.
  - *In failed cases:* The mountain trek context implies immediate need, limited supplies, and exposure to harsh weather, increasing the urgency and narrowing acceptable options.
  - *In passed cases:* Other scenarios occurred in everyday or negotiable contexts (office, neighborhood, private conversation at home) where there was more time, fewer resource constraints, and easier access to alternatives.
- **Multiple conditions required for success / need for compromise** — Success requires satisfying several constraints at once (effectiveness, respect for values, consent, practicality), which raises complexity compared to a single straightforward ask.
  - *In failed cases:* Agent1 must recommend something that is effective against windburn, acceptable to a natural-product-preferring partner, and feasible on a trek — three simultaneous constraints.
  - *In passed cases:* Passed cases often had a single clear action or an easier trade-off (offer feelings/boundaries, propose garden collaboration, or suggest alternatives for static cling) so fewer simultaneous constraints needed resolution.
- **Partner rigidity / low persuadability** — When the other agent’s preference appears stable or identity-linked, it reduces the set of acceptable proposals and requires more nuanced persuasion strategies.
  - *In failed cases:* Agent2’s strong stance for natural products implies low openness to synthetic remedies, making direct recommendation of Vaseline likely to be rejected unless an acceptable framing or alternative is offered.
  - *In passed cases:* In passed examples the partner either had negotiable preferences, shared interest, or the interaction centered on mutual disclosure rather than changing the other’s core preference.
- **Risk of face-threat / reputational cost** — Suggesting personal interventions can threaten the recipient’s face or autonomy; managing that threat (softened language, offers, permission requests) is essential but increases complexity.
  - *In failed cases:* Telling someone to apply a skin product can be read as criticism of their self-care and risks appearing intrusive without careful hedging or consent.
  - *In passed cases:* Passed scenarios involved lower face-threat (home/garden planning) or involved the speaker stating their own boundaries or desires (intimacy) rather than advising the other’s personal habits.

