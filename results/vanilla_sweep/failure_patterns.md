# Vanilla Sweep — Failure Pattern Analysis

**Episodes analyzed:** 157 failed scenarios (out of 734 total, 78.6% pass rate)
**Model:** gpt-5-mini (vanilla, no ICL)

## Overall Diagnosis

Vanilla gpt-5-mini shows a consistent pattern of stopping too soon and failing to convert social rapport or information into concrete, enforceable outcomes. It often secures polite or tentative agreement but does not extract explicit commitments, lock in next steps, or confirm follow-through, which turns many near-wins into failures. The model also exhibits two relational failure modes: being overly pushy/repetitive or, conversely, withdrawing abruptly and coldly — both damage long-term rapport. In negotiation and bargaining contexts it tends to either accept suboptimal partials or abandon the interaction rather than craft creative tradeoffs or correct arithmetic. Finally, the model under-addresses core principled objections (safety, moral, privacy) and therefore cannot reliably persuade when those concerns are central. Improving sustained turn-taking, explicit closing moves, objection-handling scripts, and repair-oriented empathy would likely reduce a large fraction of these failures.

## Core Capability Gaps

**closing and commitment extraction** — Fails to translate verbal agreement into explicit, timebound commitments, confirmations, or actionable next steps.
  *Affects patterns: failed to secure explicit commitment / confirmation, poor conversion of discovered information into action, settled for partial / conditional commitments*
**sustained interaction management** — Tends to exit conversations prematurely or fail to maintain iterative back-and-forth needed to resolve impasses and finalize deals.
  *Affects patterns: premature_exit / abrupt departure, walking away instead of negotiating tradeoffs*
**sensitive rapport maintenance and repair** — Does not pair disclosures or boundary-setting with reparative language and steps; mishandles emotionally charged topics and can damage relationships.
  *Affects patterns: damaging rapport via mishandled disclosures or cold withdrawal, overly pushy, repetitive, or interrogative behavior*
**core objection handling and risk reframing** — Fails to identify and directly rebut principled objections (safety, moral, privacy), instead offering weak or tangential counterarguments.
  *Affects patterns: failure to surface or rebut core objections (safety, moral, privacy), overly pushy, repetitive, or interrogative behavior*
**strategic negotiation and creative concessions** — Struggles to craft value-preserving tradeoffs, arithmetic-correct concessions, or alternative offers that bridge gaps in negotiations.
  *Affects patterns: walking away instead of negotiating tradeoffs, poor conversion of discovered information into action*

## Top Failure Patterns

### 1. premature_exit / abrupt departure (n≈48)

The model frequently ends interactions abruptly or walks away mid-negotiation instead of sustaining the dialogue to closure. This leaves potential agreements, clarifications, and follow-ups unresolved and turns achievable goals into failures.

*Representative tags:* `walked_out_without_speaking`, `abrupt_exit_no_closure`

### 2. failed to secure explicit commitment / confirmation (n≈62)

The model often obtains verbal or tentative agreement but fails to extract explicit, concrete commitments, confirmations, or next steps. Without locking down who will do what and when, interactions stall and the stated social goals are not realized.

*Representative tags:* `failed_to_confirm_choice`, `no_final_commitment`

### 3. settled for partial / conditional commitments (n≈36)

Rather than pushing for the target outcome, the model frequently accepts smaller, conditional, or trial-level concessions. These partial wins preserve surface agreement but leave the primary objective unmet or only partially achieved.

*Representative tags:* `settled_for_partial_amount`, `partial_commitment_only`

### 4. walking away instead of negotiating tradeoffs (n≈22)

When faced with resistance or price/position gaps the model often abandons the interaction instead of creatively trading, probing for priorities, or reframing offers. This yields missed deals and unexploited leverage.

*Representative tags:* `abandoned_negotiation`, `walked_away_without_agreement`

### 5. damaging rapport via mishandled disclosures or cold withdrawal (n≈20)

On emotionally sensitive topics the model either discloses bluntly without reparative steps or withdraws in a socially cold way, causing relationship damage. It fails to pair honesty or boundary-setting with empathy and recovery tactics.

*Representative tags:* `hurt_relationship_with_confession`, `cold_withdrawal_during_distress`

### 6. overly pushy, repetitive, or interrogative behavior (n≈15)

The model sometimes resorts to rapid-fire questioning, repeated pressure, or a demanding tone that undermines persuasion and makes partners defensive. This reduces credibility and often backfires on both goal attainment and rapport.

*Representative tags:* `overly_direct_interrogation`, `became_overly_pushy`

### 7. failure to surface or rebut core objections (safety, moral, privacy) (n≈30)

The model often fails to directly engage core principled objections (safety, ethics, privacy) and instead either skirts the issue or offers weak rationales. Without addressing the root objection, persuasion stalls even when rapport is intact.

*Representative tags:* `failed_to_address_core_resistance`, `ignored_safety_concerns`

### 8. poor conversion of discovered information into action (n≈28)

Even when the model elicits useful facts (bottom prices, preferences, author identity), it often fails to translate that information into concrete next steps or better offers. The result is missed closings and unexploited leverage.

*Representative tags:* `missed_closing_opportunity`, `failed_to_counter_effectively`

## Breakdown by Interaction Type

| Type | N failed | Dominant pattern | Notes |
|------|----------|-----------------|-------|
| normbank | 22 | premature_exit / abrupt departure | Normbank tasks often involve negotiating etiquette or timing that require follow |
| social_chemistry | 18 | failed to secure explicit commitment / confirmation | Many social_chemistry scenarios require sustained rapport and explicit buy-in; t |
| social_iqa | 14 | failed to secure explicit commitment / confirmation | These info-seeking or social QA interactions often need a clear action (e.g., at |
| hand-craft | 9 | poor conversion of discovered information into action | Hand-craft episodes need concrete coordination and follow-up; the model can brai |
| deal-or-no-deal | 8 | walking away instead of negotiating tradeoffs | Negotiation tasks require iterative tradeoffs and arithmetic; the model sometime |
| craigslist_bargains | 6 | walking away instead of negotiating tradeoffs | Price-negotiation contexts expose the model's weak creative bargaining — it ofte |
| original_creation | 6 | settled for partial / conditional commitments | Creative persuasion often ends with tentative trials or partial adoption because |
| mutual_friends | 3 | poor conversion of discovered information into action | Mutual-friends tasks require targeted questioning and verification; the model te |

## Failure Tag Frequency

| Tag | Count |
|-----|-------|
| failed_to_secure_commitment | 4 |
| settled_for_partial_commitment | 2 |
| conceded_too_quickly | 2 |
| left_before_finalizing | 2 |
| settled_for_partial_amount | 1 |
| overly_direct_interrogation | 1 |
| failed_to_close_formal_commitment | 1 |
| deferred_compliance | 1 |
| settled_for_insufficient_time | 1 |
| became_overly_pushy | 1 |
| hurt_relationship_with_confession | 1 |
| failed_to_close_sale | 1 |
| overstepped_by_pushing | 1 |
| abrupt_exit_damaged_rapport | 1 |
| repetitive_approach_undermined | 1 |
| walked_out_without_speaking | 1 |
| damaged_trust_with_confession | 1 |
| accidentally_revealed_secret | 1 |
| overly_transactional_approach | 1 |
| abrupt_exit_after_success | 1 |
| failed_to_address_core_resistance | 1 |
| refused_to_follow_instructions | 1 |
| failed_to_maintain_stance | 1 |
| cold_withdrawal_during_distress | 1 |
| walked_away_without_closing | 1 |
| abandoned_negotiation | 1 |
| arithmetic_errors_suboptimal | 1 |
| failed_to_push_for_high_value | 1 |
| abandoned_after_impasse | 1 |
| failed_to_target_top_preference | 1 |
| left_without_agreement | 1 |
| walked_away_without_agreement | 1 |
| failed_to_get_commitment | 1 |
| failed_to_confirm_common_friend | 1 |
| pursued_wrong_leads | 1 |
| searched_only_for_nonmatches | 1 |
| walked_away_over_price_gap | 1 |
| failed_to_counter_effectively | 1 |
| missed_closing_opportunity | 1 |
| couldnt_overcome_fixed_price | 1 |
| unable_to_bridge_value_gap | 1 |
| abandoned_trip_compromise | 1 |
| failed_to_secure_trial_action | 1 |
| missed_closing_request | 1 |
| conceded_without_conversion | 1 |
| provoked_unnecessary_public_alarm | 1 |
| abrupt_exit_no_closure | 1 |
| left_before_followthrough | 1 |
| failed_to_secure_joint_participation | 1 |
| abrupt_exit_during_negotiation | 1 |
| failed_to_locate_heirloom | 1 |
| settled_for_partial_compromise | 1 |
| missed_hidden_detail | 1 |
| replaced_target_with_substitute | 1 |
| investigation_incomplete | 1 |
| overplanned_no_execution | 1 |
| accepted_under_target_price | 1 |
| refused_to_budge_no_sale | 1 |
| lost_perspective_role_confusion | 1 |
| partial_commitment_only | 1 |
| won_trust_not_conviction | 1 |
| relationship_damaged_by_pressure | 1 |
| gave_up_too_early | 1 |
| did_not_confirm_final_acceptance | 1 |
| ignored_moral_objections | 1 |
| failed_to_address_privacy_concerns | 1 |
| failed_to_get_action_commitment | 1 |
| failed_to_counter_safety_concerns | 1 |
| agreement_too_conditional | 1 |
| ignored_repeated_refusals | 1 |
| failed_to_obtain_concrete_consent | 1 |
| process_over_immediate_action | 1 |
| accepted_clear_rejection | 1 |
| respected_boundaries_over_goal | 1 |
| compromised_but_not_core | 1 |
| secured_only_test_inclusion | 1 |
| allowed_uncertain_exception | 1 |
| failed_to_engage_playfully | 1 |
| did_not_close_deal | 1 |
| left_without_confirmation | 1 |
| silent_departure_no_notice | 1 |
| insufficient_value_alignment | 1 |
| failed_to_secure_buy_in | 1 |
| failed_to_close_support | 1 |
| left_without_confirming_agreement | 1 |
| failed_to_address_objection | 1 |
| left_before_confirming_schedule | 1 |
| failed_to_fully_resolve | 1 |
| sought_unethical_support | 1 |
| left_without_closing_deal | 1 |
| failed_to_demonstrate_control | 1 |
| abrupt_exit_after_confrontation | 1 |
| failed_to_connect_emotionally | 1 |
| ignored_safety_concerns | 1 |
| failed_to_offer_concrete_swap | 1 |
| left_without_reply_to_counteroffer | 1 |
| ignored_partner_risks | 1 |
| pivoted_to_policy_bypass | 1 |
| abrupt_exit_before_closing | 1 |
| failed_to_overcome_objection | 1 |
| secured_followup_not_permission | 1 |
| no_action_plan_confirmed | 1 |
| abrupt_exit_after_counter | 1 |
| left_without_acknowledgement | 1 |
| delayed_decision_pending | 1 |
| concession_not_commitment | 1 |
| shifted_to_safe_alternative | 1 |
| failed_to_persuade_due_to_abrupt_exit | 1 |
| no_final_commitment | 1 |
| explicit_request_against_constraints | 1 |
| abrupt_exit_prevents_resolution | 1 |
| failed_to_address_core_doubts | 1 |
| support_without_policy_change | 1 |
| accepted_refusal_without_persuasion | 1 |
| partial_agreement_only | 1 |
| awaiting_approval_no_settlement | 1 |
| accepted_partial_commitment | 1 |
| boundary_set_but_alienated | 1 |
| secured_plan_not_trust | 1 |
| accepted_no_without_followup | 1 |
| left_conversation_abruptly | 1 |
| secured_temporary_possession | 1 |
| abrupt_exit_ignored_compromise | 1 |
| failed_to_negotiate_sequence | 1 |
| obtained_conditional_agreement | 1 |
| failed_to_reverse_decision | 1 |
| abrupt_departure_no_confirmation | 1 |
| elicited_halfhearted_commitment | 1 |
| left_without_acknowledging_compromise | 1 |
| failed_to_convince_on_merits | 1 |
| deferred_commitment_for_review | 1 |
| open_ended_confession | 1 |
| honest_confession_hurt_relationship | 1 |
| no_commitment_to_follow_advice | 1 |
| exposed_prank_mechanics | 1 |
| settled_for_nonbinding_plan | 1 |
| failed_to_confirm_choice | 1 |
| abrupt_conversation_exit | 1 |
| failed_to_confirm_agreement | 1 |
| abruptly_ended_collaboration | 1 |
| walked_away_during_resolution | 1 |
| settled_for_compromise | 1 |
| secured_conditional_trial_only | 1 |
| damaged_relationship_irrevocably | 1 |
| abandoned_negotiation_midway | 1 |
| failed_to_confirm_recommitment | 1 |
| misinterpreted_role_and_context | 1 |
| failed_to_secure_exemption | 1 |
| ghosted_collaborator_post_agreement | 1 |
| no_firm_commitment | 1 |
| secured_temporary_solution_only | 1 |
