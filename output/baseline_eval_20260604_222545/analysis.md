# Baseline Eval Analysis

**Timestamp:** 2026-06-05T01:16:50.136757  
**Learner:** `openai/gpt-5-mini`  **Partner:** `openai/gpt-5-mini`  
**N completed:** 90  **N errors:** 0
**Goal success threshold:** 7.0 (SOTOPIA-PI convention)

## Overall SOTOPIA-EVAL Scores

| Dimension | Mean | Std | Range |
|-----------|------|-----|-------|
| believability | 8.889 | 0.314 | [0,10] |
| relationship | 1.778 | 1.218 | [-5,5] |
| knowledge | 6.111 | 1.386 | [0,10] |
| secret | -0.356 | 1.508 | [-10,0] |
| social_rules | -0.122 | 0.892 | [-10,0] |
| financial_and_material_benefits | 1.233 | 1.633 | [-5,5] |
| goal | 7.689 | 2.346 | [0,10] |
| overall_score | 3.603 | 0.622 | — |

## SOTOPIA-HARD vs. Rest

| Split | N | GOAL mean ± std | Success rate (≥7.0) |
|-------|---|-----------------|-------------|
| SOTOPIA-HARD | 14 | 6.071 ± 3.283 | 50.0% |
| Rest | 76 | 7.987 ± 1.990 | 76.3% |

## By Source (GOAL)

| Source | N | GOAL mean ± std |
|--------|---|-----------------|
| social_chemistry | 20 | 8.500 ± 1.688 |
| normbank | 9 | 8.333 ± 1.563 |
| hand-craft | 6 | 8.167 ± 2.409 |
| mutual_friends | 10 | 8.000 ± 1.789 |
| persuation_for_good | 10 | 7.600 ± 1.562 |
| deal-or-no-deal | 10 | 7.000 ± 1.342 |
| craigslist_bargains | 10 | 6.900 ± 3.239 |
| social_iqa | 15 | 6.867 ± 3.284 |

## SOTOPIA-HARD Failures (7 scenarios)

These are scenarios in the SOTOPIA-HARD set where GOAL < threshold — primary targets for improvement.

- **seed 0** (craigslist_bargains) GOAL=4.0 — One person is selling a brand new 64GB Samsung Galaxy S8 in Midnight Black for $
- **seed 4** (social_iqa) GOAL=4.0 — Two people in a romantic relationship are on a vacation
- **seed 9** (social_iqa) GOAL=0.0 — Conversation between two individuals who share a common dislike for a third pers
- **seed 13** (craigslist_bargains) GOAL=3.0 — One person is offering a Tile Mate Item Tracker for a price of $20.00, while ano
- **seed 23** (social_iqa) GOAL=4.0 — Two friends are camping in the wilderness and the temperature drops significantl
- **seed 59** (craigslist_bargains) GOAL=3.0 — One person is offering a Dresser and Matching Night Stand for $200.00, while ano
- **seed 67** (social_chemistry) GOAL=3.0 — Two friends are discussing their plans to go on a weekend trip

## All Failures by Source

**craigslist_bargains** (4 failures):
  - seed 74 GOAL=2.0 — One person is offering a 47 inch LED TV for a price of $349.0, while a
  - seed 13 🔴 HARD GOAL=3.0 — One person is offering a Tile Mate Item Tracker for a price of $20.00,
  - seed 59 🔴 HARD GOAL=3.0 — One person is offering a Dresser and Matching Night Stand for $200.00,
  - seed 0 🔴 HARD GOAL=4.0 — One person is selling a brand new 64GB Samsung Galaxy S8 in Midnight B

**deal-or-no-deal** (5 failures):
  - seed 80 GOAL=5.0 — Two friends have just finished their lunch and they have 3 apples, 2 b
  - seed 14 GOAL=6.0 — Two friends are moving out from a shared apartment and dividing their 
  - seed 19 GOAL=6.0 — Two friends are moving out of their shared apartment and need to divid
  - seed 33 GOAL=6.0 — Two friends are at a picnic and have just finished their lunch. They h
  - seed 35 GOAL=6.0 — Two roommates deciding on how to divide certain items that they bought

**hand-craft** (1 failures):
  - seed 77 GOAL=3.0 — Conversation between two friends at a tea party

**mutual_friends** (3 failures):
  - seed 3 GOAL=5.0 — 2 strangers are meeting at a party. <p viewer="environment">They have 
  - seed 36 GOAL=5.0 — 2 strangers are meeting at a party. <p viewer="environment">They have 
  - seed 65 GOAL=6.0 — 2 strangers are meeting at a party. <p viewer="environment">They have 

**normbank** (1 failures):
  - seed 37 GOAL=5.0 — Two friends are hanging out at home and deciding what music to listen 

**persuation_for_good** (3 failures):
  - seed 39 GOAL=5.0 — A conversation between two individuals at a charity gala
  - seed 22 GOAL=6.0 — Conversation taking place in an annual charity event between two atten
  - seed 50 GOAL=6.0 — A conversation between two friends during a charity fundraiser event. 

**social_chemistry** (2 failures):
  - seed 67 🔴 HARD GOAL=3.0 — Two friends are discussing their plans to go on a weekend trip
  - seed 55 GOAL=6.0 — Two friends discussing their schedules at a coffee shop

**social_iqa** (6 failures):
  - seed 9 🔴 HARD GOAL=0.0 — Conversation between two individuals who share a common dislike for a 
  - seed 6 GOAL=2.0 — Two inmates are given the chance to chat briefly before one of them is
  - seed 45 GOAL=3.0 — Two friends meeting in a coffee shop after a long time.
  - seed 4 🔴 HARD GOAL=4.0 — Two people in a romantic relationship are on a vacation
  - seed 23 🔴 HARD GOAL=4.0 — Two friends are camping in the wilderness and the temperature drops si
  - seed 31 GOAL=6.0 — A conversation between two friends at a park

## Interpretation

⚠️ **Model is near ceiling on SOTOPIA overall** (GOAL ≥ 7.5). Consider evaluating on SOTOPIA-HARD only, or switching to a weaker learner model.

SOTOPIA-HARD GOAL = 6.07 (GPT-4 baseline: ~4.85 with human partner, ~7.62 overall). Hard set shows headroom — good primary eval target.
