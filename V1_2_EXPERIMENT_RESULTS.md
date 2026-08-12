# CFTN-Text V1.2 experiment results

Status: sealed **PASS**.

Preregistered design: `V1_2_BRIDGE_REVISION.md`

## Executive conclusion

V1.2 passed every central conditional-communication gate. The result supports advancing to V1.3.

## Immutable provenance

- Generated UTC: `2026-08-10T21:10:48.431352+00:00`
- Revision hash: `7277a53ee55aa3605f8da4b6278d2b0291825562cde3965c6647187557c8f556`
- Best epoch: 11
- Best checkpoint: `G:\ctfn-text\artifacts\v1_2_conditional_bridge\bridge_conditional_contextual\bridge_bidirectional.best.pth`
- Best checkpoint SHA-256: `6d3d13eefdcb17f6c848c11baf69f4f60040a8b16f0da34c4882fa2dd3235082`
- W&B run: https://wandb.ai/kaipo/cftn-text/runs/nxzftegn
- Base configuration hash: `6beeecb14596720700d1e4865f996116896753ee37366f2a754e3e95b15d37c6`
- Dataset manifest hash: `66d5243c6af2f33e745d3cfc5eeb86b90b3c735c41cafe120627b3d14e1a8e33`
- Math checkpoint SHA-256: `97a48a484157f10e9ce41a8d9fef8ea8b168180d26ebee00d50f549bc506eb27`
- Source bridge checkpoint SHA-256: `97884cf1697ca97f2d4c2ffcb569b0d2e993a883c1ace51910ff487ffb99d85b`

## What went well

- The selected training checkpoint passed the fixed mixed-necessity generation panel.
- GPT-to-math stayed within the preregistered shared-view specialist regression limit.
- Full CFTN exceeded the strongest isolated arm on complementary tasks.
- Correct bridge messages outperformed shuffled-message controls.
- GPT-to-math made a positive causal contribution.
- Math-to-GPT made a positive causal contribution.
- The revision retained V1.1 familiar and compositional capability.

## What did not pass

- No preregistered central gate failed.

## Shared-view no-harm results

| Split | V1.2 full math | GPT-to-math disabled | Regression | V1.1 regression |
| --- | ---: | ---: | ---: | ---: |
| answer_extrapolation | 0.70% | 0.70% | 0.00% | 0.26% |
| compositional | 0.65% | 0.70% | 0.05% | 0.05% |
| extrapolation | 20.65% | 19.95% | -0.70% | 8.06% |
| heldout_language | 0.70% | 0.70% | 0.00% | 0.17% |
| test | 100.00% | 100.00% | 0.00% | 20.29% |

## Complementary causal results

| Split | V1.2 joint | V1.1 joint | Change | GPT-to-math gain | Correct-vs-shuffled |
| --- | ---: | ---: | ---: | ---: | ---: |
| answer_extrapolation | 1.35% | 1.30% | 0.05% | 1.45% | 1.35% |
| compositional | 86.35% | 87.00% | -0.65% | 85.85% | 85.75% |
| extrapolation | 12.45% | 10.25% | 2.20% | 12.35% | 12.25% |
| heldout_language | 10.35% | 13.55% | -3.20% | 9.90% | 9.85% |
| test | 98.95% | 98.65% | 0.30% | 98.65% | 98.60% |

## Acceptance gates

- PASS — `training_mixed_necessity_gate`
- PASS — `shared_specialist_no_harm`
- PASS — `complementary_synergy_gain`
- PASS — `complementary_correct_vs_shuffled`
- PASS — `complementary_gpt_to_math_gain`
- PASS — `complementary_math_to_gpt_gain`
- PASS — `preserves_v1_1_familiar_and_compositional`

## Why: diagnostic hypotheses

The statements below are hypotheses suggested by the ablations, not measured facts.

- No failure diagnosis is required; remaining limitations concern broader tower knowledge and conditional compute, which V1.2 did not test.

## Recommended fixes or next experiment

- Proceed to the preregistered V1.3 wake-gated recurrent multi-specialist experiment.

## Scope limitation

Tests when GPT-to-math information should cross; it does not test wake-gate conditional compute or multi-specialist routing.

V1.2 reuses frozen V1.1 towers, so broad specialist capability remains a separate V2 question.
