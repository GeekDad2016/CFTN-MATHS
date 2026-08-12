# CFTN-Text V2 evidence-integrated revision

Status: implemented, tested locally, and deliberately blocked from execution
until the sealed V1.3 report exists and passes. The sealed V1.2 report is
versioned under `evidence/`; V1.3 is still running, so no V1.3 result is claimed
here.

## Why V2 was revised

V1.1 established a real one-way result: answer-specific math-to-GPT messages
causally changed GPT's output. It also exposed two failures. GPT-to-math
messages damaged an already-capable specialist in shared-view evaluation, and
the standalone math tower generalized poorly beyond its narrow training
distribution.

V1.2 then passed the targeted repair. It froze the successful return path,
trained only GPT-to-math and its receivers on paired required/redundant views,
used specialist-preservation and correct-versus-shuffled losses, and selected
the checkpoint using greedy causal generation. That mechanism is now a hard V2
dependency rather than a suggestion in documentation.

V1.3 tests the next, different claim: GPT owns the natural prompt while zero,
one, or several specialists receive bounded latent requests, return results,
and can be conditionally skipped. V2 may not inherit that claim merely because
the code exists. The V2 prerequisite audit requires V1.3's sealed final gates
and verifies that its report chains to the exact sealed V1.2 report.

## Executable corrections

The V2 runner now performs these checks and stages in order:

1. verify passed, hash-chained V1.2 and V1.3 mechanism reports;
2. generate and hash the immutable 400K data contract;
3. train the scratch byte-level math transformer for all 12 curriculum epochs;
4. compare the teacher-forced best and retained epoch checkpoints on a
   validation-only greedy-generation panel, then copy the winner by hash;
5. evaluate that selected checkpoint on sealed generalization splits and stop
   unless the generative specialist gate passes;
6. train the math-to-GPT return bridge on shared complete prompts, where the
   frozen specialist can already solve and GPT learns to consume its result;
7. freeze that successful return path and train GPT-to-math with V1.2's paired
   required/redundant objective, preservation loss, causal shuffled margin,
   gate separation, and generation-led checkpoint selection;
8. stop unless shared-view evaluation shows no more than two points of
   specialist regression;
9. run complementary directional, closed, shuffled, and fixed-open causal
   collaboration arms and require positive synergy;
10. allow a later 1M-data proposal only when validation greedy-generation
    accuracy is still improving and the specialist gate passed;
11. assemble one report whose overall result is the conjunction of every gate.

The categorical integer answer head is disabled. It cannot mask failure of
autoregressive sign-and-digit or symbolic generation. Generation is allowed to
use the model's full 1,152-token context rather than the old fixed 160-token
ceiling. Evaluation reports over-context exclusions rather than silently
truncating them.

## Fail-closed behavior

The runner no longer equates “a report file exists” with “the experiment
passed.” Failed specialist, conditional-communication, shared no-harm,
collaboration, or final gates are terminal. A resumed run cannot silently skip
one of those failures.

`evidence/v1_3_final_report.json` is intentionally absent while V1.3 is active.
After V1.3 finishes, copy its sealed report there or set `CFTN_V1_3_REPORT` to
the immutable report path. A failed V1.3 report remains a hard stop; do not
replace it with an override.

## Claim boundary

This V2 runner re-tests broad specialist generation and safe communication on
controlled private views. It does not yet claim that the broad math specialist
has been integrated into V1.3's autonomous natural-prompt wake runtime. The
final V2 report states that boundary explicitly. If V1.3 passes, porting the
broad selected math checkpoint into that runtime and re-running natural-prompt
wake, causality, no-harm, and compute controls is the next integration revision.
If V1.3 fails, repair its failed mechanism first instead of hiding the result
inside a larger B200 run.
