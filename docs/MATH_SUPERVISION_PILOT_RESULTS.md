# Math supervision pilots: preserved negative results

2026-08-26. These are bounded diagnostic experiments, **not production acceptance**.
No long recovery was launched and no `math.best.pth` was promoted.

## First three-arm pilot

All three arms started from the same capacity epoch-5 weights, used 300 updates,
batch 16, identical sampled original questions, 25% replay and fresh optimizers.
Control retained old targets/loss; loss-only changed result weighting; worked
used verified longer solutions plus result weighting. These are equal-example,
not equal-token or equal-time comparisons. There was one seed and small panels.

| Native exact-answer panel | N | Zero update | Control | Loss-only | Worked |
| --- | ---: | ---: | ---: | ---: | ---: |
| Multiplication | 64 | 2 | 2 | 2 | 0 |
| Two-variable systems | 64 | 0 | 2 | 0 | 1 |
| Variables both sides | 32 | 29 | 30 | 30 | 28 |
| Nested parentheses | 32 | 25 | 27 | 29 | 29 |
| Broad diagnostic | 128 | 22 | 21 | 23 | 16 |

**All five prespecified screening conditions failed.** The worked arm hit the
1,024-byte generation cap on 59/64 multiplication and 28/64 systems questions.
No complete prescribed procedure was generated on either targeted panel.
Gold targets fit: maximum 636 multiplication bytes and 520 systems bytes, plus
EOS. Repetitive/incorrect output, not overlong gold solutions, caused these caps.

First prescribed-procedure error in the worked arm:

| First error | Multiplication (64) | Systems (64) |
| --- | ---: | ---: |
| Wrong operand binding | 49 | 29 |
| Wrong computed value | 6 | 2 |
| Malformed/incomplete | 8 | 31 |
| Wrong step/operation | 1 | 2 |

The diagnostic accepts swapped operands for commutative operations and does not
call an unterminated final number an arithmetic error. It checks the prescribed
procedure, not every possible valid derivation. Legacy arms were not trained on
this grammar, so their grammar failures are not comparative arithmetic scores.

Example: `Product of 1525 and 38.11.` starts with a generated
`p0=multiply(1525,11)=1525` rather than the required units digit 1, then repeats.
This motivates separate operand-binding, scaling and short-operation lessons;
it does **not** establish that a particular curriculum will repair transfer.

### Cost and limitations

Run elapsed: 993.64 seconds (16.56 minutes, excluding some preflight). Training
seconds were 19.04 control, 18.85 loss-only and 38.70 worked. Supervised bytes
were 249,940 / 249,940 / 1,439,700. Worked-arm evaluation alone took 198.50s on
multiplication, 215.82s on systems and 442.96s on the broad panel because of long
failed decoding. Peak training allocated memory was approximately 1.88 / 1.88 /
8.89 GiB. No CUDA/non-finite error occurred.

All arms excluded MathQA; this experiment does not isolate removal of inconsistent
MathQA supervision. Teacher-forced first-result metrics use different first
operations across representations and do not establish native improvement.

The original recipe implicitly stripped decimal points before its first visible
step. Its final fraction-to-decimal conversion was also mislabeled as copying.
Commit `484ac63` corrects the latter supervision label. A new derivative was
fully audited: same questions/target text, 1,427 changed verified span records.
This correction passed regression tests but has not itself passed a GPU pilot.

### Provenance and durable evidence

- Pilot code: `e91fa19`; posthoc analyzer: `0d9f247`.
- Source checkpoint SHA-256:
  `2ddb776715b0ee0accfd03e2d98ea4f29cb47c7b4954c02a6beb759150357b08`.
- Protected original SHA-256:
  `fe2c056a1ee1d4a3514537681d82124b0312f45c27f72f8e73d2afc747d53973`.
- Original pilot derivative manifest:
  `c06932ba78d5a8aab2fda168ed87cd3a4a95e4c5c88dfb5662b400bdd0cbe7a1`.
- Corrected derivative manifest:
  `36802610ff89aa8a4232fe6970017a1eb831899c7e6cfd94009bb313285b706e`.
- Pod artifact: `/workspace/cftn-text/artifacts/v2_broad_math_400k_r4/math_verified_supervision_pilot_v1`.
- Local copied reports, contracts, full generations, metrics and logs:
  `C:\CFTN\.runpod\diagnostics\verified-supervision-pilot-20260826`.
- Pilot model files remain on the persistent Pod volume, explicitly non-promotable.

The remote process completed and saved its summary before a local SSH client
lingered without EOF. Only that local client was terminated after checking remote
process absence and an idle GPU. Network instability is possible, not proven;
subsequent pilots must detach and save durable logs to survive SSH interruptions.

## Next bounded test: prerequisites before chains

New synthetic/versioned lessons separate public operand extraction, signed
decimal integer/scale conversion, decimal restoration, small signed multiplication,
subtraction and exact integer division. An answer-only arm and a compact-worked
arm share questions/order/update limits. No imported data is relabeled.

First measure tiny training-set recall (not generalization), then held-out
mathematical objects and replay. Composition may run only after each prerequisite
passes its predeclared gate. Full original native panels remain separate; a
synthetic lesson pass must never be described as a repaired V2 math tower.
