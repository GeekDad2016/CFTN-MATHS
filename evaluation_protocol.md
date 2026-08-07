# CFTN-Text causal evaluation protocol

## Claim under test

The claim is not merely that a GPT checkpoint and a math transformer coexist.
It is that compact messages in both directions cause the two persistent towers
to solve problems that neither private tower can solve from its own information.

Two evaluation tracks are mandatory:

1. **Shared-view utility:** both towers receive the ordinary complete problem.
2. **Complementary-view causality:** GPT receives role semantics without
   numbers, while the math tower receives shuffled numeric slots without role
   meanings.

Success on the second track cannot replace performance on the first.

## Immutable benchmark

`tools.prepare_synergy_benchmark` selects 1,000 records by hash from each sealed
source split: `test`, `heldout_language`, `extrapolation`, and `compositional`.
It creates one exact counterfactual partner per selected record, producing
4,000 pairs and 8,000 rows. The manifest records source hashes, protocol hash,
builder hash, row hashes, and selected record hashes.

Each pair is adjacent. GPT's private text and role permutation are identical
within the pair. Only the appropriate opaque numeric slot and exact target
change. Counterfactual equations are rejected if they overlap any source split.

## Evaluation arms

| Arm | Readout | Purpose |
|---|---|---|
| GPT alone | GPT with both bridges disabled | Generalist baseline |
| Math alone | Math generation with both bridges disabled | Specialist baseline |
| Full CFTN | Final GPT generation with both bridges active | Candidate system |
| GPT-to-math disabled | Math and final GPT readouts | Directional necessity |
| Math-to-GPT disabled | Final GPT readout | Directional necessity |
| Each direction shuffled | Corresponding downstream readout | Message specificity |
| Both directions shuffled | Final GPT readout | End-to-end bridge dependence |
| GPT-to-math serial readout | Math generation after the GPT message | One-way serial control |
| Fixed-open checkpoint | Final GPT generation | Conventional cross-attention control |

All arms use the same row order, greedy decoding, token budgets, checkpointed
towers, and random seed. Fixed-open and contextual arms are trained separately
with identical data and optimization limits.

## Primary statistics

For each split and the aggregate:

```text
synergy = accuracy(full CFTN) - max(accuracy(GPT alone), accuracy(math alone))
```

Paired bootstrap intervals resample per-example correctness differences. The
report also includes:

- full minus both-shuffled accuracy;
- GPT-to-math direct contribution at the math readout;
- math-to-GPT direct contribution at the final GPT readout;
- results restricted to examples GPT alone gets wrong;
- counterfactual both-correct, answer-change, and correct-delta rates;
- donor-follow rate after swapping math-to-GPT messages within pairs;
- invariance after swapping identical GPT-to-math messages within pairs;
- sender gates, message norms, execution counts, parameters, GPU memory, and
  elapsed time.

## Initial pass gate

- Synergy is at least 10 percentage points.
- The paired 95% confidence interval for synergy is above zero.
- Correct communication beats both-shuffled by at least 10 points.
- Each communication direction contributes at least two points with its
  confidence interval above zero.
- At least 90% of counterfactual pairs are both correct.
- At least 80% of pair-swapped final answers follow the donor math message.
- The final GPT readout is no more than two points below the one-way math
  readout.

The architecture claim is separate: contextual gates must beat a separately
trained fixed-open arm by at least two points, with the paired 95% confidence
interval above zero.

## Ordered post-math workflow

```powershell
py -3.11 -m tools.evaluate_math_tower
py -3.11 -m tools.train_bridges --stage m2g
py -3.11 -m tools.train_bridges --stage bidirectional --view-mode complementary
py -3.11 -m tools.evaluate --checkpoint G:\ctfn-text\artifacts\v1_linear_equations\bridge_bidirectional_contextual_complementary\bridge_bidirectional.best.pth --view-mode shared --output-root G:\ctfn-text\artifacts\v1_linear_equations\evaluation_bidirectional_contextual_shared
py -3.11 -m tools.evaluate_synergy
```

Only after the contextual candidate passes should the fixed-open replication be
trained and compared. No sealed test result is used for checkpoint selection or
hyperparameter tuning.
