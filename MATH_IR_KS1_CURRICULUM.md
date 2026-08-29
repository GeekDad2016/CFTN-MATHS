# Canonical KS1 mathematics curriculum

This dataset is a new, isolated path. It does not replace or mutate the V4 data.

Each record separates four contracts:

- `natural_language_prompt`: input for dispatcher-language training.
- `dispatcher_target`: route, criterion and canonical `math_ir` translation.
- `math_ir`, `derivation`, `target_trace`, `answer`: language-free maths-tower supervision.
- `verifier_spec`: deterministic recomputation of the answer and derivation.

The first five phases follow a bounded KS1 dependency order: Year 1 number
structure, Year 1 addition/subtraction fluency, Year 2 place value and crossing
ten, Year 2 addition/subtraction within 100, then multiplication/division by 2,
5 and 10. Validation and test objects use unseen operand combinations inside
the active taught domain. Future phases are forbidden from training and
transition evaluation.

After phase A, an epoch is intended to contain 75% active-phase examples and
25% criterion-balanced replay sampled across every previously accepted phase.
Transition requires both active mastery and cumulative retention. The replay
policy is recorded in the sealed manifest; trainer integration is a separate
fail-closed step.

The builder and auditor stream JSONL. The auditor places only hashes and split
labels in temporary SQLite tables, so duplicate and split-leak checks do not
require loading the dataset into memory.

```bash
python -m tools.build_math_curriculum_dataset prepare \
  --config config/math_ir_ks1_curriculum_v1.json \
  --output /workspace/cftn-text/data/math_ir_ks1_v1

python -m tools.build_math_curriculum_dataset summary \
  --output /workspace/cftn-text/data/math_ir_ks1_v1

python -m tools.build_math_curriculum_dataset sample \
  --output /workspace/cftn-text/data/math_ir_ks1_v1 --split validation --limit 3
```

Use `audit` for repeat verification. Do not inspect large JSONL files with an
ad-hoc parser.

On RunPod, `scripts/build_math_ir_ks1_v1.sh` performs the sealed build and
prints the compact summary. It refuses to overwrite an existing sealed output;
set `CFTN_MATH_IR_DATA_ROOT` to a new path for a new dataset revision.
