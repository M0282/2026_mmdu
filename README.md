# 2026 MMDL — MMMU Baseline Evaluation

Baseline evaluation project for **Qwen3-VL-4B-Instruct** on the **MMMU validation split**.

## Assignment target

The final submission follows the provided `SUBMISSION_TEMPLATE.md` and must be written at:

- `reports/mmmu_baseline.md`

The final baseline run must use:

- Model: `Qwen/Qwen3-VL-4B-Instruct`
- Model revision: `ebb281ec70b05090aa6165b016eac8ec08e71b17`
- Dataset: `MMMU/MMMU`
- Dataset revision: `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`
- Split: `validation`
- All 30 subject configs, 30 samples each, 900 samples total
- Model dtype: BF16 / repository default (no quantization)

## Repository layout

```text
.
├─ README.md
├─ SUBMISSION_TEMPLATE.md
├─ docs/
│  └─ assignment_guidance.md
├─ reports/
│  └─ mmmu_baseline.md
├─ evaluation/
├─ scripts/
└─ results/
```

Evaluation code, one-command reproduction scripts, exact dependencies, and final results will be added under the corresponding directories.

## Important

Preliminary smoke tests or subsampled runs are engineering checks only. The submitted score must come from the complete 900-sample MMMU validation evaluation under the fixed assignment specification.
