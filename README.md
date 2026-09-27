# 2026 MMDL — MMMU Baseline Evaluation

이 저장소는 제공된 Assignment Guidance와 Submission Template에 맞춰 구성합니다.

## 제출 위치

- 최종 보고서: `reports/mmmu_baseline.md`
- 원본 템플릿: `SUBMISSION_TEMPLATE.md`
- 과제 지침 사본: `docs/assignment_guidance.md`

## 고정 평가 스펙

- Model: `Qwen/Qwen3-VL-4B-Instruct`
- Model revision: `ebb281ec70b05090aa6165b016eac8ec08e71b17`
- dtype: BF16 / repository default, no quantization
- Dataset: `MMMU/MMMU`
- Dataset revision: `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`
- Split: `validation`
- 30 subject configs × 30 samples = 900 samples

평가 스크립트와 재현 커맨드는 최종적으로 같은 저장소에 포함합니다.
