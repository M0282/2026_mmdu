# MMMU-val Baseline Evaluation Report — Qwen3-VL-4B-Instruct

- **팀명**: 8조
- **팀원**: 우성한, 김소리, 이든솔, 함우림
- **작성일**: 2026년 9월 28일
- **재현 커맨드**: `(예: bash scripts/run_mmmu_eval.sh)`

---

## 1. 환경 / 재현성

| 항목 | 값 |
|---|---|
| 모델 checkpoint | `Qwen/Qwen3-VL-4B-Instruct` (ebb281ec70b05090aa6165b016eac8ec08e71b17) |
| 추론 백엔드 | Hugging Face `transformers` (GitHub main 브랜치 설치, `Qwen3VLForConditionalGeneration` 클래스 사용 — 정식 pip 배포 전 버전이라 정확한 버전 태그 대신 `requirements.txt`에 설치 시점 커밋 기록) |
| 사용 GPU | NVIDIA A100-SXM4-40GB (Colab Pro) |
| 실측 peak VRAM | 미측정 (`torch.cuda.max_memory_allocated()` 로깅을 추가하지 않음 — 한계로 8번에 기록) |
| 총 소요 시간 | 487.4초 (약 8.1분) / 900문제 |
| 의존성 | `requirements.txt` (`pip freeze` 결과, `scripts/` 참고) |
| 실행 커맨드 | ```bash\npython scripts/run_mmmu_eval.py \\\n    --model_path Qwen/Qwen3-VL-4B-Instruct \\\n    --data_root MMMU/MMMU \\\n    --output_path results/mmmu_results.json\n``` |

## 2. 프롬프트

**실제 모델에 들어간 프롬프트 전문** (변수 부분은 `{}`로 표시):

객관식(multiple-choice) 문제:
{question}
(A) {option_A}
(B) {option_B}
(C) {option_C}
(D) {option_D}
Answer with the option's letter from the given choices directly, without additional explanation.

주관식(open-ended) 문제 (선택지가 없는 경우):
{question}
Answer the question using a single word or phrase, without additional explanation.


- **출처**: MMMU 공식 GitHub repo(`MMMU-Benchmark/MMMU`)의 `eval/data_utils.py`에 있는 `construct_prompt` 함수 형식을 그대로 이식 (https://github.com/MMMU-Benchmark/MMMU/blob/main/eval/data_utils.py)
- **선택 이유**: 공식 벤치마크 평가 스크립트와 동일한 프롬프트 형식을 사용해야, 우리가 측정한 정확도를 공식 발표 수치(67.4)와 같은 조건에서 비교할 수 있기 때문. 자체 설계 프롬프트를 쓰면 성능 차이가 "모델 성능 문제"인지 "프롬프트 문제"인지 구분할 수 없게 됨. 마지막의 "without additional explanation" 문구는 공식 템플릿에 없는 부분으로, 실행 시간 제약상 모델이 장황한 풀이과정을 쓰지 않고 바로 답을 내도록 유도하기 위해 추가함.

## 3. 생성(Decoding) 설정

### 3.1 Sampling recipe

| 파라미터 | 값 |
|---|---|
| `do_sample` | `true` |
| `temperature` | `0.7` |
| `top_p` | `0.8` |
| `top_k` | `20` |
| `repetition_penalty` | `1.0` |
| `presence_penalty` | `1.5` |
| `seed` | `42` |

- **출처**: Qwen 공식 HuggingFace 모델카드(`huggingface.co/Qwen/Qwen3-VL-4B-Instruct`) "Generation Hyperparameters > VL" 섹션에 명시된 값을 그대로 사용. `presence_penalty`는 `transformers.generate()`가 기본 지원하지 않는 파라미터라, 커스텀 `LogitsProcessor`(생성된 토큰 중 이미 등장한 토큰에 고정 페널티 부여)를 직접 구현해서 공식 recipe를 그대로 재현함.

### 3.2 생성 예산 / 이미지 해상도

| 파라미터 | 값 |
|---|---|
| `max_new_tokens` | `512` |
| 이미지 해상도 처리 (`min_pixels`/`max_pixels` 등) | `min_pixels=256*28*28`(약 20만 px), `max_pixels=1280*28*28`(약 100만 px) |

**선택 근거**: 초기 파일럿(`max_new_tokens=128`)에서는 계산 문제(Accounting 등) 풀이 도중 응답이 잘리는 사례가 발견되어 예산을 늘릴 필요가 있었음. 반면 `768`까지 늘리자 모델이 매 문제마다 장황한 풀이 과정을 서술하며 예산을 거의 다 소진해 문제당 30초 이상, 900문제 기준 약 8시간이 소요될 것으로 추정되어 비현실적이었음. 이에 프롬프트에 "without additional explanation" 지시문을 추가해 모델이 풀이 과정 없이 바로 답하도록 유도하고, `max_new_tokens`는 512로 재조정함. 그 결과 900문제를 487초(약 8분)에 완료했으며, 응답 잘림 없이 정상적으로 답변이 수집됨을 확인함.


## 4. 채점(파싱) 방식

- 사용한 파서/로직: MMMU 공식 GitHub repo(`MMMU-Benchmark/MMMU`)의 `eval/eval_utils.py`에 있는 `parse_multi_choice_response`(객관식용), `parse_open_response` + `eval_open`(주관식용) 그대로 이식
- 동작 방식 요약:
  - MMMU는 객관식과 주관식(open-ended) 문제가 섞여 있어, `options` 필드 유무로 먼저 유형을 구분함
  - **객관식**: ① 응답에서 `(A)`처럼 괄호로 감싼 선택지 문자를 우선 탐색 → ② 없으면 괄호 없이 단독으로 등장하는 문자 탐색 → ③ 그래도 없으면 선택지 텍스트 자체가 응답에 언급됐는지 탐색 → ④ 후보가 여러 개면 응답에서 가장 마지막에 등장한 것을 채택 → ⑤ 그래도 못 찾으면 랜덤 선택(seed 고정)
  - **주관식**: 응답 문장에서 "so", "therefore", "answer" 등 결론을 나타내는 단서 뒤의 구절을 추출하고, 숫자 형태면 정규화해서 정답과 비교 (부분 문자열 포함 여부로 판정)

**검증**: 전체 오답 433건 중 객관식 388건을 전수 확인한 결과, 파서가 답을 전혀 추출하지 못해 원문이 그대로 남은 "진짜 파싱 실패" 사례는 0건이었다. 즉 객관식 오답은 모두 파서가 유효한 선택지(A/B/C/D 등)를 정상적으로 추출했으나 정답과 달랐던 경우로, 측정 도구의 결함이 아닌 모델의 실제 추론 오류로 판단된다. 다만 주관식(45건)은 부분 문자열 매칭 방식의 특성상 단위·서술 방식 차이로 인한 과소평가 가능성이 남아있어 7번 섹션에서 별도로 논의함.

## 5. 결과

| No. | Subject | Data Num | Acc |
|---|---|---|---|
| 1 | Accounting | 30 | 40.00% |
| 2 | Agriculture | 30 | 56.67% |
| 3 | Architecture_and_Engineering | 30 | 36.67% |
| 4 | Art | 30 | 56.67% |
| 5 | Art_Theory | 30 | 80.00% |
| 6 | Basic_Medical_Science | 30 | 70.00% |
| 7 | Biology | 30 | 50.00% |
| 8 | Chemistry | 30 | 26.67% |
| 9 | Clinical_Medicine | 30 | 60.00% |
| 10 | Computer_Science | 30 | 43.33% |
| 11 | Design | 30 | 80.00% |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 36.67% |
| 13 | Economics | 30 | 50.00% |
| 14 | Electronics | 30 | 43.33% |
| 15 | Energy_and_Power | 30 | 43.33% |
| 16 | Finance | 30 | 36.67% |
| 17 | Geography | 30 | 46.67% |
| 18 | History | 30 | 70.00% |
| 19 | Literature | 30 | 80.00% |
| 20 | Manage | 30 | 40.00% |
| 21 | Marketing | 30 | 63.33% |
| 22 | Materials | 30 | 36.67% |
| 23 | Math | 30 | 46.67% |
| 24 | Mechanical_Engineering | 30 | 30.00% |
| 25 | Music | 30 | 36.67% |
| 26 | Pharmacy | 30 | 70.00% |
| 27 | Physics | 30 | 40.00% |
| 28 | Psychology | 30 | 70.00% |
| 29 | Public_Health | 30 | 53.33% |
| 30 | Sociology | 30 | 63.33% |
| | **Overall (macro avg)** | **900** | **51.89%** |

계산식: `Overall = mean(30개 과목 accuracy)` (json의 `subject_accuracy`와 동일)

## 6. 공식 수치와의 비교

| | Overall (MMMU val) |
|---|---|
| 공식 (Qwen3-VL Technical Report) | 67.4 |
| 우리 재현 결과 | 51.89 |
| 차이 (Δ) | -15.51%p |

## 7. 격차 분석

과목별 정확도를 보면 격차의 원인이 명확한 패턴을 보인다. 화학(26.7%), 기계공학(30.0%), 건축/공학(36.7%), 진단검사의학(36.7%), 재무(36.7%) 등 **계산·수치 추론이 필요한 과목**은 정확도가 크게 낮은 반면, 약학(70.0%), 심리학(70.0%), 미술이론(80.0%), 디자인(80.0%), 문학(80.0%) 등 **개념·언어 기반 과목**은 상대적으로 높은 정확도를 보였다. 이는 격차의 상당 부분이 측정 방식(프롬프트/파싱)의 결함이 아니라, 4B 규모 모델이 화학식·회로도·재무제표 등 복잡한 시각적 수치 정보를 정확히 읽고 다단계 계산을 수행하는 능력이 제한적이라는 실제 모델 한계에서 기인함을 시사한다. 실제 오답 사례(예: Chemistry에서 이성질체 명칭을 혼동, Accounting에서 계산 결과값 자체가 틀림)도 파싱 실패가 아닌 순수한 추론 오류였다. 다만 주관식(open-ended) 문제는 전체 오답 433건 중 45건을 차지했는데, 이 유형은 부분 문자열 일치로 채점하므로 단위 표기나 서술 방식 차이로 실제 정답이 오답 처리됐을 가능성을 배제할 수 없어 추가 검증이 필요하다.

## 8. 기타 특이사항 / 한계 (Optional)

- 초기 파일럿(`max_new_tokens=128`, 정규식 기반 자체 파서)에서는 응답 잘림과 파싱 실패로 인한 오채점이 다수 관찰되어, MMMU 공식 파서/프롬프트로 교체하고 생성 예산을 재조정하는 과정을 거쳤음 (3.2절 참고)
- peak VRAM을 실측하지 못함 — 재실행 시 `torch.cuda.max_memory_allocated()` 추가 예정
- 주관식(open-ended) 채점의 부분 문자열 매칭 방식이 실제 정답을 과소평가할 가능성 있음 (7절 참고), 시간 관계상 수동 재검증은 진행하지 못함
