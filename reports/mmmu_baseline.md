# MMMU-val Baseline Evaluation Report — Qwen3-VL-4B-Instruct

- **팀명**: 8조
- **팀원**: 우성한, 김소리, 이든솔, 함우림
- **작성일**: 2026년 9월 28일
- **재현 커맨드**:
  1. 환경 설치: `pip install -r scripts/requirements-minimal.txt`
  2. 평가 실행: `python scripts/run_mmmu_eval.py --model_path Qwen/Qwen3-VL-4B-Instruct --data_root MMMU/MMMU --output_path results/mmmu_results.json`
  - 다른 체크포인트를 평가하려면 `--model_path`만 바꾸면 된다 (로컬 fine-tune 체크포인트로 실행해보는 것은 아직 확인하지 않음).

---

## 1. 환경 / 재현성

| 항목 | 값 |
|---|---|
| 모델 checkpoint | `Qwen/Qwen3-VL-4B-Instruct` (ebb281ec70b05090aa6165b016eac8ec08e71b17) |
| Python | 3.13 (Colab 런타임) |
| dtype | `dtype="auto"` (리포지토리 기본값 BF16), 양자화 없음 |
| 데이터셋 | `MMMU/MMMU` (98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68), validation split, 30개 과목 config × 30문제 = 900문제, 서브샘플링 없음 |
| 추론 백엔드 | Hugging Face `transformers` (`Qwen3VLForConditionalGeneration`), transformers는 GitHub 커밋 `07338b6c74a578868368e6e549dea83414e4b8cb`로 고정 |
| 백엔드 선택 이유 | Qwen 모델카드의 공식 사용 예시가 transformers 기반이며, `presence_penalty`를 포함한 공식 sampling recipe를 커스텀 `LogitsProcessor`로 그대로 재현할 수 있어 선택함. 문제를 배치 없이 순차 처리했으나 응답이 짧아 900문제가 약 8분에 끝났으므로 vLLM 등 별도 서빙 프레임워크는 사용하지 않음 |
| 사용 GPU | NVIDIA A100-SXM4-40GB (Google Colab Pro) |
| 실측 peak VRAM | 미측정 (VRAM 로깅을 추가하지 않음, 8번 참고). 참고로 BF16 가중치의 다운로드 크기는 약 8.9GB임 |
| 총 소요 시간 | 487.4초 (약 8.1분) / 900문제 |
| 의존성 | `scripts/requirements-minimal.txt` (핵심 6개 패키지), `scripts/requirements.txt` (Colab 런타임 전체 `pip freeze`) |
| 실행 커맨드 | `python scripts/run_mmmu_eval.py --model_path Qwen/Qwen3-VL-4B-Instruct --data_root MMMU/MMMU --output_path results/mmmu_results.json` (모델은 HF repo id 또는 로컬 경로, 데이터는 HF dataset repo id를 인자로 받으며 절대경로는 코드에 하드코딩하지 않음) |

## 2. 프롬프트

**실제 모델에 들어간 프롬프트 전문** (변수 부분은 `{}`로 표시). 이미지는 `image_1`~`image_7` 중 존재하는 것을 텍스트 앞에 순서대로 배치함.

```
[객관식 문제]
{question}
(A) {option_A}
(B) {option_B}
...
Answer with the option's letter from the given choices directly, without additional explanation.

[주관식(open-ended) 문제: 선택지가 없는 경우]
{question}
Answer the question using a single word or phrase, without additional explanation.
```

- **출처**: MMMU 공식 evaluation 코드(https://github.com/MMMU-Benchmark/MMMU)의 zero-shot 프롬프트 구성 방식(질문, 괄호 표기 선택지, 선택지 문자로 바로 답하라는 지시문)을 따름. 단, 각 지시문 끝의 `without additional explanation`은 우리가 추가한 문구이며 공식 형식에는 없음.
- **선택 이유**: 공식 평가 코드와 같은 형식을 따라야 공식 수치(67.4)와의 비교 조건을 최대한 맞출 수 있기 때문. 추가한 문구는 초기 실행(`max_new_tokens=768`)에서 문제당 약 30초가 걸려 900문제 완료까지 약 8시간이 예상되었기 때문에, 풀이 과정 없이 답만 출력하도록 유도해 실행 시간을 줄이려는 목적으로 넣음. 이 변경이 정확도에 영향을 주었을 가능성은 7번에서 다룸.

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
| `seed` | `42` (`torch.manual_seed`를 실행 시작 시 1회 설정, 파서의 무작위 fallback용 `random.seed(42)`는 별도) |

- **출처**: Qwen 공식 모델카드 https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct 의 Generation Hyperparameters > VL 섹션 (`greedy=false`, `top_p=0.8`, `top_k=20`, `temperature=0.7`, `repetition_penalty=1.0`, `presence_penalty=1.5`). 같은 섹션의 `out_seq_length=16384`는 따르지 않고 `max_new_tokens=512`를 사용함 (3.2 참고).
- **구현 참고**: `transformers.generate()`는 `presence_penalty`를 기본 제공하지 않아, 생성된 토큰 중 한 번이라도 등장한 토큰의 logit에서 고정값(1.5)을 빼는 커스텀 `LogitsProcessor`를 구현함. EOS 토큰은 페널티 대상에서 제외함(EOS가 페널티를 받으면 생성 종료가 억제될 수 있기 때문).

### 3.2 생성 예산 / 이미지 해상도

| 파라미터 | 값 |
|---|---|
| `max_new_tokens` | `512` |
| 이미지 해상도 처리 (`min_pixels`/`max_pixels` 등) | `min_pixels=256*28*28` (200,704), `max_pixels=1280*28*28` (1,003,520) |

**선택 근거**: 초기 파일럿(`max_new_tokens=128`)에서는 계산 문제 풀이 도중 응답이 잘리는 사례가 관찰되었다. 이후 `max_new_tokens=768`로 실행했을 때 초기 3문제 기준 문제당 약 30초가 걸려 900문제 완료까지 약 8시간이 예상되었다. 이에 프롬프트에 `without additional explanation` 지시문을 추가하고 `max_new_tokens`를 512로 낮춰 재실행했으며, 900문제를 487초에 완료했다. 두 가지를 동시에 변경했기 때문에 속도 개선이 각각 어느 변경에서 기인했는지는 분리하지 못했다. 최종 실행에서도 1건(`validation_Math_13`)은 512토큰 상한에서 응답이 문장 중간에 잘렸다.

이미지 해상도는 표·그래프·수식이 포함된 이미지가 많은 MMMU 특성상 과도하게 줄이면 숫자·기호 정보가 손실될 수 있고, 상한을 두지 않으면 VRAM과 처리 시간 부담이 커지므로 두 요소를 절충해 정했다. 이 값에 대한 별도 비교 실험은 수행하지 않았다.

## 4. 채점(파싱) 방식

- 사용한 파서/로직: MMMU 공식 evaluation 코드의 `parse_multi_choice_response`(객관식), `parse_open_response` + `eval_open`(주관식)을 기반으로 이식함. 출처: https://github.com/MMMU-Benchmark/MMMU/blob/bb0b95a945998d91dfef37e969e07d49d1139438/mmmu/utils/eval_utils.py
- 동작 방식 요약:
  - MMMU에는 객관식과 주관식(open-ended)이 섞여 있어, `options` 필드가 비어 있는지 여부로 유형을 먼저 구분함
  - **객관식**: ① 응답에서 `(A)`처럼 괄호로 감싼 선택지 문자를 탐색 → ② 없으면 괄호 없이 단독으로 등장하는 문자를 탐색 → ③ 그래도 없으면 응답이 5단어를 넘을 때 선택지 텍스트 자체가 응답에 포함되는지 탐색 → ④ 후보가 여러 개면 응답에서 가장 마지막에 등장한 것을 채택 → ⑤ 후보가 없으면 선택지 중 무작위 선택(seed 고정)
  - **주관식**: 응답에서 결론을 나타내는 단서(`so`, `therefore`, `answer` 등) 뒤의 구절과 숫자를 추출·정규화한 뒤, 정답 문자열이 포함되는지(숫자는 일치하는지)로 판정

**검증**: 객관식 오답 388건에 대해 위 ⑤의 무작위 fallback 경로(①~③ 중 어느 것도 응답에서 찾지 못한 경우)에 해당하는 건이 있는지 재현해 확인했고, 해당 사례는 0건이었다. 즉 모든 객관식 오답에서 파서가 응답 내 선택지를 추출한 뒤 정답과 비교한 결과가 오답이었다. 다만 정답 처리된 응답은 결과 파일에 저장되지 않아 같은 검증을 하지 못했다. 주관식 오답 45건은 부분 문자열 매칭 특성상 표기 차이로 인한 과소평가 가능성이 남는다.

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

계산식: `Overall = mean(30개 과목 accuracy)`. 과목당 문제 수가 30개로 같아 전체 900문제 중 467문제 정답(467/900)의 비율과 같다.

### 과목별 소요 시간

| No. | Subject | 소요 시간(초) |
|---|---|---|
| 1 | Accounting | 16.8 |
| 2 | Agriculture | 16.8 |
| 3 | Architecture_and_Engineering | 16.7 |
| 4 | Art | 11.7 |
| 5 | Art_Theory | 10.9 |
| 6 | Basic_Medical_Science | 10.4 |
| 7 | Biology | 10.8 |
| 8 | Chemistry | 11.3 |
| 9 | Clinical_Medicine | 11.5 |
| 10 | Computer_Science | 12.4 |
| 11 | Design | 10.5 |
| 12 | Diagnostics_and_Laboratory_Medicine | 11.5 |
| 13 | Economics | 14.6 |
| 14 | Electronics | 14.5 |
| 15 | Energy_and_Power | 14.5 |
| 16 | Finance | 14.7 |
| 17 | Geography | 8.6 |
| 18 | History | 13.8 |
| 19 | Literature | 9.6 |
| 20 | Manage | 8.8 |
| 21 | Marketing | 10.8 |
| 22 | Materials | 16.4 |
| 23 | Math | 115.7 |
| 24 | Mechanical_Engineering | 18.1 |
| 25 | Music | 7.7 |
| 26 | Pharmacy | 11.6 |
| 27 | Physics | 10.4 |
| 28 | Psychology | 10.7 |
| 29 | Public_Health | 12.9 |
| 30 | Sociology | 10.0 |
| | **과목별 합계** | **474.7** |

총 소요 시간은 487.4초이며, 과목별 시간의 합(474.7초)과의 차이는 과목별 측정 밖의 시간으로 추정한다. Math만 115.7초로 다른 과목(7.7~18.1초)보다 길었으며, 512토큰 상한에서 잘린 응답(`validation_Math_13`)이 확인되었으나, 정답 처리된 응답은 저장하지 않았고 문제별 소요 시간도 기록하지 않아 이것이 주원인인지는 확인하지 못했다.

## 6. 공식 수치와의 비교

| | Overall (MMMU val) |
|---|---|
| 공식 (Qwen3-VL Technical Report) | 67.4 |
| 우리 재현 결과 | 51.89 |
| 차이 (Δ) | -15.51%p |

## 7. 격차 분석

종합 정확도는 51.89%로 공식 수치(67.4%)보다 15.51%p 낮았다. 과목별로는 화학(26.67%), 기계공학(30.00%), 건축·공학, 진단검사의학, 재무(각 36.67%)의 정확도가 낮았고, 문학·디자인·미술이론(각 80.00%)은 높았다. 이러한 차이가 발생한 원인으로는 프롬프트에 추가한 without additional explanation 지시문, 공식 수치의 평가 조건과의 차이(공식 평가 설정은 확인하지 못했으며, 참고로 모델카드 권장 out_seq_length=16384에 비해 우리는 max_new_tokens=512를 사용했다), 이미지 내 수치·기호 판독 및 계산 과정의 오류 등을 고려할 수 있다. 특히 오답 응답 길이의 중앙값이 8자로, 많은 경우 풀이 과정 없이 짧은 답을 출력했다. 다만 풀이를 허용한 조건과 직접 비교하지 않았으므로 짧은 응답이 낮은 정확도의 원인인지, 틀린 문제에서 나타난 특징인지는 아직 판단할 수 없다.

max_new_tokens=512에 도달해 응답이 잘린 사례는 확인된 범위에서 1건(validation_Math_13)이므로, 생성 길이 제한만으로 전체 정확도 차이를 설명하기는 어렵다. 객관식 오답 388건에서는 파서의 무작위 fallback 사례가 없었다. 다만 이것만으로 답안 추출과 채점이 모두 정확했다고 판단할 수는 없으며, 정답 처리된 객관식 응답과 주관식 오답 45건에 대한 추가 검증이 필요하다. 그 밖에 4B 모델의 한계, 단일 시드 실행의 분산도 분리 검증하지 못했다. 계산 중심 과목에서 풀이를 허용한 프롬프트로 비교하면 원인을 구분할 수 있다.

## 8. 기타 특이사항 / 한계 (Optional)

- 초기 파일럿(`max_new_tokens=128`, 정규식 기반 자체 파서)에서는 응답 잘림과, 정답을 오답으로 채점한 사례(12건 이상 확인)가 관찰되어 MMMU 공식 파서와 프롬프트 형식으로 교체하고 생성 예산을 재조정했다 (3.2 참고).
- 프롬프트 지시문(`without additional explanation`)과 `max_new_tokens`의 영향을 분리하는 대조 실험은 수행하지 않았다. 최종 실행에서 1건(`validation_Math_13`)의 응답이 512토큰 상한에서 잘렸다.
- 결과는 단일 실행(seed 42)이며 반복 실행에 따른 분산은 측정하지 않았다.
- peak VRAM을 측정하지 못했다.
- 주관식(open-ended) 채점의 부분 문자열 매칭 방식이 실제 정답을 과소평가할 가능성이 있다 (7번 참고).
- `scripts/requirements.txt`는 Colab 런타임 전체를 `pip freeze`한 것이라 `google-colab` 등 Colab 전용 패키지가 포함되어 있어 다른 환경에서는 그대로 설치되지 않을 수 있다. 이를 위해 핵심 6개 패키지만 담은 `scripts/requirements-minimal.txt`를 함께 제공한다. 이 파일로 새 가상환경(Colab GPU 런타임, 미리 설치된 패키지 없음)을 만들어 `--limit_per_subject 2` 스모크 테스트(60문제)를 끝까지 실행해 정상 동작을 확인했으며, 전체 900문제 재실행은 수행하지 않았다. 이 파일은 6개 핵심 패키지만 고정하므로 numpy, pandas 등 간접 의존성과 PyTorch의 CUDA 빌드는 고정되지 않는다. 실제로 새 환경에는 `torch 2.11.0+cu130`이 설치되었고(원래 실행 환경은 `2.11.0+cu128`), GPU 인식은 정상이었다.
