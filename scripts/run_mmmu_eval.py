#!/usr/bin/env python
"""
MMMU validation baseline evaluation for Qwen3-VL-4B-Instruct.

과제: Qwen3-VL-4B MMMU Baseline Evaluation
- 모델/데이터셋 revision은 과제 스펙에서 고정된 값 (MODEL_REVISION, DATA_REVISION 아래 참고)
- Sampling recipe: Qwen 공식 HF 모델카드 "Generation Hyperparameters > VL" 섹션 그대로 사용
  출처: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct#generation-hyperparameters
  (greedy=false, top_p=0.8, top_k=20, temperature=0.7, repetition_penalty=1.0, presence_penalty=1.5)
  * presence_penalty는 transformers.generate()가 기본 제공하지 않는 파라미터라
    PresencePenaltyLogitsProcessor로 직접 구현해서 recipe를 그대로 지켰음 (아래 클래스 참고).
- 파싱(채점) 로직: MMMU 공식 repo(github.com/MMMU-Benchmark/MMMU, eval/eval_utils.py)의
  parse_multi_choice_response / parse_open_response / eval_multi_choice / eval_open 그대로 이식.
  (객관식과 주관식(open-ended) 문제가 섞여 있어, options 필드가 비어 있는지 여부로 유형을
  구분하고 유형별 채점 함수를 적용한다.)

사용법 (한 커맨드로 재현):
    python run_mmmu_eval.py \
        --model_path Qwen/Qwen3-VL-4B-Instruct \
        --data_root MMMU/MMMU \
        --output_path outputs/mmmu_results.json

빠른 스모크 테스트 (과목당 몇 문제만):
    python run_mmmu_eval.py --limit_per_subject 3
"""
import argparse
import json
import random
import re
import string
import time

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, LogitsProcessor

# ---------------------------------------------------------------------------
# 과제 스펙 고정값
# ---------------------------------------------------------------------------
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
DATA_REVISION = "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68"

SUBJECTS = [
    "Accounting", "Agriculture", "Architecture_and_Engineering", "Art", "Art_Theory",
    "Basic_Medical_Science", "Biology", "Chemistry", "Clinical_Medicine", "Computer_Science",
    "Design", "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics",
    "Energy_and_Power", "Finance", "Geography", "History", "Literature", "Manage",
    "Marketing", "Materials", "Math", "Mechanical_Engineering", "Music", "Pharmacy",
    "Physics", "Psychology", "Public_Health", "Sociology",
]

# 공식 sampling recipe (Qwen3-VL-4B-Instruct HF 모델카드, VL 섹션)
SAMPLING_RECIPE = dict(
    do_sample=True,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    repetition_penalty=1.0,
)
PRESENCE_PENALTY = 1.5
SEED = 42


class PresencePenaltyLogitsProcessor(LogitsProcessor):
    """OpenAI/Qwen 스타일 presence_penalty: 지금까지 생성된 토큰 중 한 번이라도
    등장한 토큰이면 등장 횟수와 무관하게 고정 페널티를 적용.
    transformers.generate()에는 이 파라미터가 기본 내장되어 있지 않아 직접 구현함
    (repetition_penalty는 곱셈 기반이라 의미가 다름).
    단, EOS 토큰은 페널티 대상에서 제외한다. EOS가 페널티를 받으면 생성 종료가
    억제되어 max_new_tokens까지 채워질 수 있기 때문이다."""

    def __init__(self, penalty: float, prompt_len: int, eos_token_ids):
        self.penalty = penalty
        self.prompt_len = prompt_len
        if eos_token_ids is None:
            eos_token_ids = []
        elif isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        self.eos_token_ids = set(eos_token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        generated = input_ids[0, self.prompt_len:]
        if generated.numel() > 0:
            seen = torch.unique(generated)
            for tok in seen.tolist():
                if tok in self.eos_token_ids:
                    continue
                scores[0, tok] = scores[0, tok] - self.penalty
        return scores


# ---------------------------------------------------------------------------
# MMMU 공식 파싱/채점 로직
# (출처: https://github.com/MMMU-Benchmark/MMMU/blob/main/eval/eval_utils.py, 그대로 이식)
# ---------------------------------------------------------------------------
random.seed(SEED)


def parse_multi_choice_response(response, all_choices, index2ans):
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True
    if len(candidates) == 0:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False
    if len(candidates) == 0:
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    start_indexes.append(response.rfind(f"({can})"))
            else:
                for can in candidates:
                    start_indexes.append(response.rfind(f" {can} "))
        else:
            for can in candidates:
                start_indexes.append(response.lower().rfind(index2ans[can].lower()))
        pred_index = candidates[start_indexes.index(max(start_indexes))]
    else:
        pred_index = candidates[0]
    return pred_index


def check_is_number(s):
    try:
        float(s.replace(",", ""))
        return True
    except ValueError:
        return False


def normalize_str(s):
    s = s.strip()
    if check_is_number(s):
        return [round(float(s.replace(",", "")), 2)]
    s = s.lower()
    if len(s) == 1:
        return [" " + s, s + " "]
    return [s]


def extract_numbers(s):
    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"
    return (
        re.findall(pattern_commas, s)
        + re.findall(pattern_scientific, s)
        + re.findall(pattern_simple, s)
    )


def parse_open_response(response):
    def get_key_subresponses(resp):
        resp = resp.strip().strip(".").lower()
        sub_responses = re.split(r"\.\s(?=[A-Z])|\n", resp)
        indicators = ["could be ", "so ", "is ", "thus ", "therefore ", "final ", "answer ", "result "]
        key_responses = []
        for i, sr in enumerate(sub_responses):
            local_indicators = indicators + (["="] if i == len(sub_responses) - 1 else [])
            shortest = None
            for ind in local_indicators:
                if ind in sr:
                    cand = sr.split(ind)[-1].strip()
                    if shortest is None or len(cand) < len(shortest):
                        shortest = cand
            if shortest and shortest not in [":", ",", ".", "!", "?", ";", ":", "'"]:
                key_responses.append(shortest)
        return key_responses if key_responses else [resp]

    key_responses = get_key_subresponses(response)
    pred_list = key_responses.copy()
    for r in key_responses:
        pred_list.extend(extract_numbers(r))
    tmp = []
    for p in pred_list:
        tmp.extend(normalize_str(p))
    return list(set(tmp))


def eval_multi_choice(gold_i, pred_i):
    if isinstance(gold_i, list):
        return pred_i in gold_i
    return gold_i == pred_i


def eval_open(gold_i, pred_i):
    norm_answers = []
    if isinstance(gold_i, list):
        for a in gold_i:
            norm_answers.extend(normalize_str(a))
    else:
        norm_answers = normalize_str(gold_i)
    for p in pred_i:
        if isinstance(p, str):
            for na in norm_answers:
                if isinstance(na, str) and na in p:
                    return True
        else:
            if p in norm_answers:
                return True
    return False


# ---------------------------------------------------------------------------
# 프롬프트 구성
# (출처: MMMU 공식 repo eval/data_utils.py의 construct_prompt 로직을 그대로 따름)
# ---------------------------------------------------------------------------
def build_prompt(sample):
    question = sample["question"]
    options = eval(sample["options"]) if sample.get("options") else []
    is_mc = len(options) > 0

    if is_mc:
        letters = list(string.ascii_uppercase[: len(options)])
        lines = "\n".join(f"({L}) {opt}" for L, opt in zip(letters, options))
        prompt = (f"{question}\n{lines}\nAnswer with the option's letter from the given "
                  f"choices directly, without additional explanation.")
        index2ans = dict(zip(letters, options))
        return prompt, is_mc, letters, index2ans
    else:
        prompt = f"{question}\nAnswer the question using a single word or phrase, without additional explanation."
        return prompt, is_mc, None, None


def gather_images(sample):
    imgs = []
    for i in range(1, 8):
        img = sample.get(f"image_{i}")
        if img is not None:
            imgs.append(img.convert("RGB"))
    return imgs


def _save_result(args, hw_spec, subject_correct, subject_total, subject_elapsed,
                  failure_cases, total_elapsed, partial=False):
    subject_accuracy = {
        s: f"{(subject_correct[s] / subject_total[s] * 100 if subject_total[s] else 0):.2f}% "
           f"({subject_correct[s]}/{subject_total[s]})"
        for s in SUBJECTS
    }
    total_correct = sum(subject_correct.values())
    total_samples = sum(subject_total.values())
    done_subjects = [s for s in SUBJECTS if subject_total[s]]
    overall_accuracy = (
        sum(subject_correct[s] / subject_total[s] for s in done_subjects) / len(done_subjects)
        if done_subjects else 0
    )
    result = {
        "partial": partial,  # True면 아직 900문제 다 안 끝난 중간 저장본
        "hw_spec": hw_spec,
        "model_revision": args.model_revision,
        "data_revision": args.data_revision,
        "sampling_recipe": {**SAMPLING_RECIPE, "presence_penalty": PRESENCE_PENALTY,
                             "max_new_tokens": args.max_new_tokens, "seed": SEED},
        "image_resolution": {"min_pixels": args.min_pixels, "max_pixels": args.max_pixels},
        "total_samples": total_samples,
        "total_correct": total_correct,
        "subject_accuracy": subject_accuracy,
        "subject_elapsed_time": {s: f"{subject_elapsed[s]:.1f}s" for s in SUBJECTS},
        "failure_cases": failure_cases,
        "overall_accuracy": f"{overall_accuracy * 100:.2f}%",
        "total_elapsed_time": f"{total_elapsed:.1f}s",
        # 아래 3개는 재시작(resume) 시 "어디까지 했는지" 정확히 복원하기 위한 원본 숫자 데이터.
        # (위의 subject_accuracy/subject_elapsed_time은 사람이 읽기 좋은 문자열 포맷이라
        #  프로그램이 다시 파싱하기 번거로움)
        "_resume_subject_correct": subject_correct,
        "_resume_subject_total": subject_total,
        "_resume_subject_elapsed": subject_elapsed,
    }
    import os
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--model_revision", default=MODEL_REVISION)
    ap.add_argument("--data_root", default="MMMU/MMMU")
    ap.add_argument("--data_revision", default=DATA_REVISION)
    ap.add_argument("--output_path", default="outputs/mmmu_results.json")
    ap.add_argument("--max_new_tokens", type=int, default=512,
                     help="생성 길이 상한. 실행 시간과 응답 잘림 사이의 절충값. "
                          "프롬프트에는 'without additional explanation' 지시문이 포함되어 있다.")
    ap.add_argument("--limit_per_subject", type=int, default=0, help="0이면 과목당 30문제 전체")
    ap.add_argument("--min_pixels", type=int, default=256 * 28 * 28,
                     help="이미지 최소 픽셀 수 (processor의 min_pixels).")
    ap.add_argument("--max_pixels", type=int, default=1280 * 28 * 28,
                     help="이미지 최대 픽셀 수 (processor의 max_pixels). 디테일 보존과 VRAM/속도의 절충값.")
    ap.add_argument("--no_resume", action="store_true",
                     help="output_path에 이전 중간저장 결과가 있어도 무시하고 처음부터 다시 실행")
    # parse_known_args: 코랩/주피터 커널이 자체적으로 넘기는 `-f <kernel>.json` 같은
    # 인자가 섞여 들어와도 무시하고 알려진 인자만 파싱 (노트북에서 직접 실행해도 안전)
    args, _unknown = ap.parse_known_args()

    import os
    from datasets import load_dataset

    # ------------------------------------------------------------------
    # 재시작(resume) 지점 확인: output_path에 이전 실행의 중간/완료 결과가 있으면
    # 거기서부터 이어서 실행한다 (코랩 끊김 등으로 중단됐다가 다시 돌릴 때
    # 처음부터 다시 안 해도 되게).
    # ------------------------------------------------------------------
    resume_correct = {s: 0 for s in SUBJECTS}
    resume_total = {s: 0 for s in SUBJECTS}
    resume_elapsed = {s: 0.0 for s in SUBJECTS}
    resume_failures = []
    resume_prev_elapsed = 0.0
    if not args.no_resume and os.path.exists(args.output_path):
        try:
            with open(args.output_path) as f:
                prev = json.load(f)
            if "_resume_subject_total" in prev and sum(prev["_resume_subject_total"].values()) < len(SUBJECTS) * 30:
                resume_correct = {s: prev["_resume_subject_correct"].get(s, 0) for s in SUBJECTS}
                resume_total = {s: prev["_resume_subject_total"].get(s, 0) for s in SUBJECTS}
                resume_elapsed = {s: prev["_resume_subject_elapsed"].get(s, 0.0) for s in SUBJECTS}
                resume_failures = prev.get("failure_cases", [])
                resume_prev_elapsed = sum(resume_elapsed.values())
                n_prev = sum(resume_total.values())
                print(f"[재시작] 이전 결과 발견: {n_prev}문제까지 완료된 상태에서 이어서 실행합니다.")
        except (json.JSONDecodeError, KeyError):
            print("[재시작] 이전 결과 파일을 읽지 못해 처음부터 실행합니다.")

    print("[1/3] 데이터 로딩...")
    # 과목명을 sample['id']에서 역으로 추출하는 건 깨지기 쉬워서(예: subject 이름에 '_'가 여러 개 들어감),
    # (subject_name, dataset) 쌍으로 유지한 채 순회한다.
    subject_datasets = []
    for subj in SUBJECTS:
        ds = load_dataset(args.data_root, subj, split="validation", revision=args.data_revision)
        already_done = resume_total[subj]  # 재시작 시 이 과목에서 이미 처리한 문제 수만큼 건너뜀
        if already_done:
            ds = ds.select(range(already_done, len(ds)))
        if args.limit_per_subject:
            ds = ds.select(range(min(args.limit_per_subject, len(ds))))
        subject_datasets.append((subj, ds))
    total_n_remaining = sum(len(ds) for _, ds in subject_datasets)
    total_n = total_n_remaining + sum(resume_total.values())
    print(f"  총 {total_n}문제 중 {total_n_remaining}문제 남음 ({len(subject_datasets)}개 과목)")

    print("[2/3] 모델 로딩...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, revision=args.model_revision, dtype="auto", device_map="auto"
    )
    # 이미지 해상도(min_pixels/max_pixels) 명시적 지정.
    # 근거: MMMU에는 표/그래프/수식이 포함된 이미지가 많아 과도하게 줄이면 숫자·기호
    # 정보가 손실될 수 있고, 상한을 두지 않으면 VRAM/속도 부담이 커진다. 이 두 가지를
    # 절충해 min_pixels=256*28*28, max_pixels=1280*28*28로 설정했다.
    # (이 값에 대한 별도 ablation은 수행하지 않았다.)
    processor = AutoProcessor.from_pretrained(
        args.model_path, revision=args.model_revision,
        min_pixels=args.min_pixels, max_pixels=args.max_pixels,
    )
    torch.manual_seed(SEED)
    hw_spec = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print("[3/3] 추론 + 채점 시작...")
    subject_correct = dict(resume_correct)
    subject_total = dict(resume_total)
    subject_elapsed = dict(resume_elapsed)
    failure_cases = list(resume_failures)
    total_start = time.time() - resume_prev_elapsed  # 이전 실행 경과시간까지 반영해서 ETA 계산
    n_done = sum(resume_total.values())

    for subj, ds in subject_datasets:
        for sample in ds:
            t0 = time.time()

            prompt, is_mc, letters, index2ans = build_prompt(sample)
            images = gather_images(sample)

            content = [{"type": "image", "image": img} for img in images]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]

            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device)
            prompt_len = inputs["input_ids"].shape[-1]

            presence_processor = PresencePenaltyLogitsProcessor(
                PRESENCE_PENALTY, prompt_len, model.generation_config.eos_token_id
            )
            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    logits_processor=[presence_processor],
                    **SAMPLING_RECIPE,
                )
            gen_trimmed = gen_ids[0][prompt_len:]
            response = processor.decode(gen_trimmed, skip_special_tokens=True).strip()

            gold = sample["answer"]
            if is_mc:
                pred = parse_multi_choice_response(response, letters, index2ans)
                correct = eval_multi_choice(gold, pred)
            else:
                pred = parse_open_response(response)
                correct = eval_open(gold, pred)

            subject_total[subj] += 1
            subject_elapsed[subj] += time.time() - t0
            if correct:
                subject_correct[subj] += 1
            else:
                failure_cases.append({
                    "subject": subj, "id": sample["id"], "question": sample["question"],
                    "options": sample.get("options", "[]"), "ground_truth": gold,
                    "model_response": response, "parsed_pred": pred,
                })

            n_done += 1
            elapsed = time.time() - total_start
            avg = elapsed / n_done
            remaining = avg * (total_n - n_done)
            acc_so_far = sum(subject_correct.values()) / n_done * 100
            print(f"  [{n_done}/{total_n}] ({subj}) 방금 문제 {time.time()-t0:.1f}초 | "
                  f"누적 accuracy={acc_so_far:.1f}% | 경과 {elapsed/60:.1f}분 | "
                  f"예상 남은 시간 {remaining/60:.1f}분")

            # 100문제마다 중간 저장 (코랩 끊김 대비 — 끊겨도 여기까지 결과는 살아있음)
            if n_done % 100 == 0:
                _save_result(args, hw_spec, subject_correct, subject_total, subject_elapsed,
                             failure_cases, time.time() - total_start, partial=True)

    total_elapsed = time.time() - total_start
    result = _save_result(args, hw_spec, subject_correct, subject_total, subject_elapsed,
                           failure_cases, total_elapsed, partial=False)

    print(f"\nOverall (macro avg) accuracy: {result['overall_accuracy']}  "
          f"({result['total_correct']}/{result['total_samples']})")
    print(f"결과 저장: {args.output_path}")


if __name__ == "__main__":
    main()
