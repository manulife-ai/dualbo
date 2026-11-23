import re
import json
import math
import os
import random
import hashlib
import string
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from openai import OpenAI
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _build_openai_client() -> OpenAI:
    _load_env_file()

    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    missing = [name for name, value in {
        "OPENAI_BASE_URL": base_url,
        "OPENAI_API_KEY": api_key,
    }.items() if not value]
    if missing:
        raise RuntimeError(
            "Missing OpenAI configuration in Code_Submission/.env: " + ", ".join(missing)
        )

    return OpenAI(base_url=base_url, api_key=api_key)


client = _build_openai_client()

TRACE_STATE: Dict[str, Any] = {
    "active": False,
    "run_id": None,
    "script_name": None,
    "started_at": None,
    "config": {},
    "include_llm_calls": False,
    "captured_call_types": set(),
    "llm_calls": [],
    "llm_call_counts": {},
}


def _empty_llm_call_counts() -> Dict[str, int]:
    return {
        "total_model_calls": 0,
        "embedding_model_calls": 0,
        "task_model_calls": 0,
        "prompt_generation_model_calls": 0,
    }


def _llm_call_bucket(call_type: str) -> str:
    if call_type == "embedding_generation":
        return "embedding_model_calls"
    if call_type == "chat_completion":
        return "task_model_calls"
    return "prompt_generation_model_calls"

# ---------------------------
# Basic utilities
# ---------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def stable_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def current_hour_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H")

def create_run_id() -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"

def sanitize_filename_part(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    return text.strip("-._") or "value"

def start_run_trace(
    script_name: str,
    config: Dict[str, Any],
    include_llm_calls: bool = False,
    captured_call_types: Optional[List[str]] = None,
) -> str:
    run_id = create_run_id()
    TRACE_STATE.update(
        {
            "active": True,
            "run_id": run_id,
            "script_name": script_name,
            "started_at": utc_now_iso(),
            "config": dict(config),
            "include_llm_calls": include_llm_calls,
            "captured_call_types": set(captured_call_types or []),
            "llm_calls": [],
            "llm_call_counts": _empty_llm_call_counts(),
        }
    )
    return run_id

def record_llm_call(
    call_type: str,
    model: str,
    prompt: Optional[str],
    input_text: Any,
    output_text: Any,
) -> None:
    if not TRACE_STATE["active"]:
        return

    counts = TRACE_STATE["llm_call_counts"]
    counts["total_model_calls"] += 1
    counts[_llm_call_bucket(call_type)] += 1

    should_capture_call = TRACE_STATE["include_llm_calls"] or call_type in TRACE_STATE["captured_call_types"]
    if not should_capture_call:
        return

    TRACE_STATE["llm_calls"].append(
        {
            "index": len(TRACE_STATE["llm_calls"]) + 1,
            "timestamp": utc_now_iso(),
            "type": call_type,
            "model": model,
            "prompt": prompt,
            "input": input_text,
            "output": output_text,
        }
    )

def build_run_log_payload(
    total_runtime_seconds: float,
    optimizer_trace: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    payload = {
        "run_id": TRACE_STATE["run_id"],
        "script_name": TRACE_STATE["script_name"],
        "started_at": TRACE_STATE["started_at"],
        "finished_at": utc_now_iso(),
        "total_runtime_seconds": total_runtime_seconds,
        "config": dict(TRACE_STATE["config"]),
        "summary": dict(summary),
        "model_call_summary": dict(TRACE_STATE["llm_call_counts"]),
        "optimizer_trace": optimizer_trace,
    }
    if TRACE_STATE["llm_calls"]:
        payload["llm_calls"] = list(TRACE_STATE["llm_calls"])
    TRACE_STATE.update(
        {
            "active": False,
            "run_id": None,
            "script_name": None,
            "started_at": None,
            "config": {},
            "include_llm_calls": False,
            "captured_call_types": set(),
            "llm_calls": [],
            "llm_call_counts": {},
        }
    )
    return payload

def write_run_log(
    output_dir: str,
    filename_prefix: str,
    filename_parts: List[Any],
    payload: Dict[str, Any],
) -> str:
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    sanitized_parts = [sanitize_filename_part(part) for part in filename_parts if str(part).strip()]
    file_name = f"{filename_prefix}_{'_'.join(sanitized_parts)}.json"
    file_path = log_dir / file_name
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return str(file_path)

def example_id(example: Dict[str, Any]) -> str:
    return stable_hash(json.dumps(example, sort_keys=True, ensure_ascii=False))

def sample_subset(data: List[Dict], k: int, seed: int = 42) -> List[Dict]:
    rng = random.Random(seed)
    if k >= len(data):
        return data[:]
    return rng.sample(data, k)

def _sample_balanced_indices_by_key(
    data: List[Dict[str, Any]],
    indices: List[int],
    size: int,
    key: str,
    rng: random.Random,
) -> List[int]:
    if size <= 0 or not indices:
        return []

    grouped_indices: Dict[Any, List[int]] = {}
    for idx in indices:
        grouped_indices.setdefault(data[idx][key], []).append(idx)

    for group in grouped_indices.values():
        rng.shuffle(group)

    group_order = list(grouped_indices.keys())
    rng.shuffle(group_order)
    selected: List[int] = []
    selected_counts = {group_key: 0 for group_key in group_order}

    for _ in range(min(size, len(indices))):
        available_keys = [group_key for group_key in group_order if grouped_indices[group_key]]
        if not available_keys:
            break

        min_selected = min(selected_counts[group_key] for group_key in available_keys)
        candidate_keys = [
            group_key for group_key in available_keys if selected_counts[group_key] == min_selected
        ]
        max_remaining = max(len(grouped_indices[group_key]) for group_key in candidate_keys)
        candidate_keys = [
            group_key for group_key in candidate_keys if len(grouped_indices[group_key]) == max_remaining
        ]
        chosen_key = rng.choice(candidate_keys)

        selected.append(grouped_indices[chosen_key].pop())
        selected_counts[chosen_key] += 1

    return selected

def sample_disjoint_subsets(
    data: List[Dict],
    sizes: List[int],
    seed: int = 42,
    shuffle_subsets_after_sampling: bool = True,
) -> List[List[Dict]]:
    rng = random.Random(seed)
    indices = list(range(len(data)))

    stratify_key = None
    if data and all(isinstance(example, dict) and example.get("category") for example in data):
        stratify_key = "category"

    if stratify_key is None:
        rng.shuffle(indices)

    subsets = []
    for size in sizes:
        if stratify_key is None:
            take_size = min(max(size, 0), len(indices))
            subset_indices = indices[:take_size]
            del indices[:take_size]
        else:
            subset_indices = _sample_balanced_indices_by_key(
                data,
                indices,
                max(size, 0),
                stratify_key,
                rng,
            )
            selected_set = set(subset_indices)
            indices = [idx for idx in indices if idx not in selected_set]
            if shuffle_subsets_after_sampling:
                rng.shuffle(subset_indices)
        subsets.append([data[idx] for idx in subset_indices])
    return subsets

def accuracy_from_preds(preds: List[int], labels: List[int]) -> float:
    return float(np.mean([int(p == y) for p, y in zip(preds, labels)]))

def normalize_text_answer(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text

def is_prediction_correct(pred: Any, label: Any, task_type: str) -> int:
    if task_type == "classification":
        return int(pred == label)
    if task_type == "qa":
        return int(normalize_text_answer(str(pred)) == normalize_text_answer(str(label)))
    raise ValueError(f"Unknown task type: {task_type}")

def qa_f1_score(pred: Any, label: Any) -> float:
    pred_tokens = normalize_text_answer(str(pred)).split()
    label_tokens = normalize_text_answer(str(label)).split()

    if not pred_tokens and not label_tokens:
        return 1.0
    if not pred_tokens or not label_tokens:
        return 0.0

    overlap = Counter(pred_tokens) & Counter(label_tokens)
    num_same = sum(overlap.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(label_tokens)
    return 2 * precision * recall / (precision + recall)

def compute_prediction_metrics(preds: List[Any], examples: List[Dict], task_type: str) -> Dict[str, float]:
    correctness = [is_prediction_correct(pred, ex["label"], task_type) for pred, ex in zip(preds, examples)]
    metrics = {
        "accuracy": float(np.mean(correctness)) if correctness else 0.0,
    }
    if task_type == "qa":
        f1_scores = [qa_f1_score(pred, ex["label"]) for pred, ex in zip(preds, examples)]
        metrics["f1"] = float(np.mean(f1_scores)) if f1_scores else 0.0
    return metrics

def score_predictions(preds: List[Any], examples: List[Dict], task_type: str) -> float:
    return compute_prediction_metrics(preds, examples, task_type)["accuracy"]

def prediction_correctness_vector(preds: List[Any], examples: List[Dict], task_type: str) -> np.ndarray:
    return np.array(
        [is_prediction_correct(pred, ex["label"], task_type) for pred, ex in zip(preds, examples)],
        dtype=np.float64,
    )

def truncate_text(text: str, max_chars: int = 600) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."

def format_prediction_value(value: Any, label_names: List[str], task_type: str) -> str:
    if task_type == "classification":
        if isinstance(value, (int, np.integer)) and 0 <= int(value) < len(label_names):
            return label_names[int(value)]
        return str(value)
    if task_type == "qa":
        return str(value)
    raise ValueError(f"Unknown task type: {task_type}")

def collect_incorrect_examples(
    preds: List[Any],
    examples: List[Dict],
    task_type: str,
    label_names: List[str],
    max_examples: int = 3,
) -> List[Dict[str, str]]:
    mistakes = []
    for pred, ex in zip(preds, examples):
        if is_prediction_correct(pred, ex["label"], task_type):
            continue
        mistakes.append(
            {
                "text": truncate_text(str(ex["text"])),
                "prediction": format_prediction_value(pred, label_names, task_type),
                "label": format_prediction_value(ex["label"], label_names, task_type),
            }
        )
        if len(mistakes) >= max_examples:
            break
    return mistakes

def format_reflection_examples(reflection_examples: List[Dict[str, str]]) -> str:
    sections = []
    for idx, example in enumerate(reflection_examples, start=1):
        sections.append(
            "\n".join(
                [
                    f"Mistake {idx}:",
                    f"Input:\n{example['text']}",
                    f"Model prediction: {example['prediction']}",
                    f"Correct answer: {example['label']}",
                ]
            )
        )
    return "\n\n".join(sections)

class SkipExampleError(Exception):
    pass

def _flatten_exception_messages(exc: BaseException) -> str:
    messages = []
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).strip()
        if message:
            messages.append(message)
        current = current.__cause__ or current.__context__
    return "\n".join(messages)

def is_skippable_prediction_error(exc: BaseException) -> bool:
    candidate = exc
    if isinstance(exc, RetryError):
        try:
            candidate = exc.last_attempt.exception() or exc
        except Exception:
            candidate = exc

    message = _flatten_exception_messages(candidate).lower()
    return (
        "contentpolicyviolationerror" in message
        or "content management policy" in message
        or "content was filtered" in message
        or "triggering azure openai's content management policy" in message
    )

# ---------------------------
# OpenAI classification call
# ---------------------------

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=20))
def get_semantic_embedding(text: str, model_name="text-embedding-3-small", dimension=16) -> np.ndarray:
    response = client.embeddings.create(
        model=model_name,
        input=text,
        dimensions=dimension,
    )
    embedding = np.array(response.data[0].embedding, dtype=np.float64)
    record_llm_call(
        call_type="embedding_generation",
        model=model_name,
        prompt=None,
        input_text=text,
        output_text={"dimensions": int(len(embedding))},
    )
    return embedding

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=20))
def call_gpt_for_prediction(
    prompt_template: str,
    text: str,
    task_type: str,
    label_names: List[str],
    model: str = "gpt-5.4-nano"
) -> str:
    if task_type == "classification":
        label_desc = ", ".join(label_names)
        user_input = f"""Task text:
{text}

Valid labels:
{label_desc}

Return ONLY a JSON object like:
{{"label": "<one label from the valid labels>"}}"""
    elif task_type == "qa":
        user_input = f"""Question answering task:
{text}

Return ONLY a JSON object like:
{{"answer": "<short answer>"}}"""
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": user_input},
        ],
    )
    out = response.choices[0].message.content.strip()
    record_llm_call(
        call_type="chat_completion",
        model=model,
        prompt=prompt_template,
        input_text=user_input,
        output_text=out,
    )
    return out

def parse_label(output_text: str, label_names: List[str]) -> Optional[str]:
    # Try JSON first
    try:
        obj = json.loads(output_text)
        label = obj["label"].strip()
        if label in label_names:
            return label
    except Exception:
        pass

    # Fallback: exact label match
    for lab in label_names:
        if re.search(rf"\b{re.escape(lab)}\b", output_text, flags=re.IGNORECASE):
            return lab

    print(f"Warning: Failed to parse label from output: using dummy invalid class. Actual response: {output_text}")
    return None

def parse_qa_answer(output_text: str) -> str:
    try:
        obj = json.loads(output_text)
        answer = str(obj["answer"]).strip()
        if answer:
            return answer
    except Exception:
        pass

    fallback = output_text.strip()
    if fallback.startswith("```"):
        fallback = re.sub(r"^```(?:json)?", "", fallback).strip()
        fallback = re.sub(r"```$", "", fallback).strip()
    return fallback.splitlines()[0].strip() if fallback else ""

class LLMLabeler:
    def __init__(self, label_names: List[str], model: str = "gpt-5.4-nano", task_type: str = "classification"):
        self.label_names = label_names
        self.model = model
        self.task_type = task_type
        self.cache = {}

    def predict_one(self, prompt: str, example: Dict) -> Any:
        text = example["text"]
        cache_suffix = "|".join(self.label_names) if self.label_names else self.task_type
        key = stable_hash(prompt + "||" + text + "||" + cache_suffix)
        if key in self.cache:
            return self.cache[key]

        raw = call_gpt_for_prediction(
            prompt,
            text,
            task_type=self.task_type,
            label_names=self.label_names,
            model=self.model,
        )

        if self.task_type == "classification":
            label_str = parse_label(raw, self.label_names)
            prediction = self.label_names.index(label_str) if label_str is not None else -1
        elif self.task_type == "qa":
            prediction = parse_qa_answer(raw)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

        self.cache[key] = prediction
        return prediction

    def predict_batch(self, prompt: str, examples: List[Dict]) -> List[Any]:
        return [self.predict_one(prompt, ex) for ex in examples]

# ---------------------------
# Prompt expansion via GPT
# ---------------------------

def _parse_json_string_list(text: str, key: str) -> List[str]:
    try:
        obj = json.loads(text)
        values = obj[key]
        return [value.strip() for value in values if isinstance(value, str) and value.strip()]
    except Exception:
        lines = [line.strip("-• \n\t") for line in text.split("\n") if line.strip()]
        return [line for line in lines if line]


def _dedupe_string_list(values: List[str]) -> List[str]:
    unique_values = []
    seen = set()

    for value in values:
        normalized_value = " ".join(value.split()).strip().lower()
        if not normalized_value or normalized_value in seen:
            continue
        seen.add(normalized_value)
        unique_values.append(value)

    return unique_values


def _chat_complete_text(
    model: str,
    instruction: str,
    user_content: str,
    call_type: str,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_content},
        ],
    )
    text = response.choices[0].message.content.strip()
    record_llm_call(
        call_type=call_type,
        model=model,
        prompt=instruction,
        input_text=user_content,
        output_text=text,
    )
    return text

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=20))
def expand_prompts_with_gpt(
    current_best_prompt: str,
    task_description: str,
    label_names: List[str],
    n_candidates: int = 8,
    model: str = "gpt-5.4-nano",
    task_type: str = "classification",
    answer_format_description: str = "",
    reflection_examples: List[Dict[str, str]] = None,
    mode: str = "reflection",
) -> List[str]:
    generation_target = max(n_candidates, 0)

    if task_type == "classification":
        output_constraints = "each prompt should instruct the model to output exactly one valid label"
        task_prefix = "You are a reflection model improving a classification prompt."
    elif task_type == "qa":
        output_constraints = "each prompt should instruct the model to answer the question using the provided context and return only the final answer"
        task_prefix = "You are a reflection model improving a question-answering prompt."
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    if mode not in {"baseline", "reflection", "diverse"}:
        raise ValueError(f"Unknown prompt extension mode: {mode}")

    reflection_examples = reflection_examples or []
    reflection_block = ""
    if mode in {"reflection", "diverse"} and reflection_examples:
        reflection_block = f"""

Reflection examples from the current prompt's mistakes:
{format_reflection_examples(reflection_examples)}

Use these mistakes to infer failure modes and revise the prompt so the same errors are less likely.
"""

    base_context = f"""
{task_prefix}

Task:
{task_description}

Output requirement:
{answer_format_description or output_constraints}
{reflection_block}
"""

    if mode == "baseline":
        instruction = f"""
{base_context}

Given the current prompt, generate {generation_target} improved alternative prompts.
Goals:
- keep the task definition correct
- rewrite the prompt without using additional diagnostic information
- vary wording, instruction structure, reasoning procedure, and decision rules
- make prompts concise and clear
- do NOT include examples
- {output_constraints}

Return ONLY valid JSON:
{{"prompts": ["prompt 1", "prompt 2", ...]}}
"""
        text = _chat_complete_text(
            model=model,
            instruction=instruction,
            user_content=current_best_prompt,
            call_type="prompt_expansion_baseline",
        )
        return _parse_json_string_list(text, "prompts")[:n_candidates]

    if mode == "reflection":
        instruction = f"""
{base_context}

Given the current prompt, generate {generation_target} improved alternative prompts.
Goals:
- keep the task definition correct
- use the reflection examples when provided to fix the observed failure modes
- make each candidate materially different from the others, not just a light paraphrase
- vary wording, instruction structure, reasoning procedure, and decision rules
- cover multiple prompt styles such as terse rule-based, checklist-driven, label-definition-focused, evidence-first, elimination-based, and ambiguity-aware prompts when they fit the task
- make prompts concise and clear
- do NOT include examples
- {output_constraints}

Return ONLY valid JSON:
{{"prompts": ["prompt 1", "prompt 2", ...]}}
"""
        text = _chat_complete_text(
            model=model,
            instruction=instruction,
            user_content=current_best_prompt,
            call_type="prompt_expansion_reflection",
        )
        return _parse_json_string_list(text, "prompts")[:n_candidates]

    strategy_instruction = f"""
{base_context}

Propose {generation_target} distinct rewriting strategies for improving the current prompt.
Each strategy should be short, concrete, and describe a different rewrite direction, such as tightening the decision rule, emphasizing edge cases, simplifying wording, adding ambiguity handling, or changing the reasoning order.

Return ONLY valid JSON:
{{"strategies": ["strategy 1", "strategy 2", ...]}}
"""
    strategy_text = _chat_complete_text(
        model=model,
        instruction=strategy_instruction,
        user_content=current_best_prompt,
        call_type="prompt_expansion_diverse_stage1",
    )
    strategies = _dedupe_string_list(_parse_json_string_list(strategy_text, "strategies"))[:n_candidates]

    if not strategies:
        return []

    strategy_block = "\n".join(f"{idx}. {strategy}" for idx, strategy in enumerate(strategies, start=1))
    prompt_instruction = f"""
{base_context}

Generate one improved prompt candidate for each rewriting strategy below.

Strategies:
{strategy_block}

Requirements:
- produce exactly one prompt per strategy and keep the same order as the strategy list
- each prompt should clearly reflect its assigned strategy
- do NOT include examples
- keep the task definition correct
- make each candidate materially different from the others, not just a light paraphrase
- vary wording, instruction structure, reasoning procedure, and decision rules
- ensure each candidate emphasizes a different strategy, constraint, or prioritization scheme that could lead to different model behavior
- avoid repeating the same opening phrase, sentence template, or ordering of instructions across candidates
- {output_constraints}

Return ONLY valid JSON:
{{"prompts": ["prompt 1", "prompt 2", ...]}}
"""
    prompt_text = _chat_complete_text(
        model=model,
        instruction=prompt_instruction,
        user_content=current_best_prompt,
        call_type="prompt_expansion_diverse_stage2",
    )
    return _parse_json_string_list(prompt_text, "prompts")[:n_candidates]

# ---------------------------
# GP utilities for BO-LLM baseline
# ---------------------------

def rbf_from_prediction_vectors(v1: np.ndarray, v2: np.ndarray, length_scale: float = 1.0) -> float:
    dist2 = float(np.sum((v1 - v2) ** 2))
    return math.exp(-dist2 / (2.0 * (length_scale ** 2)))

def build_kernel_matrix(pred_vectors: List[np.ndarray], length_scale: float, noise: float) -> np.ndarray:
    n = len(pred_vectors)
    K = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            K[i, j] = rbf_from_prediction_vectors(pred_vectors[i], pred_vectors[j], length_scale=length_scale)
    K += noise * np.eye(n)
    return K

def build_cross_kernel_matrix(
    left_vectors: List[np.ndarray],
    right_vectors: List[np.ndarray],
    length_scale: float,
) -> np.ndarray:
    n = len(left_vectors)
    m = len(right_vectors)
    K = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            K[i, j] = rbf_from_prediction_vectors(
                left_vectors[i],
                right_vectors[j],
                length_scale=length_scale,
            )
    return K

def combine_kernel_matrices(
    first_kernel: np.ndarray,
    second_kernel: np.ndarray,
    lambda_t: float,
) -> np.ndarray:
    # print(f"combining_kernel with t: {lambda_t:.4f}")
    return lambda_t * first_kernel + (1.0 - lambda_t) * second_kernel

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(embedding))
    if norm <= 1e-12:
        return embedding.astype(np.float64, copy=True)
    return np.array(embedding / norm, dtype=np.float64)

def _symmetrize_matrix(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)

def solve_psd_linear_system(matrix: np.ndarray, rhs: np.ndarray, jitter: float = 1e-8, max_attempts: int = 6) -> np.ndarray:
    identity = np.eye(matrix.shape[0], dtype=np.float64)
    stabilized = _symmetrize_matrix(np.array(matrix, dtype=np.float64, copy=True))
    current_jitter = max(float(jitter), 0.0)

    for _ in range(max_attempts):
        try:
            factor = np.linalg.cholesky(stabilized + current_jitter * identity)
            intermediate = np.linalg.solve(factor, rhs)
            return np.linalg.solve(factor.T, intermediate)
        except np.linalg.LinAlgError:
            current_jitter = 1e-8 if current_jitter == 0.0 else current_jitter * 10.0

    return np.linalg.solve(stabilized + current_jitter * identity, rhs)

def stabilize_covariance_matrix(covariance: np.ndarray, min_eigenvalue: float = 1e-8) -> np.ndarray:
    symmetric = _symmetrize_matrix(np.array(covariance, dtype=np.float64, copy=True))
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.clip(eigenvalues, min_eigenvalue, None)
    stabilized = eigenvectors @ np.diag(clipped) @ eigenvectors.T
    return _symmetrize_matrix(stabilized)

def gp_posterior_single_candidate(
    train_pred_vectors: List[np.ndarray],
    train_obs_acc: np.ndarray,
    train_control_acc: np.ndarray,
    cand_pred_vector: np.ndarray,
    length_scale: float = 1.0,
    noise: float = 1e-3,
) -> Tuple[float, float]:
    """
    BO-LLM style:
    mean function m(p) = control accuracy of prompt p
    kernel over prediction vectors on B_GP
    """
    K = build_kernel_matrix(train_pred_vectors, length_scale=length_scale, noise=noise)
    k_star = np.array([
        rbf_from_prediction_vectors(cand_pred_vector, pv, length_scale=length_scale)
        for pv in train_pred_vectors
    ], dtype=np.float64)

    m_train = train_control_acc
    m_star = np.mean(cand_pred_vector)

    residual = train_obs_acc - m_train
    alpha = solve_psd_linear_system(K, residual)
    solved_k_star = solve_psd_linear_system(K, k_star)

    mu = m_star + k_star @ alpha
    k_star_star = rbf_from_prediction_vectors(cand_pred_vector, cand_pred_vector, length_scale=length_scale)
    var = k_star_star - k_star @ solved_k_star
    var = max(float(var), 1e-8)
    return float(mu), float(var)

def gp_posterior_candidates_baseline(
    train_pred_vectors: List[np.ndarray],
    train_obs_acc: np.ndarray,
    train_control_acc: np.ndarray,
    cand_pred_vectors: List[np.ndarray],
    length_scale: float = 1.0,
    noise: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    BO-LLM style joint posterior over a finite candidate set.
    Mean function m(p) = control accuracy of prompt p.
    Kernel is computed over prediction vectors on B_GP.
    """
    m = len(cand_pred_vectors)

    K = build_kernel_matrix(train_pred_vectors, length_scale=length_scale, noise=noise)
    K_inv = np.linalg.inv(K)

    K_s = build_cross_kernel_matrix(train_pred_vectors, cand_pred_vectors, length_scale=length_scale)
    K_ss = build_kernel_matrix(cand_pred_vectors, length_scale=length_scale, noise=0.0)
    K_ss += 1e-8 * np.eye(m)

    m_train = train_control_acc
    m_cand = np.array([float(np.mean(v)) for v in cand_pred_vectors], dtype=np.float64)

    residual = train_obs_acc - m_train
    alpha = solve_psd_linear_system(K, residual)
    solved_cross_kernel = solve_psd_linear_system(K, K_s)

    mu = m_cand + K_s.T @ alpha
    cov = K_ss - K_s.T @ solved_cross_kernel
    cov = stabilize_covariance_matrix(cov + 1e-8 * np.eye(m))
    return mu, cov

def ucb_score(mu: float, sigma: float, kappa: float) -> float:
    return mu + kappa * sigma
