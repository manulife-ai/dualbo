import csv
from pathlib import Path
import random

from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent / "Data"
ETHOS_PATH = DATA_DIR / "Ethos_Dataset_Binary.csv"
ARC_LABEL_NAMES = ["A", "B", "C", "D"]
CHOICE_LABEL_NAMES = [chr(ord("A") + idx) for idx in range(26)]

# ---------------------------
# Data loading
# ---------------------------

def format_hotpot_context(context):
    titles = context.get("title", [])
    sentences = context.get("sentences", [])
    sections = []
    for title, sentence_list in zip(titles, sentences):
        paragraph = " ".join(sentence_list).strip()
        if paragraph:
            sections.append(f"{title}: {paragraph}")
    return "\n\n".join(sections)

def format_hotpot_example(example):
    context_text = format_hotpot_context(example["context"])
    return f"Question: {example['question']}\n\nContext:\n{context_text}"

def load_hotpot_qa(seed=42):
    ds = load_dataset("hotpot_qa", "distractor")
    label_names = []

    train_data = [
        {"text": format_hotpot_example(x), "label": x["answer"]}
        for x in ds["train"]
    ]
    val_data = [
        {"text": format_hotpot_example(x), "label": x["answer"]}
        for x in ds["validation"]
    ]
    random.Random(seed).shuffle(train_data)
    random.Random(seed + 1).shuffle(val_data)

    task_description = "Answer the question using only the provided context passages. Return a short final answer, including yes or no when appropriate."
    seed_prompts = [
        "You are a question-answering system. Read the question and context, then return only the shortest correct answer supported by the context.",
        "Answer the question using the provided context only. Return just the final answer phrase, with no explanation.",
        "Use the context to answer the question accurately. Output only the final short answer and nothing else."
    ]
    task_config = {
        "task_type": "qa",
        "answer_format_description": "Return only the final short answer as JSON under the key 'answer'.",
    }
    return train_data, val_data, label_names, task_description, seed_prompts, task_config


def _normalize_arc_choice_label(label):
    normalized_label = str(label).strip().upper()
    numeric_to_alpha = {
        "1": "A",
        "2": "B",
        "3": "C",
        "4": "D",
    }
    return numeric_to_alpha.get(normalized_label, normalized_label)


def format_arc_example(example):
    question = (example.get("question") or "").strip()
    choices = example.get("choices") or {}
    choice_labels = choices.get("label") or []
    choice_texts = choices.get("text") or []

    formatted_choices = []
    for raw_label, choice_text in zip(choice_labels, choice_texts):
        normalized_label = _normalize_arc_choice_label(raw_label)
        choice_text = str(choice_text).strip()
        if normalized_label not in ARC_LABEL_NAMES or not choice_text:
            continue
        formatted_choices.append(f"{normalized_label}. {choice_text}")

    if not question or not formatted_choices:
        return ""

    return f"Question: {question}\n\nAnswer choices:\n" + "\n".join(formatted_choices)


def _build_arc_split(split):
    examples = []
    for record in split:
        prompt_text = format_arc_example(record)
        answer_key = _normalize_arc_choice_label(record.get("answerKey", ""))
        if not prompt_text or answer_key not in ARC_LABEL_NAMES:
            continue
        examples.append({
            "text": prompt_text,
            "label": ARC_LABEL_NAMES.index(answer_key),
        })
    return examples


def load_arc(seed=42):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    label_names = ARC_LABEL_NAMES

    train_data = _build_arc_split(ds["train"])
    val_data = _build_arc_split(ds["validation"])
    random.Random(seed).shuffle(train_data)
    random.Random(seed + 1).shuffle(val_data)

    task_description = "Answer the science multiple-choice question by selecting the single correct option label from the provided choices."
    seed_prompts = [
        "You are a science question-answering system. Read the question and answer choices, then return exactly one valid option label.",
        "Choose the single best answer to the science question. Return only the exact option label from the listed choices.",
        "Determine which answer choice is correct for the science question. Output one valid label only."
    ]
    task_config = {
        "task_type": "classification",
        "answer_format_description": "Return exactly one valid option label: A, B, C, or D.",
    }
    return train_data, val_data, label_names, task_description, seed_prompts, task_config


def _extract_mmlu_pro_option_texts(record):
    raw_options = record.get("options") or record.get("choices") or []

    option_texts = []
    if isinstance(raw_options, dict):
        labels = raw_options.get("label") or []
        texts = raw_options.get("text") or []
        for label, text in zip(labels, texts):
            text = str(text).strip()
            if text:
                option_texts.append(text)
    else:
        for option in raw_options:
            if isinstance(option, dict):
                text = str(option.get("text") or option.get("content") or option.get("option") or "").strip()
            else:
                text = str(option).strip()
            if text:
                option_texts.append(text)

    return option_texts


def _normalize_mmlu_pro_answer_label(answer_value, option_texts):
    if answer_value is None:
        return None

    if isinstance(answer_value, int):
        return CHOICE_LABEL_NAMES[answer_value] if 0 <= answer_value < len(option_texts) else None

    answer_text = str(answer_value).strip()
    if not answer_text:
        return None

    if answer_text.isdigit():
        index = int(answer_text)
        return CHOICE_LABEL_NAMES[index] if 0 <= index < len(option_texts) else None

    upper_text = answer_text.upper()
    if upper_text in CHOICE_LABEL_NAMES[: len(option_texts)]:
        return upper_text

    for index, option_text in enumerate(option_texts):
        if answer_text == option_text:
            return CHOICE_LABEL_NAMES[index]

    return None


def _format_mmlu_pro_example(question_text, option_texts):
    formatted_choices = [
        f"{CHOICE_LABEL_NAMES[idx]}. {option_text}"
        for idx, option_text in enumerate(option_texts)
    ]
    return f"Question: {question_text}\n\nAnswer choices:\n" + "\n".join(formatted_choices)


def _build_mmlu_pro_split(split):
    examples = []
    max_option_count = 0

    for record in split:
        question_text = str(record.get("question") or record.get("question_text") or "").strip()
        category = str(record.get("category") or record.get("subject") or "").strip()
        option_texts = _extract_mmlu_pro_option_texts(record)
        answer_label = _normalize_mmlu_pro_answer_label(
            record.get("answer_index", record.get("answer", record.get("answerKey"))),
            option_texts,
        )

        if not question_text or len(option_texts) < 2 or answer_label is None:
            continue

        max_option_count = max(max_option_count, len(option_texts))
        examples.append(
            {
                "text": _format_mmlu_pro_example(question_text, option_texts),
                "label": CHOICE_LABEL_NAMES.index(answer_label),
                **({"category": category} if category else {}),
            }
        )

    return examples, max_option_count


def load_mmlu_pro(seed=42):
    ds = load_dataset("TIGER-Lab/MMLU-Pro")

    all_examples, max_option_count = _build_mmlu_pro_split(ds["test"])
    shuffled_examples = list(all_examples)
    random.Random(seed).shuffle(shuffled_examples)

    val_size = min(1000, len(shuffled_examples))
    split_index = max(len(shuffled_examples) - val_size, 0)
    train_data = shuffled_examples[:split_index]
    val_data = shuffled_examples[split_index:]
    label_names = CHOICE_LABEL_NAMES[:max_option_count]

    task_description = "Answer the multiple-choice question by selecting the single correct option label from the provided list of answer choices."
    seed_prompts = [
        "Read the question and answer choices, then return exactly one valid option label for the best answer.",
        "Choose the single best answer to the question. Return only the exact option label from the listed choices.",
        "Determine which answer choice is correct and output one valid label only."
    ]
    task_config = {
        "task_type": "classification",
        "answer_format_description": "Return exactly one valid option label that matches the indexed answer choices shown in the question.",
        "reflection_example_count": 8,
        "shuffle_subsets_after_sampling": True,
    }
    return train_data, val_data, label_names, task_description, seed_prompts, task_config

def _stratified_split(examples, validation_fraction=0.2, seed=42):
    grouped_examples = {}
    for example in examples:
        grouped_examples.setdefault(example["label"], []).append(example)

    rng = random.Random(seed)
    train_data = []
    val_data = []
    for group in grouped_examples.values():
        shuffled_group = list(group)
        rng.shuffle(shuffled_group)

        if len(shuffled_group) <= 1:
            train_data.extend(shuffled_group)
            continue

        val_size = max(1, int(round(len(shuffled_group) * validation_fraction)))
        val_size = min(val_size, len(shuffled_group) - 1)

        val_data.extend(shuffled_group[:val_size])
        train_data.extend(shuffled_group[val_size:])

    rng.shuffle(train_data)
    rng.shuffle(val_data)
    return train_data, val_data


def load_ethos(seed=42):
    raw_examples = []
    with ETHOS_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            comment = (row.get("comment") or "").strip()
            raw_label = (row.get("isHate") or "").strip()
            if not comment or raw_label == "":
                continue

            label_value = int(round(float(raw_label)))
            raw_examples.append({
                "text": comment,
                "label": label_value,
            })

    train_data, val_data = _stratified_split(raw_examples, validation_fraction=0.2, seed=seed)
    label_names = ["non_hate", "hate"]

    task_description = "Classify whether the comment contains hate speech."
    seed_prompts = [
        "You are a hate-speech classifier. Read the comment and return exactly one valid label.",
        "Classify the comment as hate or non_hate. Return only the exact label.",
        "Determine whether the comment expresses hate speech. Output one valid label only."
    ]
    task_config = {
        "task_type": "classification",
        "answer_format_description": "Return exactly one valid label: non_hate or hate.",
    }
    return train_data, val_data, label_names, task_description, seed_prompts, task_config



def get_dataset(name: str, seed: int):
    if name == "arc":
        return load_arc(seed=seed)
    elif name == "mmlu_pro":
        return load_mmlu_pro(seed=seed)
    elif name == "hotpot_qa":
        return load_hotpot_qa(seed=seed)
    elif name == "ethos":
        return load_ethos(seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    

def test_data_loading(name: str, seed: int):
    train_data, val_data, label_names, task_description, seed_prompts, task_config = get_dataset(name, seed=seed)
    print(f"Dataset: {name}")
    print(f"Number of training examples: {len(train_data)}")
    print(f"Number of validation examples: {len(val_data)}")
    print(f"Label names: {label_names}")
    print(f"Task description: {task_description}")
    print(f"Task config: {task_config}")
    # print(f"Seed prompts: {seed_prompts}")
    print(f"First 3 training examples: {train_data[:3]}")
    print(f"First 3 validation examples: {val_data[:3]}")
    print("-" * 50)
    
if __name__ == "__main__":
    test_data_loading("arc", seed=42)
    test_data_loading("mmlu_pro", seed=42)
    test_data_loading("hotpot_qa", seed=42)
    test_data_loading("ethos", seed=42)