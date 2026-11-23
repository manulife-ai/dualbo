import argparse
import json
import time
import numpy as np
from tqdm import tqdm

from common_prompt_bo import (
    set_seed,
    sample_subset,
    sample_disjoint_subsets,
    score_predictions,
    compute_prediction_metrics,
    LLMLabeler,
    collect_incorrect_examples,
    example_id,
    is_skippable_prediction_error,
    start_run_trace,
    build_run_log_payload,
    write_run_log,
    _chat_complete_text,
    _parse_json_string_list,
    format_reflection_examples,
)
from BO_data_loader import get_dataset


def dedupe_string_list(values):
    unique_values = []
    seen = set()
    for value in values:
        normalized_value = " ".join(str(value).split()).strip().lower()
        if not normalized_value or normalized_value in seen:
            continue
        seen.add(normalized_value)
        unique_values.append(str(value).strip())
    return unique_values


def build_prompt_context(task_description, task_type, answer_format_description, reflection_examples):
    if task_type == "classification":
        output_constraints = "each prompt should instruct the model to output exactly one valid label"
        role_prefix = "You are a ProTeGi-style prompt optimizer improving a classification prompt with textual critiques."
    elif task_type == "qa":
        output_constraints = "each prompt should instruct the model to answer the question using the provided context and return only the final answer"
        role_prefix = "You are a ProTeGi-style prompt optimizer improving a question-answering prompt with textual critiques."
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    reflection_examples = reflection_examples or []
    reflection_block = ""
    if reflection_examples:
        reflection_block = f"""

Observed failures from the current prompt:
{format_reflection_examples(reflection_examples)}

Use these failures as evidence for what the prompt should improve.
"""

    base_context = f"""
{role_prefix}

Task:
{task_description}

Output requirement:
{answer_format_description or output_constraints}
{reflection_block}
"""
    return base_context, output_constraints


def generate_protegi_candidates(current_prompt, task_description, label_names, n_candidates, model, task_type, answer_format_description, reflection_examples):
    generation_target = max(n_candidates, 0)
    base_context, output_constraints = build_prompt_context(
        task_description=task_description,
        task_type=task_type,
        answer_format_description=answer_format_description,
        reflection_examples=reflection_examples,
    )

    critique_instruction = f"""
{base_context}

Generate {generation_target} concise textual critiques of the current prompt.
Each critique must identify one concrete weakness and one corresponding revision direction.
Keep critiques short, actionable, and materially different from one another.

Return ONLY valid JSON:
{{"critiques": ["critique 1", "critique 2", ...]}}
"""
    critique_text = _chat_complete_text(
        model=model,
        instruction=critique_instruction,
        user_content=current_prompt,
        call_type="protegi_critique_generation",
    )
    critiques = dedupe_string_list(_parse_json_string_list(critique_text, "critiques"))[:n_candidates]

    if not critiques:
        return [], []

    critique_block = "\n".join(f"{idx}. {critique}" for idx, critique in enumerate(critiques, start=1))
    edit_instruction = f"""
{base_context}

Given the current prompt and the critiques below, generate one improved prompt candidate per critique.

Critiques:
{critique_block}

Requirements:
- produce exactly one prompt per critique and keep the same order as the critique list
- each prompt should directly address its assigned critique
- keep the task definition correct
- do NOT include examples
- keep prompts concise and clear
- {output_constraints}

Return ONLY valid JSON:
{{"prompts": ["prompt 1", "prompt 2", ...]}}
"""
    edit_text = _chat_complete_text(
        model=model,
        instruction=edit_instruction,
        user_content=current_prompt,
        call_type="protegi_prompt_editing",
    )
    prompts = dedupe_string_list(_parse_json_string_list(edit_text, "prompts"))[:n_candidates]
    return critiques, prompts


class ProTeGiOptimizer:
    def __init__(
        self,
        labeler,
        task_description,
        label_names,
        task_type,
        answer_format_description,
        b_gp,
        b_alpha,
        final_eval_set,
        seed_prompts,
        rounds=8,
        n_candidates=8,
        reflection_example_count=3,
    ):
        self.labeler = labeler
        self.task_description = task_description
        self.label_names = label_names
        self.task_type = task_type
        self.answer_format_description = answer_format_description
        self.b_gp = b_gp
        self.b_alpha = b_alpha
        self.final_eval_set = final_eval_set
        self.seed_prompts = seed_prompts
        self.rounds = rounds
        self.n_candidates = n_candidates
        self.reflection_example_count = reflection_example_count

        self.obs_prompts = []
        self.obs_gp_acc = []
        self.obs_alpha_acc = []
        self.obs_alpha_metrics = []
        self.obs_gp_preds = []
        self.obs_alpha_preds = []
        self.obs_reflection_examples = []

        self.active_b_gp_examples = list(self.b_gp)
        self.active_b_alpha_examples = list(self.b_alpha)
        self.active_final_eval_set = list(self.final_eval_set)

        self.tracks = []
        self.final_metrics = None

    def evaluate_prompt(self, prompt, examples):
        preds = []
        used_examples = []
        skipped_example_ids = []

        for ex in examples:
            try:
                preds.append(self.labeler.predict_one(prompt, ex))
                used_examples.append(ex)
            except Exception as exc:
                if not is_skippable_prediction_error(exc):
                    raise
                skipped_example_ids.append(example_id(ex))
                print(f"Skipping sample due to content policy filter: {example_id(ex)}")

        return preds, used_examples, score_predictions(preds, used_examples, task_type=self.task_type), skipped_example_ids

    def _remove_examples_by_id(self, examples, skipped_ids):
        if not skipped_ids:
            return list(examples)
        skipped_id_set = set(skipped_ids)
        return [ex for ex in examples if example_id(ex) not in skipped_id_set]

    def _rebuild_observation_statistics(self):
        self.obs_gp_acc = [
            score_predictions(preds, self.active_b_gp_examples, task_type=self.task_type)
            for preds in self.obs_gp_preds
        ]
        self.obs_alpha_metrics = [
            compute_prediction_metrics(preds, self.active_b_alpha_examples, task_type=self.task_type)
            for preds in self.obs_alpha_preds
        ]
        self.obs_alpha_acc = [metrics["accuracy"] for metrics in self.obs_alpha_metrics]
        self.obs_reflection_examples = [
            collect_incorrect_examples(
                preds=preds,
                examples=self.active_b_gp_examples,
                task_type=self.task_type,
                label_names=self.label_names,
                max_examples=self.reflection_example_count,
            )
            for preds in self.obs_gp_preds
        ]

    def _apply_new_skips(self, gp_skipped_ids=None, alpha_skipped_ids=None):
        gp_skipped_ids = gp_skipped_ids or []
        alpha_skipped_ids = alpha_skipped_ids or []

        if gp_skipped_ids:
            keep_indices = [idx for idx, ex in enumerate(self.active_b_gp_examples) if example_id(ex) not in set(gp_skipped_ids)]
            self.active_b_gp_examples = self._remove_examples_by_id(self.active_b_gp_examples, gp_skipped_ids)
            self.obs_gp_preds = [[preds[idx] for idx in keep_indices] for preds in self.obs_gp_preds]

        if alpha_skipped_ids:
            keep_indices = [idx for idx, ex in enumerate(self.active_b_alpha_examples) if example_id(ex) not in set(alpha_skipped_ids)]
            self.active_b_alpha_examples = self._remove_examples_by_id(self.active_b_alpha_examples, alpha_skipped_ids)
            self.obs_alpha_preds = [[preds[idx] for idx in keep_indices] for preds in self.obs_alpha_preds]

        if gp_skipped_ids or alpha_skipped_ids:
            self._rebuild_observation_statistics()

    def add_observation(self, prompt):
        gp_preds, used_b_gp, gp_acc, gp_skipped_ids = self.evaluate_prompt(prompt, self.active_b_gp_examples)
        alpha_preds, used_b_alpha, alpha_acc, alpha_skipped_ids = self.evaluate_prompt(prompt, self.active_b_alpha_examples)

        if gp_skipped_ids or alpha_skipped_ids:
            self._apply_new_skips(gp_skipped_ids=gp_skipped_ids, alpha_skipped_ids=alpha_skipped_ids)
            gp_preds, used_b_gp, gp_acc, _ = self.evaluate_prompt(prompt, self.active_b_gp_examples)
            alpha_preds, used_b_alpha, alpha_acc, _ = self.evaluate_prompt(prompt, self.active_b_alpha_examples)

        reflection_examples = collect_incorrect_examples(
            preds=gp_preds,
            examples=used_b_gp,
            task_type=self.task_type,
            label_names=self.label_names,
            max_examples=self.reflection_example_count,
        )
        alpha_metrics = compute_prediction_metrics(alpha_preds, used_b_alpha, task_type=self.task_type)

        self.obs_prompts.append(prompt)
        self.obs_gp_acc.append(gp_acc)
        self.obs_alpha_acc.append(alpha_metrics["accuracy"])
        self.obs_alpha_metrics.append(alpha_metrics)
        self.obs_gp_preds.append(gp_preds)
        self.obs_alpha_preds.append(alpha_preds)
        self.obs_reflection_examples.append(reflection_examples)

    def current_best_prompt(self):
        idx = int(np.argmax(self.obs_alpha_acc))
        return self.obs_prompts[idx], self.obs_alpha_acc[idx]

    def current_best_reflection_examples(self):
        idx = int(np.argmax(self.obs_alpha_acc))
        return self.obs_reflection_examples[idx]

    def current_best_alpha_metrics(self):
        idx = int(np.argmax(self.obs_alpha_acc))
        return self.obs_alpha_metrics[idx]

    def run(self):
        for prompt in tqdm(self.seed_prompts, desc="Seed prompt evaluation"):
            self.add_observation(prompt)

        for round_index in range(self.rounds):
            best_prompt, best_score = self.current_best_prompt()
            best_metrics = self.current_best_alpha_metrics()
            reflection_examples = self.current_best_reflection_examples()
            if self.task_type == "qa":
                print(f"\n[Round {round_index + 1}] current best alpha acc = {best_score:.4f}, f1 = {best_metrics['f1']:.4f}")
            else:
                print(f"\n[Round {round_index + 1}] current best alpha acc = {best_score:.4f}")
            round_track = {
                "round": round_index,
                "best_prompt": best_prompt,
                "best_alpha_acc": best_score,
                "candidate_scores": [],
            }
            if self.task_type == "qa":
                round_track["best_alpha_f1"] = best_metrics["f1"]
            self.tracks.append(round_track)

            try:
                critiques, candidates = generate_protegi_candidates(
                    current_prompt=best_prompt,
                    task_description=self.task_description,
                    label_names=self.label_names,
                    n_candidates=self.n_candidates,
                    model=self.labeler.model,
                    task_type=self.task_type,
                    answer_format_description=self.answer_format_description,
                    reflection_examples=reflection_examples,
                )
            except Exception as exc:
                if not is_skippable_prediction_error(exc):
                    raise
                print("ProTeGi prompt editing was blocked by the content policy filter; stopping early.")
                break

            if not candidates:
                print("No ProTeGi candidates returned; stopping early.")
                break

            while True:
                candidate_infos = []
                restart_candidate_scoring = False

                for idx, candidate in enumerate(tqdm(candidates, desc=f"Scoring ProTeGi candidates round {round_index + 1}")):
                    alpha_preds, used_alpha, alpha_acc, alpha_skipped_ids = self.evaluate_prompt(candidate, self.active_b_alpha_examples)
                    if alpha_skipped_ids:
                        self._apply_new_skips(alpha_skipped_ids=alpha_skipped_ids)
                        print("Restarting ProTeGi candidate scoring after skipped B_alpha samples were removed.")
                        restart_candidate_scoring = True
                        break

                    alpha_metrics = compute_prediction_metrics(alpha_preds, used_alpha, task_type=self.task_type)

                    candidate_infos.append(
                        {
                            "prompt": candidate,
                            "alpha_acc": alpha_metrics["accuracy"],
                            **({"alpha_f1": alpha_metrics["f1"]} if self.task_type == "qa" else {}),
                            "critique": critiques[idx] if idx < len(critiques) else "",
                        }
                    )

                if not restart_candidate_scoring:
                    break

            if not candidate_infos:
                print("No valid ProTeGi candidates remained; stopping early.")
                break

            round_track["candidate_scores"] = candidate_infos
            best_candidate = max(candidate_infos, key=lambda item: item["alpha_acc"])
            self.add_observation(best_candidate["prompt"])

        best_prompt, best_alpha_acc = self.current_best_prompt()
        self.tracks.append({
            "round": self.rounds,
            "best_prompt": best_prompt,
            "best_alpha_acc": best_alpha_acc,
        })
        if self.task_type == "qa":
            self.tracks[-1]["best_alpha_f1"] = self.current_best_alpha_metrics()["f1"]
        self.tracks.append({
            "obs": [
                {
                    "prompt": prompt,
                    "gp_acc": gp_acc,
                    "alpha_acc": alpha_acc,
                    **({"alpha_f1": alpha_metrics["f1"]} if self.task_type == "qa" else {}),
                }
                for prompt, gp_acc, alpha_acc, alpha_metrics in zip(self.obs_prompts, self.obs_gp_acc, self.obs_alpha_acc, self.obs_alpha_metrics)
            ]
        })

        final_preds, used_final_eval, final_acc, final_skipped_ids = self.evaluate_prompt(best_prompt, self.active_final_eval_set)
        if final_skipped_ids:
            self.active_final_eval_set = self._remove_examples_by_id(self.active_final_eval_set, final_skipped_ids)
            final_preds, used_final_eval, final_acc, _ = self.evaluate_prompt(best_prompt, self.active_final_eval_set)

        print("\n========== FINAL ==========")
        print("Best prompt:")
        print(best_prompt)
        print(f"B_alpha best acc: {best_alpha_acc:.4f}")
        self.final_metrics = compute_prediction_metrics(final_preds, used_final_eval, task_type=self.task_type)
        if self.task_type == "qa":
            print(f"B_alpha best F1: {self.current_best_alpha_metrics()['f1']:.4f}")
        print(f"Final held-out acc: {final_acc:.4f}")
        if self.task_type == "qa":
            print(f"Final held-out F1: {self.final_metrics['f1']:.4f}")
        return best_prompt, final_acc, self.tracks


def current_day_timestamp():
    return time.strftime("%Y%m%d")


def format_runtime(total_seconds):
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"


def main():
    start_time = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="arc", choices=["arc", "mmlu_pro", "hotpot_qa", "ethos"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--b_gp_size", type=int, default=10)
    parser.add_argument("--b_alpha_size", type=int, default=32)
    parser.add_argument("--final_eval_size", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--n_candidates", type=int, default=3)
    parser.add_argument("--model", type=str, default="gpt-5.4-nano", choices=["gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4", "DeepSeek-V3.2"])
    parser.add_argument("--reflection_example_count", type=int, default=None)
    parser.add_argument("--LLM_call_trace", action="store_true", default=False)
    args = parser.parse_args()

    set_seed(args.seed)
    train_data, val_data, label_names, task_description, seed_prompts, task_config = get_dataset(args.dataset, args.seed)
    reflection_example_count = args.reflection_example_count if args.reflection_example_count is not None else task_config.get("reflection_example_count", 3)

    b_gp, b_alpha = sample_disjoint_subsets(
        train_data,
        [args.b_gp_size, args.b_alpha_size],
        seed=args.seed,
        shuffle_subsets_after_sampling=task_config.get("shuffle_subsets_after_sampling", True),
    )
    final_eval = sample_subset(val_data, args.final_eval_size, seed=args.seed + 2)

    print(f"b_gp_size: {len(b_gp)}, b_alpha_size: {len(b_alpha)}, final_eval_size: {len(final_eval)}")

    start_run_trace(
        script_name="run_protegi.py",
        config={
            **vars(args),
            "task_type": task_config["task_type"],
            "label_names": label_names,
            "reflection_example_count": reflection_example_count,
            "actual_b_gp_size": len(b_gp),
            "actual_b_alpha_size": len(b_alpha),
            "actual_final_eval_size": len(final_eval),
        },
        include_llm_calls=args.LLM_call_trace,
    )

    labeler = LLMLabeler(label_names=label_names, model=args.model, task_type=task_config["task_type"])
    optimizer = ProTeGiOptimizer(
        labeler=labeler,
        task_description=task_description,
        label_names=label_names,
        task_type=task_config["task_type"],
        answer_format_description=task_config["answer_format_description"],
        b_gp=b_gp,
        b_alpha=b_alpha,
        final_eval_set=final_eval,
        seed_prompts=seed_prompts,
        rounds=args.rounds,
        n_candidates=args.n_candidates,
        reflection_example_count=reflection_example_count,
    )
    _, _, tracks = optimizer.run()

    total_runtime = time.perf_counter() - start_time
    print(f"Total runtime: {format_runtime(total_runtime)}")
    payload = build_run_log_payload(
        total_runtime_seconds=total_runtime,
        optimizer_trace=tracks,
        summary={
            "dataset": args.dataset,
            "seed": args.seed,
            "total_runtime": format_runtime(total_runtime),
            "best_prompt": optimizer.current_best_prompt()[0],
            "best_alpha_acc": optimizer.current_best_prompt()[1],
            **({"best_alpha_f1": optimizer.current_best_alpha_metrics()["f1"]} if task_config["task_type"] == "qa" else {}),
            "final_accuracy": optimizer.final_metrics["accuracy"],
            **({"final_f1": optimizer.final_metrics["f1"]} if task_config["task_type"] == "qa" else {}),
        },
    )
    log_path = write_run_log(
        output_dir="Results",
        filename_prefix="protegi",
        filename_parts=[
            current_day_timestamp(),
            args.dataset,
            f"seed{args.seed}",
            args.model,
            f"n{args.n_candidates}",
            f"rounds{args.rounds}",
        ],
        payload=payload,
    )
    print(f"Saved run log: {log_path}")


if __name__ == "__main__":
    main()
