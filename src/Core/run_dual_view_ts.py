import argparse
import time
import numpy as np
from tqdm import tqdm

from common_prompt_bo import (
    set_seed, sample_subset, sample_disjoint_subsets, score_predictions,
    LLMLabeler, expand_prompts_with_gpt, get_semantic_embedding,
    build_kernel_matrix, build_cross_kernel_matrix, combine_kernel_matrices,
    normalize_embedding, solve_psd_linear_system, stabilize_covariance_matrix,
    ucb_score,
    prediction_correctness_vector, collect_incorrect_examples, compute_prediction_metrics,
    example_id, is_skippable_prediction_error,
    start_run_trace, build_run_log_payload, write_run_log,
)

from BO_data_loader import get_dataset

# ---------------------------
# Dual-view GP + TS
# ---------------------------

def format_runtime(total_seconds: float) -> str:
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"

def current_day_timestamp() -> str:
    return time.strftime("%Y%m%d")


MAX_LOGGED_WRONG_EXAMPLES = 5

def build_train_kernel_matrix(train_beh, train_sem, lambda_t, beh_ls, sem_ls, noise):
    K_beh = build_kernel_matrix(train_beh, length_scale=beh_ls, noise=0.0)
    K_sem = build_kernel_matrix(train_sem, length_scale=sem_ls, noise=0.0)
    K = combine_kernel_matrices(K_beh, K_sem, lambda_t=lambda_t)
    K += noise * np.eye(len(train_beh))
    return K

def gp_posterior_candidates_dual_view(
    train_beh,
    train_sem,
    train_obs_acc,
    train_control_acc,
    cand_beh,
    cand_sem,
    lambda_t=0.7,
    beh_ls=1.0,
    sem_ls=1.0,
    noise=1e-4,
):
    """
    Returns posterior mean vector and covariance matrix over candidates.
    Mean function m(p) = control accuracy on B_GP, same spirit as BO-LLM.
    """
    m = len(cand_beh)

    K = build_train_kernel_matrix(train_beh, train_sem, lambda_t, beh_ls, sem_ls, noise)
    K_s_beh = build_cross_kernel_matrix(train_beh, cand_beh, length_scale=beh_ls)
    K_s_sem = build_cross_kernel_matrix(train_sem, cand_sem, length_scale=sem_ls)
    K_s = combine_kernel_matrices(K_s_beh, K_s_sem, lambda_t=lambda_t)

    K_ss_beh = build_kernel_matrix(cand_beh, length_scale=beh_ls, noise=0.0)
    K_ss_sem = build_kernel_matrix(cand_sem, length_scale=sem_ls, noise=0.0)
    K_ss = combine_kernel_matrices(K_ss_beh, K_ss_sem, lambda_t=lambda_t)
    K_ss += 1e-8 * np.eye(m)

    m_train = train_control_acc
    m_cand = np.array([float(np.mean(v)) for v in cand_beh], dtype=np.float64)

    residual = train_obs_acc - m_train
    alpha = solve_psd_linear_system(K, residual)
    solved_cross_kernel = solve_psd_linear_system(K, K_s)

    mu = m_cand + K_s.T @ alpha
    cov = K_ss - K_s.T @ solved_cross_kernel
    cov = stabilize_covariance_matrix(cov + 1e-8 * np.eye(m))
    return mu, cov

class SemanticEncoder:
    def __init__(self, model_name="text-embedding-3-small"):
        self.model = model_name
        self.cache = {}

    def encode(self, text: str):
        if text not in self.cache:
            response = get_semantic_embedding(text, model_name=self.model, dimension=512)
            self.cache[text] = normalize_embedding(response)
        return self.cache[text]

class DualViewTSOptimizer:
    def __init__(
        self,
        labeler,
        semantic_encoder,
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
        noise=1e-4,
        beh_ls=1.0,
        sem_ls=1.0,
        semantic_to_behavior_switch_round=8,
        random_state=42,
        kappa_start=2.0,
        kappa_end=0.5,
        acquisition_method="ucb",
        prompt_extension_mode="reflection",
        log_wrong_examples=False,
    ):
        self.labeler = labeler
        self.semantic_encoder = semantic_encoder
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
        self.noise = noise
        self.kappa_start = kappa_start
        self.kappa_end = kappa_end
        self.acquisition_method = acquisition_method
        self.prompt_extension_mode = prompt_extension_mode
        self.log_wrong_examples = log_wrong_examples
        self.beh_ls = beh_ls
        self.sem_ls = sem_ls
        self.semantic_to_behavior_switch_round = semantic_to_behavior_switch_round
        self.rng = np.random.default_rng(random_state)

        self.obs_prompts = []
        self.obs_beh = []
        self.obs_sem = []
        self.obs_control_acc = []
        self.obs_alpha_acc = []
        self.obs_alpha_metrics = []
        self.obs_alpha_preds = []
        self.obs_reflection_examples = []

        self.active_b_gp_examples = list(self.b_gp)
        self.active_b_alpha_examples = list(self.b_alpha)
        self.active_final_eval_set = list(self.final_eval_set)
        
        self.tracks = []
        self.final_metrics = None

    def _build_wrong_example_log(self, prompt, preds, examples):
        wrong_examples = []
        for pred, ex in zip(preds, examples):
            if score_predictions([pred], [ex], task_type=self.task_type) >= 1.0:
                continue
            wrong_examples.append(
                {
                    "prompt": prompt,
                    "input": ex["text"],
                    "output": str(pred),
                    "label": str(ex["label"]),
                }
            )
            if len(wrong_examples) >= MAX_LOGGED_WRONG_EXAMPLES:
                break
        return wrong_examples

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

    def correctness_vector(self, preds, examples):
        return prediction_correctness_vector(preds, examples, task_type=self.task_type)

    def _remove_examples_by_id(self, examples, skipped_ids):
        if not skipped_ids:
            return list(examples)
        skipped_id_set = set(skipped_ids)
        return [ex for ex in examples if example_id(ex) not in skipped_id_set]

    def _rebuild_observation_statistics(self):
        self.obs_control_acc = [float(np.mean(vec)) if len(vec) else 0.0 for vec in self.obs_beh]
        self.obs_alpha_metrics = [
            compute_prediction_metrics(preds, self.active_b_alpha_examples, task_type=self.task_type)
            for preds in self.obs_alpha_preds
        ]
        self.obs_alpha_acc = [metrics["accuracy"] for metrics in self.obs_alpha_metrics]
        self.obs_reflection_examples = [
            collect_incorrect_examples(
                preds=preds,
                examples=self.active_b_alpha_examples,
                task_type=self.task_type,
                label_names=self.label_names,
                max_examples=self.reflection_example_count,
            )
            for preds in self.obs_alpha_preds
        ]

    def _apply_new_skips(self, gp_skipped_ids=None, alpha_skipped_ids=None):
        gp_skipped_ids = gp_skipped_ids or []
        alpha_skipped_ids = alpha_skipped_ids or []

        if gp_skipped_ids:
            keep_mask = np.array([example_id(ex) not in set(gp_skipped_ids) for ex in self.active_b_gp_examples], dtype=bool)
            self.active_b_gp_examples = self._remove_examples_by_id(self.active_b_gp_examples, gp_skipped_ids)
            self.obs_beh = [vec[keep_mask] for vec in self.obs_beh]

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

        sem_vec = self.semantic_encoder.encode(prompt)
        beh_vec = self.correctness_vector(gp_preds, used_b_gp)
        alpha_metrics = compute_prediction_metrics(alpha_preds, used_b_alpha, task_type=self.task_type)
        reflection_examples = collect_incorrect_examples(
            preds=alpha_preds,
            examples=used_b_alpha,
            task_type=self.task_type,
            label_names=self.label_names,
            max_examples=self.reflection_example_count,
        )

        self.obs_prompts.append(prompt)
        self.obs_beh.append(beh_vec)
        self.obs_sem.append(np.array(sem_vec, dtype=np.float64))
        self.obs_control_acc.append(gp_acc)
        self.obs_alpha_acc.append(alpha_metrics["accuracy"])
        self.obs_alpha_metrics.append(alpha_metrics)
        self.obs_alpha_preds.append(alpha_preds)
        self.obs_reflection_examples.append(reflection_examples)

        observation = {
            "prompt": prompt,
            "control_acc": gp_acc,
            "alpha_acc": alpha_metrics["accuracy"],
            "reflection_examples": reflection_examples,
        }
        if self.task_type == "qa":
            observation["alpha_f1"] = alpha_metrics["f1"]
        if self.log_wrong_examples:
            observation["wrong_examples"] = self._build_wrong_example_log(prompt, alpha_preds, used_b_alpha)
        return observation

    def current_best_prompt(self):
        idx = int(np.argmax(self.obs_alpha_acc))
        return self.obs_prompts[idx], self.obs_alpha_acc[idx]

    def current_best_reflection_examples(self):
        idx = int(np.argmax(self.obs_alpha_acc))
        return self.obs_reflection_examples[idx]

    def current_best_alpha_metrics(self):
        idx = int(np.argmax(self.obs_alpha_acc))
        return self.obs_alpha_metrics[idx]

    def lambda_t(self, round_idx):
        # early rounds: more semantic prior
        # later rounds: more behavioral signal
        return min(1.0, (round_idx + 1) / float(self.semantic_to_behavior_switch_round))

    def run(self):
        for p in tqdm(self.seed_prompts, desc="Seed prompt evaluation"):
            self.add_observation(p)

        for t in range(self.rounds):
            best_prompt, best_score = self.current_best_prompt()
            best_metrics = self.current_best_alpha_metrics()
            reflection_examples = self.current_best_reflection_examples()
            lam = self.lambda_t(t)
            kappa = self.kappa_start + (self.kappa_end - self.kappa_start) * (t / max(1, self.rounds - 1))
            round_track = {
                "round": t,
                "best_prompt": best_prompt,
                "best_alpha_acc": best_score,
                "lambda_t": lam,
                "kappa": kappa,
                "candidate_surrogate_estimates": [],
            }
            if self.task_type == "qa":
                round_track["best_alpha_f1"] = best_metrics["f1"]
            self.tracks.append(round_track)

            if self.task_type == "qa":
                print(f"\n[Round {t+1}] current best alpha acc = {best_score:.4f}, f1 = {best_metrics['f1']:.4f}, lambda_t={lam:.3f}")
            else:
                print(f"\n[Round {t+1}] current best alpha acc = {best_score:.4f}, lambda_t={lam:.3f}")
            # print(f"Observations: {[(x, y, z, w, v[0:16]) for x, y, z, w, v in zip(self.obs_prompts, self.obs_control_acc, self.obs_alpha_acc, self.obs_beh, self.obs_sem)]}")

            try:
                candidates = expand_prompts_with_gpt(
                    current_best_prompt=best_prompt,
                    task_description=self.task_description,
                    label_names=self.label_names,
                    n_candidates=self.n_candidates,
                    model=self.labeler.model,
                    task_type=self.task_type,
                    answer_format_description=self.answer_format_description,
                    reflection_examples=reflection_examples,
                    mode=self.prompt_extension_mode,
                )
            except Exception as exc:
                if not is_skippable_prediction_error(exc):
                    raise
                print("Prompt expansion was blocked by the content policy filter; stopping early.")
                break

            if not candidates:
                print("No candidates returned; stopping early.")
                break

            while True:
                candidate_behaviors = []
                candidate_semantics = []
                candidate_infos = []
                restart_candidate_scoring = False

                for cand in tqdm(candidates, desc=f"Embedding/scoring candidates round {t+1}"):
                    cand_preds, used_b_gp, _, gp_skipped_ids = self.evaluate_prompt(cand, self.active_b_gp_examples)
                    if gp_skipped_ids:
                        self._apply_new_skips(gp_skipped_ids=gp_skipped_ids)
                        print("Restarting candidate scoring after skipped B_GP samples were removed.")
                        restart_candidate_scoring = True
                        break

                    candidate_behaviors.append(self.correctness_vector(cand_preds, used_b_gp))
                    candidate_semantics.append(np.array(self.semantic_encoder.encode(cand), dtype=np.float64))

                if not restart_candidate_scoring:
                    break

            mu, cov = gp_posterior_candidates_dual_view(
                train_beh=self.obs_beh,
                train_sem=self.obs_sem,
                train_obs_acc=np.array(self.obs_alpha_acc, dtype=np.float64),
                train_control_acc=np.array(self.obs_control_acc, dtype=np.float64),
                cand_beh=candidate_behaviors,
                cand_sem=candidate_semantics,
                lambda_t=lam,
                beh_ls=self.beh_ls,
                sem_ls=self.sem_ls,
                noise=self.noise,
            )
            
            candidate_variances = np.maximum(np.diag(cov), 1e-8)
            sigma = np.sqrt(candidate_variances)
            if self.acquisition_method == "ucb":
                for cand, cand_mu, cand_var, cand_sigma in zip(candidates, mu, candidate_variances, sigma):
                    score = ucb_score(float(cand_mu), float(cand_sigma), kappa=kappa)
                    candidate_infos.append((cand, float(cand_mu), float(cand_var), float(cand_sigma), float(score)))
            elif self.acquisition_method == "ts":
                sampled_f = self.rng.multivariate_normal(mean=mu, cov=cov)
                candidate_infos = [
                    (cand, float(cand_mu), float(cand_var), float(cand_sigma), float(sampled_score))
                    for cand, cand_mu, cand_var, cand_sigma, sampled_score in zip(candidates, mu, candidate_variances, sigma, sampled_f)
                ]
            else:
                raise ValueError(f"Unknown acquisition method: {self.acquisition_method}")

            round_track["candidate_surrogate_estimates"] = [
                {
                    "prompt": cand,
                    "surrogate_mean": float(cand_mu),
                    "surrogate_variance": float(cand_var),
                }
                for cand, cand_mu, cand_var, _, _ in candidate_infos
            ]

            candidate_infos.sort(key=lambda x: x[4], reverse=True)
            # print(f"candidates_info: {[(x[0], x[1], x[2], x[3]) for x in candidate_infos]}")
            
            chosen_prompt = candidate_infos[0][0]
            observation = self.add_observation(chosen_prompt)
            if self.log_wrong_examples:
                round_track["wrong_examples"] = observation["wrong_examples"]
        
        
        best_prompt, best_alpha_acc = self.current_best_prompt()
        self.tracks.append({
                "round": self.rounds,
                "best_prompt": best_prompt,
                "best_alpha_acc": best_alpha_acc,
                "lambda_t": self.lambda_t(self.rounds),
                "kappa": self.kappa_end,
            })
        if self.task_type == "qa":
            self.tracks[-1]["best_alpha_f1"] = self.current_best_alpha_metrics()["f1"]
        self.tracks.append({"obs": [{
            "prompt": prompt,
            "control_acc": control_acc,
            "alpha_acc": alpha_acc,
            **({"alpha_f1": alpha_metrics["f1"]} if self.task_type == "qa" else {}),
            "beh": beh.tolist(),
            "sem": sem[0:16].tolist(),
        } for prompt, control_acc, alpha_acc, alpha_metrics, beh, sem in zip(self.obs_prompts, self.obs_control_acc, self.obs_alpha_acc, self.obs_alpha_metrics, self.obs_beh, self.obs_sem)]})
        final_preds, used_final_eval, final_acc, final_skipped_ids = self.evaluate_prompt(best_prompt, self.active_final_eval_set)
        if final_skipped_ids:
            self.active_final_eval_set = self._remove_examples_by_id(self.active_final_eval_set, final_skipped_ids)
            final_preds, used_final_eval, final_acc, _ = self.evaluate_prompt(best_prompt, self.active_final_eval_set)

        self.final_metrics = compute_prediction_metrics(final_preds, used_final_eval, task_type=self.task_type)

        print("\n========== FINAL ==========")
        print("Best prompt:")
        print(best_prompt)
        print(f"B_alpha best acc: {best_alpha_acc:.4f}")
        if self.task_type == "qa":
            print(f"B_alpha best F1: {self.current_best_alpha_metrics()['f1']:.4f}")
        print(f"Final held-out acc: {final_acc:.4f}")
        if self.task_type == "qa":
            print(f"Final held-out F1: {self.final_metrics['f1']:.4f}")
        return best_prompt, final_acc, self.tracks

def main():
    start_time = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="arc", choices=["arc", "mmlu_pro", "hotpot_qa", "ethos"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--b_gp_size", type=int, default=12)
    parser.add_argument("--b_alpha_size", type=int, default=64)
    parser.add_argument("--final_eval_size", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--n_candidates", type=int, default=3)
    parser.add_argument("--model", type=str, default="gpt-5.4-nano", choices=["gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4", "DeepSeek-V3.2"])
    parser.add_argument("--acquisition_method", type=str, default="ucb", choices=["ucb", "ts"])
    parser.add_argument("--prompt_extension_mode", type=str, default="reflection", choices=["baseline", "reflection", "diverse"])
    parser.add_argument("--beh_length_scale", type=float, default=8.0)
    parser.add_argument("--sem_length_scale", type=float, default=8.0)
    parser.add_argument("--semantic_to_behavior_switch_round", type=int, default=8)
    parser.add_argument("--reflection_example_count", type=int, default=None)
    parser.add_argument("--LLM_call_trace", action="store_true", default=False)
    parser.add_argument("--log_wrong_examples", action="store_true", default=True)
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
        script_name="run_dual_view_ts.py",
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
        captured_call_types=(
            ["prompt_expansion_diverse_stage1", "prompt_expansion_diverse_stage2"]
            if args.prompt_extension_mode == "diverse"
            else None
        ),
    )

    labeler = LLMLabeler(label_names=label_names, model=args.model, task_type=task_config["task_type"])
    semantic_encoder = SemanticEncoder()

    optimizer = DualViewTSOptimizer(
        labeler=labeler,
        semantic_encoder=semantic_encoder,
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
        beh_ls=args.beh_length_scale,
        sem_ls=args.sem_length_scale,
        semantic_to_behavior_switch_round=args.semantic_to_behavior_switch_round,
        random_state=args.seed,
        acquisition_method=args.acquisition_method,
        prompt_extension_mode=args.prompt_extension_mode,
        log_wrong_examples=args.log_wrong_examples,
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
        filename_prefix="dualview",
        filename_parts=[
            current_day_timestamp(),
            args.dataset,
            f"seed{args.seed}",
            args.model,
            args.acquisition_method,
            args.prompt_extension_mode,
            f"n{args.n_candidates}",
            f"bgp{args.b_gp_size}",
            f"behLs{args.beh_length_scale:g}",
            f"semLs{args.sem_length_scale:g}",
            f"switch{args.semantic_to_behavior_switch_round}",
        ],
        payload=payload,
    )
    print(f"Saved run log: {log_path}")

if __name__ == "__main__":
    main()
