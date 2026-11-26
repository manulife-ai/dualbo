<div align="center">

# DualBO

### Diversity-Aware Candidate Generation and Semantics-Behavior Coupled Bayesian Optimization for Prompt Search

[![Paper](https://img.shields.io/badge/Paper-arXiv%20(coming%20soon)-b31b1b.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)

**Ruotian Wu**<sup>1,2</sup> · **Ahmad Rashid**<sup>1,2</sup> · **Syed Shariyar Murtaza**<sup>3</sup> · **Utkarsh Soni**<sup>3</sup> · **Eugene Wen**<sup>3</sup> · **Pascal Poupart**<sup>1,2</sup>

<sup>1</sup>University of Waterloo &nbsp;&nbsp; <sup>2</sup>Vector Institute &nbsp;&nbsp; <sup>3</sup>Manulife

📄 **[Paper](#)** (link coming soon) &nbsp;|&nbsp; 💻 **[Code](#quickstart)** &nbsp;|&nbsp; 📚 **[Citation](#citation)**

</div>

---

## Overview

**DualBO** is a Bayesian optimization framework for black-box prompt optimization. It does two things:

- **Generates diverse prompt candidates.** The LLM first proposes distinct rewriting strategies, then turns each strategy into one candidate prompt.
- **Uses a dual-view Gaussian process surrogate.** The surrogate combines a prompt's **semantic** embedding with its **behavior** (per-example correctness). Early rounds rely on semantics, and later rounds shift toward behavior.

See the [paper](#) for the method details, results, and ablations.

## What's included

| | |
|---|---|
| **Methods** | **DualBO (ours)** · BO-LLM (behavior-only GP) · SemBO (semantic-only GP) · ProTeGi · Reflexion |
| **Datasets** | ETHOS · ARC-Challenge · MMLU-Pro · HotpotQA |
| **Models** | GPT-5.4-mini · DeepSeek-V3.2 (also supports `gpt-5.4-nano` and `gpt-5.4`) |
| **Embeddings** | `text-embedding-3-small` |

Every method can be run on every dataset with every supported model.

---

## Installation

**Prerequisites:** Python 3.12 or later, and [`uv`](https://github.com/astral-sh/uv).

```powershell
git clone https://github.com/manulife-ai/dualbo.git
cd dualbo
uv sync
```

Create a `.env` file in the repository root:

```text
OPENAI_BASE_URL=https://your-endpoint/v1
OPENAI_API_KEY=your-api-key
```

`src/Core/common_prompt_bo.py` reads these values. Use the same endpoint for the task LLM and for `text-embedding-3-small`.

---

## Quickstart

Run DualBO with the paper settings on ARC:

```powershell
uv run python .\src\Core\run_dual_view_ts.py `
    --dataset arc --seed 48 --model gpt-5.4-mini `
    --b_gp_size 64 --b_alpha_size 256 --final_eval_size 256 `
    --rounds 10 --n_candidates 5 `
    --acquisition_method ts --prompt_extension_mode diverse `
    --beh_length_scale 4 --sem_length_scale 4 `
    --semantic_to_behavior_switch_round 8
```

> 💡 **Quick check:** to confirm your setup works before a full run, use smaller values such as `--rounds 2 --n_candidates 2 --b_gp_size 8 --b_alpha_size 16 --final_eval_size 16`.

Run a baseline, for example BO-LLM on HotpotQA with DeepSeek-V3.2:

```powershell
uv run python .\src\Core\run_bo_llm_baseline.py --dataset hotpot_qa --seed 48 --b_gp_size 64 --b_alpha_size 256 --rounds 10 --n_candidates 5 --acquisition_method ts --prompt_extension_mode diverse --length_scale 4 --model DeepSeek-V3.2
```

### Experiment grids

The PowerShell wrappers in `src/scripts/` loop over seeds, datasets, and models. Edit the arrays at the top of each script to choose the grid, then launch the script from the repository root:

| Method | Script |
|---|---|
| **DualBO** | `powershell -File .\src\scripts\dualview_run.ps1` |
| BO-LLM | `powershell -File .\src\scripts\BO_LLM_run.ps1` |
| SemBO | `powershell -File .\src\scripts\semantic_baseline_run.ps1` |
| ProTeGi | `powershell -File .\src\scripts\ProTeGi_run.ps1` |
| Reflexion | `powershell -File .\src\scripts\Reflxion_run.ps1` |

### Outputs

Each run writes a JSON log to `Results/`. The log contains the run configuration, the per-round optimizer trace (candidates, GP posteriors, and selected prompts), summary metrics including final held-out accuracy, and model-call statistics.

---

## Repository structure

```text
dualbo/
├── data/
│   └── Ethos_Dataset_Binary.csv     # local ETHOS data (other datasets come from Hugging Face)
├── src/
│   ├── Core/
│   │   ├── common_prompt_bo.py      # shared: sampling, evaluation, embeddings, GP helpers, logging
│   │   ├── BO_data_loader.py        # dataset loading and task configuration
│   │   ├── run_dual_view_ts.py      # ⭐ DualBO (dual-view GP + staged generation)
│   │   ├── run_bo_llm_baseline.py   # BO-LLM (behavior-only GP)
│   │   ├── run_semantic_baseline.py # SemBO (semantic-only GP)
│   │   ├── run_protegi.py           # ProTeGi baseline
│   │   └── run_reflxion.py          # Reflexion baseline
│   └── scripts/                     # PowerShell experiment-grid launchers
├── Results/                         # JSON run logs (created automatically)
└── pyproject.toml
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{wu2026dualbo,
  title   = {DualBO: Diversity-Aware Candidate Generation and Semantics-Behavior Coupled Bayesian Optimization for Prompt Search},
  author  = {Wu, Ruotian and Rashid, Ahmad and Murtaza, Syed Shariyar and Soni, Utkarsh and Wen, Eugene and Poupart, Pascal},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

---

## Acknowledgments

This work was supported by **Manulife** and the **Vector Institute**.

## License

The code is released under the **MIT License**. The datasets keep their original licenses:

| Dataset | License |
|---|---|
| MMLU-Pro | MIT |
| ETHOS | AGPL-3.0 |
| ARC | CC-BY-SA-4.0 |
| HotpotQA | CC-BY-SA-4.0 |

> ⚠️ **Content warning:** ETHOS contains offensive and hateful text, which is used for hate-speech classification.

## Contact

For questions about the paper or code, contact Ruotian Wu ([r82wu@uwaterloo.ca](mailto:r82wu@uwaterloo.ca)) or open an issue.
