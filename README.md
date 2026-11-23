# DUAlBO: Diversity-Aware Candidate Generation and Semantics-Behavior Coupled Bayesian Optimization for Prompt Search

This folder contains an anonymized, self-contained copy of the baseline code used for prompt optimization experiments.

## Folder structure

- `pyproject.toml`: dependency and environment definition for `uv`.
- `Data/`: local data files required by some tasks. This submission includes `Ethos_Dataset_Binary.csv` for the Ethos dataset.
- `Results/`: output directory where JSON run logs are written.
- `src/Core/`: Python source files for the baseline methods and shared utilities.
- `src/scripts/`: PowerShell wrappers that launch the baseline experiments.

## Environment setup

From the `Code_Submission` folder:

1. Create or edit the `.env` file in the `Code_Submission` root and set the API configuration:

```text
OPENAI_BASE_URL=https://your-endpoint/v1
OPENAI_API_KEY=your-api-key
```

2. Install the Python environment with `uv`:

```powershell
uv sync
```

## Running the code

All PowerShell wrappers are intended to be launched from the `Code_Submission` root.

Baseline BO-LLM:

```powershell
powershell -File .\src\scripts\BO_LLM_run.ps1
```

DualBO:

```powershell
powershell -File .\src\scripts\dualview_run.ps1
```

Semantic-only baseline:

```powershell
powershell -File .\src\scripts\semantic_baseline_run.ps1
```

ProTeGi baseline:

```powershell
powershell -File .\src\scripts\ProTeGi_run.ps1
```

Reflxion baseline:

```powershell
powershell -File .\src\scripts\Reflxion_run.ps1
```

You can also run the Python entry points directly. For example:

```powershell
uv run python .\src\Core\run_bo_llm_baseline.py --dataset hotpot_qa --seed 48 --b_gp_size 64 --b_alpha_size 256 --rounds 10 --n_candidates 5 --acquisition_method ts --prompt_extension_mode diverse --length_scale 4 --model DeepSeek-V3.2
```

## What each Python file does

- `src/Core/common_prompt_bo.py`: shared utilities for sampling, evaluation, logging, prompt expansion, embedding calls, and Gaussian process helper functions.
- `src/Core/BO_data_loader.py`: dataset loading and task configuration logic for all supported datasets.
- `src/Core/run_bo_llm_baseline.py`: baseline BO-LLM prompt optimization using behavior-space Gaussian process surrogate modeling.
- `src/Core/run_dual_view_ts.py`: dual-view prompt optimization combining behavior and semantic views in the surrogate.
- `src/Core/run_semantic_baseline.py`: semantic-only surrogate baseline using prompt embeddings.
- `src/Core/run_protegi.py`: ProTeGi-style prompt editing baseline.
- `src/Core/run_reflxion.py`: Reflxion-style reflection and rewrite baseline.

## What each PowerShell script does

- `src/scripts/BO_LLM_run.ps1`: runs the BO-LLM baseline for the configured seeds, datasets, models, acquisition methods, and prompt modes..
- `src/scripts/dualview_run.ps1`: runs the dual-view baseline for the configured experiment grid.
- `src/scripts/semantic_baseline_run.ps1`: runs the semantic-only baseline for the configured experiment grid.
- `src/scripts/ProTeGi_run.ps1`: runs the ProTeGi baseline.
- `src/scripts/Reflxion_run.ps1`: runs the Reflxion baseline.

## Outputs

Each run writes a JSON log into `Results/`. The logs contain the run configuration, optimizer trace, summary metrics, and model call statistics.

## Notes

- The Ethos task uses the local CSV file included in `Data/`.
- Other datasets are loaded programmatically by the Python loaders.
- `src/Core/common_prompt_bo.py` reads `OPENAI_BASE_URL` and `OPENAI_API_KEY` from .env file.

## License

This submission is released under the MIT License. You may use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, provided that the copyright notice and permission notice are included in all copies or substantial portions of the software. The software is provided "as is", without warranty of any kind.
