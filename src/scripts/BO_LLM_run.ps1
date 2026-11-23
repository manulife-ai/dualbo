$seeds = @(
    48,
)

$datasets = @(
    # "mmlu_pro"
    # "arc",
    # "ethos",
    "hotpot_qa"
)

$acquisitionMethods = @(
    "ts"
    # "ucb"
)

$promptExtensionModes = @(
    # "reflection"
    "diverse"
)

$lengthScales = @(
    4
)

$models = @(
    # "gpt-5.4-mini", 
    "DeepSeek-V3.2"
)

foreach ($seed in $seeds) {
    Write-Host "Running seed: $seed"

    foreach ($dataset in $datasets) {
        Write-Host "Running dataset: $dataset"
        $reflectionExampleCount = if ($dataset -eq "mmlu_pro") { 8 } else { 3 }

        foreach ($method in $acquisitionMethods) {
            foreach ($promptExtensionMode in $promptExtensionModes) {
                foreach ($lengthScale in $lengthScales) {
                    foreach ($model in $models) {
                        Write-Host "Baseline | seed=$seed | dataset=$dataset | acquisition=$method | mode=$promptExtensionMode | ls=$lengthScale | model=$model"
                        uv run python (Join-Path $PSScriptRoot "..\Core\run_bo_llm_baseline.py") `
                            --seed=$seed `
                            --dataset=$dataset `
                            --b_gp_size=64 `
                            --b_alpha_size=256 `
                            --rounds=10 `
                            --n_candidates=5 `
                            --reflection_example_count=$reflectionExampleCount `
                            --acquisition_method=$method `
                            --prompt_extension_mode=$promptExtensionMode `
                            --length_scale=$lengthScale `
                            --model=$model
                    }
                }
            }
        }

        Write-Host "Finished dataset: $dataset"
        Write-Host "----------------------------------------"
    }

    Write-Host "Finished seed: $seed"
    Write-Host "========================================"
}
