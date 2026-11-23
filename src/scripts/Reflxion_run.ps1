$seeds = @(
    48
)

$datasets = @(
    # "mmlu_pro"
    # "arc",
    # "ethos",
    "hotpot_qa"
)

$models = @(
    # "gpt-5.4-mini",
    "DeepSeek-V3.2"
)

foreach ($seed in $seeds) {
    Write-Host "Running seed: $seed"

    foreach ($dataset in $datasets) {
        foreach ($model in $models) {
            Write-Host "Running dataset: $dataset"
            Write-Host "Reflxion | seed=$seed | dataset=$dataset | model=$model"
            $reflectionExampleCount = if ($dataset -eq "mmlu_pro") { 8 } else { 3 }

            uv run python (Join-Path $PSScriptRoot "..\Core\run_reflxion.py") `
                --seed=$seed `
                --dataset=$dataset `
                --b_gp_size=64 `
                --b_alpha_size=256 `
                --rounds=10 `
                --n_candidates=5 `
                --reflection_example_count=$reflectionExampleCount `
                --model=$model
        }

        Write-Host "Finished dataset: $dataset"
        Write-Host "----------------------------------------"
    }

    Write-Host "Finished seed: $seed"
    Write-Host "========================================"
}
