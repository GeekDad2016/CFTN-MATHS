param(
    [string]$OutputRoot = "C:\CFTN\.datasets\math_master_experiment_100k_v3",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
& $Python -m tools.build_math_curriculum_dataset prepare `
    --config config/math_master_experiment_v1.json `
    --output $OutputRoot
& $Python -m tools.build_math_curriculum_dataset summary --output $OutputRoot
