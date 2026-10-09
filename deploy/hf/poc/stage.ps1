# Copies exactly what the PoC dashboard Space needs into a folder (your local
# clone of the Space repo), mirroring the project layout. Copies files only:
# git add / commit / push stay manual (see deploy/hf/poc/DEPLOY.md).
#
# Usage, from the project root:
#   .\deploy\hf\poc\stage.ps1 -Target C:\Users\DFILATOVA\Desktop\hf-poc
param(
    [Parameter(Mandatory = $true)][string]$Target
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path "$PSScriptRoot\..\..\..").Path

$checkpoints = @(
    "backend\model\resnet50_best.pth",
    "model_backups\mask2former_swint_best.pth",
    "model_backups\eomt_base_best.pth"
)
foreach ($c in $checkpoints) {
    if (-not (Test-Path (Join-Path $Root $c))) { throw "Missing checkpoint: $c" }
}
foreach ($f in @("poc_app\data\eda_stats.json", "poc_app\data\poc_results.json")) {
    if (-not (Test-Path (Join-Path $Root $f))) {
        throw "Missing $f - run scripts/build_eda_stats.py and scripts/build_poc_dashboard.py"
    }
}

New-Item -ItemType Directory -Force $Target | Out-Null

# Space files (Dockerfile, README with the Space header) go at the root
foreach ($f in @("Dockerfile", "README.md")) {
    Copy-Item (Join-Path $PSScriptRoot $f) (Join-Path $Target $f) -Force
}

# LFS rules: keep the Space's own .gitattributes (HF creates one with ~35
# rules) and only append ours that are missing (*.pth, *.png)
$attrTarget = Join-Path $Target ".gitattributes"
$existing = if (Test-Path $attrTarget) { Get-Content $attrTarget } else { @() }
$added = @()
foreach ($rule in Get-Content (Join-Path $PSScriptRoot ".gitattributes")) {
    $pattern = ($rule -split "\s+")[0]
    if (-not ($existing | Where-Object { ($_ -split "\s+")[0] -eq $pattern })) {
        Add-Content -Path $attrTarget -Value $rule -Encoding utf8
        $added += $pattern
    }
}
if ($added) { "LFS rules added to .gitattributes: $($added -join ', ')" }

# Code + data folders (without caches)
foreach ($d in @("cityvision", "src", "poc_app")) {
    $dest = Join-Path $Target $d
    if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
    Copy-Item (Join-Path $Root $d) $dest -Recurse
    Get-ChildItem $dest -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
}

# Single files, keeping their relative path
foreach ($f in @("scripts\evaluate_poc.py") + $checkpoints) {
    $dest = Join-Path $Target $f
    New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
    Copy-Item (Join-Path $Root $f) $dest -Force
}

$files = Get-ChildItem $Target -Recurse -File | Where-Object { $_.FullName -notmatch "\\\.git\\" }
$size = ($files | Measure-Object Length -Sum).Sum / 1MB
$ckpt = ($files | Where-Object Extension -eq ".pth" | Measure-Object Length -Sum).Sum / 1MB
"Staged into $Target ({0:N0} MB, of which {1:N0} MB checkpoints via Git LFS)" -f $size, $ckpt
