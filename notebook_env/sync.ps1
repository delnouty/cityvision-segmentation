<#
.SYNOPSIS
    Refresh notebook_env/ from the repository copies.

.DESCRIPTION
    notebook_env/ holds copies, so it goes stale every time a notebook is re-run
    and saved. This script re-copies the tracked files and verifies each one by
    SHA256 afterwards.

.PARAMETER Check
    Report drift without copying anything. Exits 1 if any file is stale, so it
    can be used as a pre-commit guard.

.EXAMPLE
    .\sync.ps1
    Copy anything that has drifted, then verify.

.EXAMPLE
    .\sync.ps1 -Check
    Only report. Exit code 1 means something is out of sync.
#>
[CmdletBinding()]
param(
    [switch]$Check
)

$ErrorActionPreference = 'Stop'

# This script lives in notebook_env/, so the repo root is its parent.
$envRoot  = $PSScriptRoot
$repoRoot = Split-Path -Parent $envRoot

# source (relative to repo root) -> destination (relative to notebook_env/)
$files = [ordered]@{
    'notebooks/training_segnet.ipynb'     = 'notebooks/training_segnet.ipynb'
    'notebooks/training_comparison.ipynb' = 'notebooks/training_comparison.ipynb'
    'notebooks/training_augmented.ipynb'  = 'notebooks/training_augmented.ipynb'
    'src/dataloader.py'                   = 'src/dataloader.py'
}

function Get-Sha256($path) {
    if (-not (Test-Path -LiteralPath $path)) { return $null }
    return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
}

$mode = if ($Check) { 'CHECK' } else { 'SYNC' }
Write-Host ""
Write-Host "notebook_env $mode  ($repoRoot)"
Write-Host ("-" * 62)

$stale = 0
$copied = 0
$failed = 0

foreach ($src in $files.Keys) {
    $srcPath = Join-Path $repoRoot $src
    $dstPath = Join-Path $envRoot  $files[$src]
    $name    = Split-Path -Leaf $src

    if (-not (Test-Path -LiteralPath $srcPath)) {
        Write-Host ("  {0,-30} MISSING in repo" -f $name) -ForegroundColor Red
        $failed++
        continue
    }

    $srcHash = Get-Sha256 $srcPath
    $dstHash = Get-Sha256 $dstPath

    if ($srcHash -eq $dstHash) {
        Write-Host ("  {0,-30} in sync" -f $name) -ForegroundColor DarkGray
        continue
    }

    $stale++
    if ($Check) {
        $state = if ($null -eq $dstHash) { 'absent' } else { 'differs' }
        Write-Host ("  {0,-30} STALE ({1})" -f $name, $state) -ForegroundColor Yellow
        continue
    }

    # Copy, then verify the copy actually matches — a silent partial write here
    # would be worse than the drift we are fixing.
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dstPath) | Out-Null
    Copy-Item -LiteralPath $srcPath -Destination $dstPath -Force

    if ((Get-Sha256 $dstPath) -eq $srcHash) {
        Write-Host ("  {0,-30} updated" -f $name) -ForegroundColor Green
        $copied++
    }
    else {
        Write-Host ("  {0,-30} COPY VERIFY FAILED" -f $name) -ForegroundColor Red
        $failed++
    }
}

Write-Host ("-" * 62)

if ($failed -gt 0) {
    Write-Host "$failed file(s) failed." -ForegroundColor Red
    exit 2
}

if ($Check) {
    if ($stale -gt 0) {
        Write-Host "$stale file(s) out of sync. Run .\sync.ps1 to fix." -ForegroundColor Yellow
        exit 1
    }
    Write-Host "All files in sync." -ForegroundColor Green
    exit 0
}

if ($copied -eq 0) {
    Write-Host "Nothing to do — already in sync." -ForegroundColor Green
}
else {
    Write-Host "$copied file(s) updated, all verified by SHA256." -ForegroundColor Green
}
exit 0
