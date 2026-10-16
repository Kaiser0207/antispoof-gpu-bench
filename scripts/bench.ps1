# One-command setup and run on Windows (PowerShell):
#   powershell -ExecutionPolicy ByPass -File scripts\bench.ps1 [full|quick|doctor]
#
# Does exactly the steps of AGENTS.md: install uv if missing, pick the PyTorch build from the
# NVIDIA driver version (>= 580 -> cu130, 570-579 -> cu128), uv sync, doctor, fetch, run, zip results.
# Default suite: full (all 49 scenarios, about 35-50 min; it runs the quick subset first).
# Override the build with $env:AGBENCH_EXTRA = "cu130" or "cu128". Extra arguments are passed to agbench.
param(
    [ValidateSet("quick", "full", "doctor")]
    [string]$Suite = "full",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest = @()
)
$ErrorActionPreference = "Continue"   # native exit codes are checked explicitly
Set-Location (Join-Path $PSScriptRoot "..")

function Invoke-Step([string]$Exe, [string[]]$Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Step failed (exit code $LASTEXITCODE): $Exe $($Arguments -join ' ')"
        exit $LASTEXITCODE
    }
}

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Host "nvidia-smi not found: install the NVIDIA driver (580 or newer) first, then rerun."
    exit 2
}
$driver = (& nvidia-smi --query-gpu=driver_version --format=csv,noheader | Select-Object -First 1).Trim()
$major = [int]($driver.Split(".")[0])
Write-Host "NVIDIA driver: $driver"

$extra = $env:AGBENCH_EXTRA
if (-not $extra) {
    if ($major -ge 580) { $extra = "cu130" }
    elseif ($major -ge 570) { $extra = "cu128" }
    else {
        Write-Host "Driver $driver is too old: install NVIDIA driver 580 or newer (570 at the very least), then rerun."
        exit 2
    }
}
Write-Host "PyTorch build: $extra"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (https://astral.sh/uv) ..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Invoke-Step "uv" @("sync", "--frozen", "--extra", $extra)
Invoke-Step "uv" @("run", "--no-sync", "python", "-m", "agbench", "doctor")
if ($Suite -eq "doctor") { exit 0 }
Invoke-Step "uv" @("run", "--no-sync", "python", "-m", "agbench", "fetch")

& uv run --no-sync python -m agbench $Suite @Rest
$status = $LASTEXITCODE

& uv run --no-sync python -m zipfile -c agbench_results.zip results/
if ($LASTEXITCODE -eq 0) { Write-Host "Results zipped: agbench_results.zip (send this file back)" }
Write-Host "Exit code: $status (0 = all scenarios ok/oom/skipped, 1 = some error/timeout/crash, 2 = setup problem)"
exit $status
