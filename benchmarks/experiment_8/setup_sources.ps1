param(
    [string]$CoditRepository = "https://github.com/kaustubhsridhar/time-series-OOD.git",
    [string]$CoditSource = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$externalRoot = Join-Path $repoRoot "benchmarks\external"
$checkout = Join-Path $externalRoot "time-series-OOD"
$workspaceCopy = Join-Path (Split-Path $repoRoot -Parent) "time-series-OOD-main"

if (-not $CoditSource) {
    if (Test-Path (Join-Path $workspaceCopy "ours\gait\check_OOD_gait.py")) {
        $CoditSource = $workspaceCopy
    } else {
        $CoditSource = $checkout
    }
}

if (-not (Test-Path (Join-Path $CoditSource "ours\gait\check_OOD_gait.py"))) {
    if (Test-Path $CoditSource) {
        throw "CoDiT path exists but has no ours/gait/check_OOD_gait.py: $CoditSource"
    }
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw "Git is required to clone CoDiT. Install Git for Windows or pass -CoditSource."
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $CoditSource -Parent) | Out-Null
    git clone --depth 1 --single-branch --branch main $CoditRepository $CoditSource
    if ($LASTEXITCODE -ne 0) { throw "Cloning the official CoDiT repository failed." }
}

$gaitReadme = Join-Path $CoditSource "ours\gait\README.md"
if (-not (Test-Path $gaitReadme)) {
    throw "CoDiT GAIT instructions are missing: $gaitReadme"
}

Write-Host "CoDiT source: $((Resolve-Path $CoditSource).Path)"
Write-Host "Read the official GAIT README below. It links the exact dataset archive and checkpoints."
Get-Content -LiteralPath $gaitReadme
