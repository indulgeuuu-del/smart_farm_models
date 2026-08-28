[CmdletBinding()]
param(
    [string]$Destination,
    [switch]$InstallDependencies
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$LockPath = Join-Path $ProjectRoot 'third_party\paddledetection.lock.json'
if (-not (Test-Path -LiteralPath $LockPath)) {
    throw "Missing PaddleDetection lock file: $LockPath"
}

$Lock = Get-Content -LiteralPath $LockPath -Raw | ConvertFrom-Json
if ([string]::IsNullOrWhiteSpace($Destination)) {
    $Destination = Join-Path $ProjectRoot $Lock.destination
}
$Destination = [System.IO.Path]::GetFullPath($Destination)

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw 'git is required to obtain the pinned PaddleDetection source.'
}

if (Test-Path -LiteralPath $Destination) {
    if (-not (Test-Path -LiteralPath (Join-Path $Destination '.git'))) {
        throw "Destination exists but is not a Git checkout: $Destination"
    }
    $Status = git -C $Destination status --porcelain
    if ($Status) {
        throw "Refusing to change a dirty third-party checkout: $Destination"
    }
    git -C $Destination fetch --tags --force origin $Lock.commit
}
else {
    $Parent = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Force -Path $Parent | Out-Null
    git clone --filter=blob:none $Lock.upstream $Destination
}
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to fetch PaddleDetection.'
}

git -C $Destination checkout --detach $Lock.commit
if ($LASTEXITCODE -ne 0) {
    throw "Failed to check out locked PaddleDetection commit $($Lock.commit)."
}
$Head = (git -C $Destination rev-parse HEAD).Trim()
if ($Head -ne $Lock.commit) {
    throw "PaddleDetection commit mismatch. Expected $($Lock.commit), got $Head"
}

if ($InstallDependencies) {
    $Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $Python)) {
        throw "Training environment is missing: $Python. Run bootstrap_environment.ps1 first."
    }
    & $Python -m pip install --requirement (Join-Path $Destination 'requirements.txt')
    & $Python -m pip check
}

Write-Output "PADDLEDET_ROOT=$Destination"
Write-Output "PADDLEDET_COMMIT=$Head"
Write-Output "PADDLEDET_LOCK=$LockPath"
