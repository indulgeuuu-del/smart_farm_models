[CmdletBinding()]
param(
    [string]$Destination,
    [switch]$InstallDependencies,
    [string]$PythonPath,
    [string]$RequirementsProfile
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
    if ([string]::IsNullOrWhiteSpace($PythonPath)) {
        $PythonPath = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
    }
    $Python = [System.IO.Path]::GetFullPath($PythonPath)
    if (-not (Test-Path -LiteralPath $Python)) {
        throw "Training environment is missing: $Python. Run bootstrap_environment.ps1 first."
    }
    if ([string]::IsNullOrWhiteSpace($RequirementsProfile)) {
        $RequirementsProfile = Join-Path $ProjectRoot 'requirements\training-windows-py310-gpu.txt'
    }
    $RequirementsProfile = [System.IO.Path]::GetFullPath($RequirementsProfile)
    if (-not (Test-Path -LiteralPath $RequirementsProfile)) {
        throw "Missing repository requirements profile: $RequirementsProfile"
    }

    $FilteredRequirements = Join-Path ([System.IO.Path]::GetTempPath()) ("paddledetection-requirements-{0}.txt" -f ([guid]::NewGuid().ToString('N')))
    try {
        $UpstreamLines = Get-Content -LiteralPath (Join-Path $Destination 'requirements.txt')
        # PaddleDetection historically listed sklearn==0.0, a deprecated shim
        # whose package name is not installable. Keep every other upstream line.
        $FilteredLines = @($UpstreamLines | Where-Object {
            $_ -notmatch '^\s*sklearn\s*==\s*0\.0(?:\s*#.*)?\s*$'
        })
        if ($FilteredLines.Count -eq 0) {
            throw 'PaddleDetection requirements became empty after filtering.'
        }
        Set-Content -LiteralPath $FilteredRequirements -Value $FilteredLines -Encoding UTF8

        # Use the repository profile as a constraint while resolving upstream
        # dependencies, then reapply it below so the documented versions win.
        & $Python -m pip install --constraint $RequirementsProfile --requirement $FilteredRequirements
        if ($LASTEXITCODE -ne 0) {
            throw 'Failed to install PaddleDetection requirements.'
        }
        & $Python -m pip install --requirement $RequirementsProfile --upgrade
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to reapply repository requirements profile: $RequirementsProfile"
        }
        & $Python scripts\common\check_requirements_profiles.py --requirements-dir (Join-Path $ProjectRoot 'requirements')
        if ($LASTEXITCODE -ne 0) {
            throw 'Repository requirements profiles are inconsistent.'
        }
        & $Python -m pip check
        if ($LASTEXITCODE -ne 0) {
            throw 'Installed dependencies are inconsistent. Run pip check manually for details.'
        }
    }
    finally {
        if (Test-Path -LiteralPath $FilteredRequirements) {
            Remove-Item -LiteralPath $FilteredRequirements -Force
        }
    }
}

Write-Output "PADDLEDET_ROOT=$Destination"
Write-Output "PADDLEDET_COMMIT=$Head"
Write-Output "PADDLEDET_LOCK=$LockPath"
