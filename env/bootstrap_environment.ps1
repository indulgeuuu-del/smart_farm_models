[CmdletBinding()]
param(
    [ValidateSet('ci', 'training', 'export', 'paddlelite')]
    [string]$Profile = 'training',
    [string]$PythonExecutable = 'py',
    [string]$PythonVersion = '3.10',
    [string]$VenvPath
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path

if ([string]::IsNullOrWhiteSpace($VenvPath)) {
    $VenvName = @{
        ci = '.venv_ci'
        training = '.venv'
        export = '.venv_paddle261_export'
        paddlelite = '.venv_paddlelite_opt'
    }[$Profile]
    $VenvPath = Join-Path $ProjectRoot $VenvName
}

$RequirementFile = Join-Path $ProjectRoot (Join-Path 'requirements' (@{
    ci = 'ci.txt'
    training = 'training-windows-py310-gpu.txt'
    export = 'export-windows-py310-paddle261.txt'
    paddlelite = 'paddlelite-windows-py310.txt'
}[$Profile]))

if (-not (Test-Path -LiteralPath $RequirementFile)) {
    throw "Missing requirements profile: $RequirementFile"
}

if (-not (Test-Path -LiteralPath (Join-Path $VenvPath 'Scripts\python.exe'))) {
    if ($PythonExecutable -eq 'py') {
        & py ("-{0}" -f $PythonVersion) -m venv $VenvPath
    }
    else {
        & $PythonExecutable -m venv $VenvPath
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create virtual environment: $VenvPath"
    }
}

$Python = Join-Path $VenvPath 'Scripts\python.exe'
& $Python -m pip install --upgrade pip setuptools wheel
& $Python -m pip install --requirement $RequirementFile
& $Python -m pip check
& $Python -c "import sys; print(sys.executable)"
Write-Output "PROFILE=$Profile"
Write-Output "VENV=$VenvPath"
Write-Output "REQUIREMENTS=$RequirementFile"
