param(
    [switch]$ProbeOnly,
    [int]$WaitSeconds = 5,
    [string]$LogDir
)

$ErrorActionPreference = "Stop"
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ImageDir = Join-Path $ProjectRoot "datasets\00_lane_seg\raw\image_set_labelme_sample"
$Labels = Join-Path $ProjectRoot "datasets\00_lane_seg\metadata\image_set_labelme_sample\labels.txt"
$OutputDir = Join-Path $ProjectRoot "datasets\00_lane_seg\labelme_json\image_set_labelme_sample"

if (-not $LogDir) {
    $LogDir = Join-Path $ProjectRoot "logs\labelme"
}

foreach ($Path in @($Python, $ImageDir, $Labels, $OutputDir)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required path not found: $Path"
    }
}

$Version = & $Python -m labelme --version
Write-Output "project_root=$ProjectRoot"
Write-Output "python=$Python"
Write-Output "labelme_version=$Version"
Write-Output "image_dir=$ImageDir"
Write-Output "labels=$Labels"
Write-Output "output_dir=$OutputDir"

if ($ProbeOnly) {
    Write-Output "probe_only=True"
    exit 0
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$StdoutLog = Join-Path $LogDir "labelme_lane_seg_sample_stdout.log"
$StderrLog = Join-Path $LogDir "labelme_lane_seg_sample_stderr.log"
Remove-Item -LiteralPath $StdoutLog, $StderrLog -Force -ErrorAction SilentlyContinue

$Args = @(
    "-m", "labelme",
    $ImageDir,
    "--labels", $Labels,
    "--output", $OutputDir,
    "--logger-level", "debug"
)

$Process = Start-Process `
    -FilePath $Python `
    -ArgumentList $Args `
    -RedirectStandardOutput $StdoutLog `
    -RedirectStandardError $StderrLog `
    -PassThru

Start-Sleep -Seconds $WaitSeconds

Write-Output "labelme_pid=$($Process.Id)"
Write-Output "stdout_log=$StdoutLog"
Write-Output "stderr_log=$StderrLog"

if ($Process.HasExited) {
    Write-Output "labelme_alive=False"
    Write-Output "labelme_exit_code=$($Process.ExitCode)"
    Write-Output "--- stderr tail ---"
    if (Test-Path -LiteralPath $StderrLog) {
        Get-Content -LiteralPath $StderrLog -Encoding UTF8 -Tail 80
    }
    exit 1
}

Write-Output "labelme_alive=True"
