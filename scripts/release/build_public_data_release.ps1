param(
    [string]$Root = (Get-Location).Path,
    [string]$WorkDir = '',
    [double]$TargetGiB = 1.55
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path -LiteralPath $Root).Path.TrimEnd('\')
if ([string]::IsNullOrWhiteSpace($WorkDir)) {
    $WorkDir = Join-Path $Root 'tmp/public_release_20260823'
}
$WorkDir = [System.IO.Path]::GetFullPath($WorkDir)
$PartsDir = Join-Path $WorkDir 'parts'
$Roots = @(
    'Dataset_all',
    'MyDataset',
    'datasets',
    'outputs',
    'output',
    'bbox_pr_curve',
    'vdl_log_dir',
    'tmp',
    'logs',
    'deploy/exported_models',
    'deploy/onboard_handoff'
)

if (Test-Path -LiteralPath $WorkDir) {
    Remove-Item -LiteralPath $WorkDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $PartsDir | Out-Null

$utf8 = New-Object System.Text.UTF8Encoding($false)
$manifestWriter = New-Object System.IO.StreamWriter(
    (Join-Path $WorkDir 'source_manifest.tsv'),
    $false,
    $utf8
)
$manifestWriter.WriteLine("path`tbytes")

$targetBytes = [int64]($TargetGiB * 1GB)
$partNumber = 1
$partBytes = [int64]0
$partFiles = 0
$partWriter = $null
$totalBytes = [int64]0
$totalFiles = 0
$plan = @()

try {
    foreach ($relativeRoot in $Roots) {
        $absoluteRoot = Join-Path $Root $relativeRoot
        if (-not (Test-Path -LiteralPath $absoluteRoot)) {
            continue
        }

        foreach ($path in [System.IO.Directory]::EnumerateFiles(
            $absoluteRoot,
            '*',
            [System.IO.SearchOption]::AllDirectories
        )) {
            if ($path.StartsWith(
                $WorkDir + [System.IO.Path]::DirectorySeparatorChar,
                [System.StringComparison]::OrdinalIgnoreCase
            )) {
                continue
            }

            $file = New-Object System.IO.FileInfo($path)
            if ($null -eq $partWriter) {
                $partPath = Join-Path $PartsDir ('part-{0:D3}.txt' -f $partNumber)
                $partWriter = New-Object System.IO.StreamWriter($partPath, $false, $utf8)
            }

            if ($partBytes -gt 0 -and ($partBytes + $file.Length) -gt $targetBytes) {
                $partWriter.Dispose()
                $plan += [pscustomobject]@{
                    Part = 'part-{0:D3}' -f $partNumber
                    Files = $partFiles
                    Bytes = $partBytes
                    GiB = [math]::Round($partBytes / 1GB, 3)
                }
                $partNumber++
                $partBytes = 0
                $partFiles = 0
                $partPath = Join-Path $PartsDir ('part-{0:D3}.txt' -f $partNumber)
                $partWriter = New-Object System.IO.StreamWriter($partPath, $false, $utf8)
            }

            $relativePath = $path.Substring($Root.Length + 1)
            $partWriter.WriteLine($relativePath)
            $manifestWriter.WriteLine("$relativePath`t$($file.Length)")
            $partBytes += $file.Length
            $partFiles++
            $totalBytes += $file.Length
            $totalFiles++
        }
    }
}
finally {
    if ($null -ne $partWriter) {
        $partWriter.Dispose()
    }
    $manifestWriter.Dispose()
}

$plan += [pscustomobject]@{
    Part = 'part-{0:D3}' -f $partNumber
    Files = $partFiles
    Bytes = $partBytes
    GiB = [math]::Round($partBytes / 1GB, 3)
}

$plan | ConvertTo-Csv -NoTypeInformation | Set-Content -LiteralPath (Join-Path $WorkDir 'part_plan.csv') -Encoding utf8
[pscustomobject]@{
    CreatedAt = (Get-Date).ToString('o')
    Repository = $Root
    Roots = $Roots
    Files = $totalFiles
    Bytes = $totalBytes
    GiB = [math]::Round($totalBytes / 1GB, 3)
    TargetPartGiB = $TargetGiB
    Parts = $plan.Count
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $WorkDir 'scope.json') -Encoding utf8

Get-Content -LiteralPath (Join-Path $WorkDir 'scope.json')
$plan | Format-Table -AutoSize
