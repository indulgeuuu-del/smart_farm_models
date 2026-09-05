[CmdletBinding()]
param(
    [string]$Root = (Get-Location).Path,
    [string]$WorkDir = '',
    [double]$TargetGiB = 1.55
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path -LiteralPath $Root).Path.TrimEnd('\')
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

if ($TargetGiB -le 0) {
    throw 'TargetGiB must be greater than zero.'
}

if ([string]::IsNullOrWhiteSpace($WorkDir)) {
    $WorkDir = Join-Path ([System.IO.Path]::GetTempPath()) 'smart_farm_models_public_release'
}
$WorkDir = [System.IO.Path]::GetFullPath($WorkDir).TrimEnd('\')
if ($WorkDir.Equals($Root, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'WorkDir must not be the repository root.'
}

$inputRoots = @(
    foreach ($relativeRoot in $Roots) {
        $absoluteRoot = [System.IO.Path]::GetFullPath((Join-Path $Root $relativeRoot)).TrimEnd('\')
        if (Test-Path -LiteralPath $absoluteRoot) {
            $absoluteRoot
        }
    }
)
foreach ($inputRoot in $inputRoots) {
    $inputPrefix = $inputRoot + [System.IO.Path]::DirectorySeparatorChar
    if (
        $WorkDir.Equals($inputRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
        $WorkDir.StartsWith($inputPrefix, [System.StringComparison]::OrdinalIgnoreCase)
    ) {
        throw "WorkDir must not be inside an input root because rebuilding would delete source data: $inputRoot"
    }
}

if (Test-Path -LiteralPath $WorkDir) {
    $marker = Join-Path $WorkDir '.public_release_workdir'
    if (-not (Test-Path -LiteralPath $marker -PathType Leaf)) {
        throw "WorkDir already exists and is not a release-builder directory. Choose another path or remove it explicitly: $WorkDir"
    }
    Remove-Item -LiteralPath $WorkDir -Recurse -Force
}

$PartsDir = Join-Path $WorkDir 'parts'
$ArchivesDir = $WorkDir
New-Item -ItemType Directory -Force -Path $PartsDir | Out-Null
Set-Content -LiteralPath (Join-Path $WorkDir '.public_release_workdir') -Value 'Created by build_public_data_release.ps1.' -Encoding ascii

$tar = Get-Command tar.exe -ErrorAction SilentlyContinue
if ($null -eq $tar) {
    throw 'tar.exe is required. Windows 10/11 normally provides it; install bsdtar or use a supported Windows environment.'
}

function Test-ExcludedArchivePath {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    $normalized = $RelativePath.Replace('\', '/')
    if ($normalized -match '(^|/)logs/(?:.*/)?[^/]*progress[^/]*\.md$') {
        return $true
    }
    return $false
}

function Write-Utf8NoBomLines {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][System.Collections.IEnumerable]$Lines
    )

    $encoding = New-Object System.Text.UTF8Encoding($false)
    $writer = New-Object System.IO.StreamWriter($Path, $false, $encoding)
    try {
        foreach ($line in $Lines) {
            $writer.WriteLine([string]$line)
        }
    }
    finally {
        $writer.Dispose()
    }
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

$entriesByPath = @{}
foreach ($relativeRoot in $Roots) {
    $absoluteRoot = Join-Path $Root $relativeRoot
    if (-not (Test-Path -LiteralPath $absoluteRoot -PathType Container)) {
        continue
    }

    foreach ($file in Get-ChildItem -LiteralPath $absoluteRoot -File -Recurse -Force) {
        if ($file.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
            throw "Refusing to archive a reparse point: $($file.FullName)"
        }
        $relativePath = $file.FullName.Substring($Root.Length + 1).Replace('\', '/')
        if ($relativePath.Contains("`r") -or $relativePath.Contains("`n")) {
            throw "A source filename contains a newline and cannot be represented safely: $relativePath"
        }
        if (Test-ExcludedArchivePath -RelativePath $relativePath) {
            continue
        }
        if ($entriesByPath.ContainsKey($relativePath)) {
            throw "The same relative path was discovered more than once: $relativePath"
        }
        $entriesByPath[$relativePath] = [pscustomobject]@{
            Path = $relativePath
            File = $file
            Bytes = [int64]$file.Length
        }
    }
}

$entries = @($entriesByPath.Values | Sort-Object -Property Path)
if ($entries.Count -eq 0) {
    throw 'No public-release source files were found.'
}

$targetBytes = [int64][math]::Floor($TargetGiB * 1GB)
if ($targetBytes -lt 1) {
    throw 'TargetGiB is too small after conversion to bytes.'
}

$parts = New-Object 'System.Collections.Generic.List[object]'
$currentPart = New-Object 'System.Collections.Generic.List[object]'
$currentBytes = [int64]0
foreach ($entry in $entries) {
    if ($currentPart.Count -gt 0 -and ($currentBytes + $entry.Bytes) -gt $targetBytes) {
        $parts.Add($currentPart.ToArray())
        $currentPart = New-Object 'System.Collections.Generic.List[object]'
        $currentBytes = [int64]0
    }
    $currentPart.Add($entry)
    $currentBytes += $entry.Bytes
}
if ($currentPart.Count -gt 0) {
    $parts.Add($currentPart.ToArray())
}

$utf8 = New-Object System.Text.UTF8Encoding($false)
$manifestPath = Join-Path $WorkDir 'source_manifest.tsv'
$manifestWriter = New-Object System.IO.StreamWriter($manifestPath, $false, $utf8)
try {
    $manifestWriter.WriteLine("path`tbytes")
    foreach ($entry in $entries) {
        $manifestWriter.WriteLine("$($entry.Path)`t$($entry.Bytes)")
    }
}
finally {
    $manifestWriter.Dispose()
}

$plan = New-Object 'System.Collections.Generic.List[object]'
for ($index = 0; $index -lt $parts.Count; $index++) {
    $partName = 'part-{0:D3}' -f ($index + 1)
    $listPath = Join-Path $PartsDir ($partName + '.txt')
    $archivePath = Join-Path $ArchivesDir ($partName + '.tar.gz')
    $partEntries = $parts[$index]
    Write-Utf8NoBomLines -Path $listPath -Lines ($partEntries | ForEach-Object { $_.Path })

    & $tar.Source -czf $archivePath -C $Root -T $listPath
    if ($LASTEXITCODE -ne 0) {
        throw "tar.exe failed while creating $archivePath (exit code $LASTEXITCODE)."
    }

    $partBytes = [int64](($partEntries | Measure-Object -Property Bytes -Sum).Sum)
    $archiveFile = Get-Item -LiteralPath $archivePath
    $plan.Add([pscustomobject]@{
        Part = $partName
        Files = $partEntries.Count
        Bytes = $partBytes
        GiB = [math]::Round($partBytes / 1GB, 3)
        Archive = $partName + '.tar.gz'
        ArchiveBytes = [int64]$archiveFile.Length
        ArchiveSHA256 = Get-Sha256 -Path $archivePath
    })
}

$plan | ConvertTo-Csv -NoTypeInformation | Set-Content -LiteralPath (Join-Path $WorkDir 'part_plan.csv') -Encoding utf8
$sumLines = @($plan | ForEach-Object { "$($_.ArchiveSHA256) *$($_.Archive)" })
Write-Utf8NoBomLines -Path (Join-Path $WorkDir 'SHA256SUMS.txt') -Lines $sumLines

$scope = [ordered]@{
    SchemaVersion = 1
    CreatedAtUtc = [DateTime]::UtcNow.ToString('o')
    Repository = 'smart_farm_models'
    Roots = $Roots
    ExcludedRelativePathPatterns = @(
        'logs/**/*progress*.md'
    )
    Files = $entries.Count
    Bytes = [int64](($entries | Measure-Object -Property Bytes -Sum).Sum)
    GiB = [math]::Round((($entries | Measure-Object -Property Bytes -Sum).Sum) / 1GB, 3)
    TargetPartGiB = $TargetGiB
    Parts = $plan.Count
    Archives = @($plan | ForEach-Object { $_.Archive })
}
$scope | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $WorkDir 'scope.json') -Encoding utf8

$notes = @(
    '# smart_farm_models public data archive'
    ''
    'This directory was generated by scripts/release/build_public_data_release.ps1.'
    'The source manifest uses repository-relative paths and byte sizes only.'
    ''
    '## Contents'
    ''
    ('- {0} tar.gz archives, split at approximately {1} GiB of source bytes.' -f $plan.Count, $TargetGiB)
    '- source_manifest.tsv: source-relative file paths and byte sizes.'
    '- part_plan.csv: per-part source and archive statistics.'
    '- scope.json: non-machine-specific scope and exclusion metadata.'
    '- SHA256SUMS.txt: SHA256 values for every tar.gz archive.'
    ''
    '## Exclusions'
    ''
    '- Markdown progress logs below logs/ whose names contain progress.'
    ''
    '## Restore'
    ''
    'Download every tar.gz archive and verify SHA256SUMS.txt before extraction.'
    'Extract all archives into an empty repository root, then compare file paths and sizes with source_manifest.tsv.'
)
Write-Utf8NoBomLines -Path (Join-Path $WorkDir 'RELEASE_NOTES.md') -Lines $notes

Get-Content -LiteralPath (Join-Path $WorkDir 'scope.json')
$plan | Format-Table -AutoSize
Write-Output "OUTPUT_DIR=$WorkDir"
