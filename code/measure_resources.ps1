param(
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,
    [string]$Output,
    [string]$Python,
    [int]$Threads = 4,
    [string[]]$AssetPath = @()
)

$codeRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

function Resolve-CodePath([string]$PathText) {
    if ([IO.Path]::IsPathRooted($PathText)) {
        return [IO.Path]::GetFullPath($PathText)
    }
    return [IO.Path]::GetFullPath((Join-Path $codeRoot $PathText))
}

$checkpointPath = Resolve-CodePath $Checkpoint
if (-not (Test-Path -LiteralPath $checkpointPath -PathType Leaf)) {
    throw "Checkpoint not found: $checkpointPath"
}

if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path (Split-Path -Parent $checkpointPath) 'test_cpu_fp32_measured.json'
} else {
    $Output = Resolve-CodePath $Output
}

if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = (Get-Command python -ErrorAction Stop).Source
}

$stdoutPath = Join-Path ([IO.Path]::GetTempPath()) ("mp1-evaluate-{0}.out" -f [guid]::NewGuid())
$stderrPath = Join-Path ([IO.Path]::GetTempPath()) ("mp1-evaluate-{0}.err" -f [guid]::NewGuid())
$arguments = @(
    'evaluate.py',
    '--checkpoint', $checkpointPath,
    '--device', 'cpu',
    '--precision', 'fp32',
    '--threads', [string]$Threads,
    '--split', 'test',
    '--output', $Output
)

$watch = [Diagnostics.Stopwatch]::StartNew()
$process = Start-Process -FilePath $Python -WorkingDirectory $codeRoot `
    -ArgumentList $arguments -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath -PassThru -WindowStyle Hidden

[long]$peakWorkingSet = 0
[long]$peakPrivateBytes = 0
[long]$samples = 0

try {
    while (-not $process.HasExited) {
        try {
            $process.Refresh()
            $peakWorkingSet = [Math]::Max($peakWorkingSet, [long]$process.WorkingSet64)
            $peakPrivateBytes = [Math]::Max($peakPrivateBytes, [long]$process.PrivateMemorySize64)
            $samples++
        } catch {
            # The process may exit between HasExited and Refresh.
        }
        Start-Sleep -Milliseconds 50
    }
    $process.WaitForExit()
    try {
        $process.Refresh()
        $peakWorkingSet = [Math]::Max($peakWorkingSet, [long]$process.PeakWorkingSet64)
        $peakPrivateBytes = [Math]::Max($peakPrivateBytes, [long]$process.PrivateMemorySize64)
    } catch {
        # The last sampled values remain usable for a very short process.
    }
} finally {
    $watch.Stop()
}

$stdout = if (Test-Path -LiteralPath $stdoutPath) { Get-Content -LiteralPath $stdoutPath -Raw } else { '' }
$stderr = if (Test-Path -LiteralPath $stderrPath) { Get-Content -LiteralPath $stderrPath -Raw } else { '' }
Remove-Item -LiteralPath $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue

if ($process.ExitCode -ne 0) {
    throw "evaluate.py failed with exit code $($process.ExitCode).`n$stderr"
}

$assetFiles = @($checkpointPath)
foreach ($assetPath in $AssetPath) {
    $resolvedAsset = Resolve-CodePath $assetPath
    if (-not (Test-Path -LiteralPath $resolvedAsset -PathType Leaf)) {
        throw "Asset not found: $resolvedAsset"
    }
    $assetFiles += $resolvedAsset
}

$assetRows = foreach ($assetFile in ($assetFiles | Select-Object -Unique)) {
    $item = Get-Item -LiteralPath $assetFile
    [PSCustomObject]@{
        path = $item.FullName
        bytes = [long]$item.Length
        MiB = [Math]::Round($item.Length / 1MB, 4)
    }
}
$assetBytes = [long](($assetRows | Measure-Object -Property bytes -Sum).Sum)
$gib = 1024L * 1024L * 1024L
$mib = 1024L * 1024L
$ramLimit = 4L * $gib
$assetLimit = 64L * $mib

[PSCustomObject]@{
    exit_code = $process.ExitCode
    elapsed_seconds = [Math]::Round($watch.Elapsed.TotalSeconds, 3)
    samples = $samples
    peak_working_set_bytes = $peakWorkingSet
    peak_working_set_gib = [Math]::Round($peakWorkingSet / $gib, 4)
    peak_private_sampled_bytes = $peakPrivateBytes
    peak_private_sampled_gib = [Math]::Round($peakPrivateBytes / $gib, 4)
    ram_limit_bytes = $ramLimit
    ram_within_limit = ($peakWorkingSet -le $ramLimit)
    asset_bytes = $assetBytes
    asset_mib = [Math]::Round($assetBytes / $mib, 4)
    asset_limit_bytes = $assetLimit
    assets_within_limit = ($assetBytes -le $assetLimit)
    assets = @($assetRows)
    evaluator_stdout = $stdout.Trim()
} | ConvertTo-Json -Depth 5
