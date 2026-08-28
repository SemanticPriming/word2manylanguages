# FileChunker.ps1 — Split / Recombine large files (e.g., .bz2) with progress & verification
# Run from a PowerShell window:
#   powershell -NoExit -ExecutionPolicy Bypass -File .\FileChunker.ps1

$ErrorActionPreference = 'Stop'
$script:FC_ShouldPause = $true  # controls the final pause; Exit sets this to $false

function Parse-SizeToBytes([string]$text) {
    if ([string]::IsNullOrWhiteSpace($text)) { return 1GB }
    $m = [regex]::Match($text.Trim(), '^\s*(\d+(?:\.\d+)?)\s*([KMGTP]?B?)\s*$', 'IgnoreCase')
    if (-not $m.Success) { throw "Invalid size: '$text' (try 1GB, 500MB, 100M, 1024KB)" }
    $num = [double]$m.Groups[1].Value
    $unit = $m.Groups[2].Value.ToUpper()
    switch ($unit) {
        ''   { return [long]$num }
        'B'  { return [long]$num }
        'KB' { return [long]($num * 1KB) }
        'MB' { return [long]($num * 1MB) }
        'GB' { return [long]($num * 1GB) }
        'TB' { return [long]($num * 1TB) }
        'K'  { return [long]($num * 1KB) }
        'M'  { return [long]($num * 1MB) }
        'G'  { return [long]($num * 1GB) }
        'T'  { return [long]($num * 1TB) }
        default { throw "Unknown unit '$unit'" }
    }
}

function Start-MyTranscript($contextPath) {
    try {
        $dir = if ($contextPath) { Split-Path $contextPath -Parent } else { $PWD.Path }
        if (-not (Test-Path $dir)) { $dir = $PWD.Path }
        $ts = Join-Path $dir ("FileChunkerLog_{0:yyyyMMdd_HHmmss}.txt" -f (Get-Date))
        Start-Transcript -Path $ts -ErrorAction SilentlyContinue | Out-Null
        Write-Host "Logging to: $ts"
    } catch {}
}

function Split-File {
    param(
        [Parameter(Mandatory=$true)][string]$InputPath,
        [Parameter(Mandatory=$true)][long]$ChunkBytes
    )
    if (-not (Test-Path $InputPath)) { throw "File not found: $InputPath" }
    $dir  = Split-Path $InputPath
    $base = Split-Path $InputPath -Leaf
    $prefix = Join-Path $dir "$base.part_"

    Write-Host "Splitting:`n  File: $InputPath`n  Part size: $ChunkBytes bytes`n  Output prefix: $prefix"
    $buffer = New-Object byte[] (4MB)
    $part = 0
    $writtenTotal = 0

    $fs = [System.IO.File]::OpenRead($InputPath)
    try {
        $total = $fs.Length
        while ($fs.Position -lt $total) {
            $outPath = "{0}{1:D3}" -f $prefix, $part
            $ofs = [System.IO.File]::Create($outPath)
            $writtenPart = 0
            try {
                while ($writtenPart -lt $ChunkBytes) {
                    $toRead = [int][math]::Min($buffer.Length, $ChunkBytes - $writtenPart)
                    $n = $fs.Read($buffer, 0, $toRead)
                    if ($n -le 0) { break }
                    $ofs.Write($buffer, 0, $n)
                    $writtenPart += $n
                    $writtenTotal += $n
                    $pct = if ($total -gt 0) { [int](($writtenTotal / $total) * 100) } else { 0 }
                    Write-Progress -Activity "Splitting $base" -Status "Part $part ($writtenPart / $ChunkBytes bytes)" -PercentComplete $pct
                }
            } finally {
                $ofs.Close()
            }
            Write-Host "Created $outPath ($writtenPart bytes)"
            $part++
        }
        Write-Host "✅ Done. Created $part part(s) in $dir"
    } finally {
        $fs.Close()
    }
}

function Recombine-Files {
    param(
        [Parameter(Mandatory=$true)][string]$PartPrefix,
        [Parameter(Mandatory=$true)][string]$OutputPath
    )
    $parts = Get-ChildItem ($PartPrefix + "*") | Sort-Object Name
    if (-not $parts -or $parts.Count -eq 0) { throw "No parts found with prefix: $PartPrefix" }

    Write-Host "Recombining:"
    Write-Host "  Prefix: $PartPrefix"
    Write-Host "  Output: $OutputPath"
    $ofs = [System.IO.File]::Create($OutputPath)
    try {
        $idx = 0
        foreach ($p in $parts) {
            Write-Host ("  -> {0}" -f $p.FullName)
            $bytes = [System.IO.File]::ReadAllBytes($p.FullName)
            $ofs.Write($bytes, 0, $bytes.Length)
            $idx++
            Write-Progress -Activity "Recombining" -Status "Part $idx of $($parts.Count)" -PercentComplete ([int](($idx/$($parts.Count))*100))
        }
        Write-Host "✅ Recombined into $OutputPath"
    } finally {
        $ofs.Close()
    }
}

function Hash-File([string]$Path) {
    if (-not (Test-Path $Path)) { throw "File not found: $Path" }
    (Get-FileHash -Algorithm SHA256 -Path $Path).Hash
}

# --------- Main ---------
Start-MyTranscript $null

:MainLoop while ($true) {
    Write-Host ""
    Write-Host "========== FileChunker =========="
    Write-Host "1) Split a file into parts"
    Write-Host "2) Recombine parts into a file"
    Write-Host "3) Verify two files match (SHA-256)"
    Write-Host "4) Exit"
    $choice = Read-Host "Select an option (1-4)"

    try {
        switch ($choice) {
            '1' {
                $in = Read-Host "Enter full path to file to split (e.g., C:\data\bigfile.bz2)"
                $sizeInput = Read-Host "Chunk size? (e.g., 1GB, 500MB) [default: 1GB]"
                if ([string]::IsNullOrWhiteSpace($sizeInput)) { $sizeInput = '1GB' }
                $chunk = Parse-SizeToBytes $sizeInput
                Start-MyTranscript $in
                Split-File -InputPath $in -ChunkBytes $chunk
                Read-Host "Press Enter to return to menu"
            }
            '2' {
                $prefix = Read-Host "Enter path+prefix (e.g., C:\data\bigfile.bz2.part_)"
                $out = Read-Host "Enter output file (e.g., C:\data\bigfile_recombined.bz2)"
                Start-MyTranscript $out
                Recombine-Files -PartPrefix $prefix -OutputPath $out
                Read-Host "Press Enter to return to menu"
            }
            '3' {
                $f1 = Read-Host "Enter first file path"
                $f2 = Read-Host "Enter second file path"
                $h1 = Hash-File $f1
                $h2 = Hash-File $f2
                Write-Host "SHA-256:"
                Write-Host "  $f1 -> $h1"
                Write-Host "  $f2 -> $h2"
                if ($h1 -eq $h2) { Write-Host "✅ MATCH" -ForegroundColor Green } else { Write-Host "❌ DIFFER" -ForegroundColor Red }
                Read-Host "Press Enter to return to menu"
            }
            '4' {
                Write-Host "Goodbye!"
                try { Stop-Transcript | Out-Null } catch {}
                $script:FC_ShouldPause = $false
                break MainLoop
            }
            default {
                Write-Host "Please choose 1, 2, 3, or 4."
            }
        }
    }
    catch {
        Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
        Read-Host "Press Enter to return to menu"
    }
}
# Finalizer — only pause if not exiting via option 4
try { Stop-Transcript | Out-Null } catch {}
if ($script:FC_ShouldPause) {
    Write-Host ""
    Read-Host "Press Enter to close"
}
