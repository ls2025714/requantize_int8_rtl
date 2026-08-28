# Task D: export full VCD from repo root (PowerShell friendly).
# Usage:  .\scripts\run_task_d_export_vcd.ps1

$ErrorActionPreference = "Stop"
$repo = "F:\Users\22563\Desktop\FPGAlearn\requantize_int8_rtl"
$bat  = Join-Path $repo "scripts\run_task_d_export_vcd.bat"

Write-Host "=== Task D VCD export (via cmd) ==="
cmd /c "`"$bat`""
if ($LASTEXITCODE -ne 0) {
    Write-Error "Export failed with exit code $LASTEXITCODE"
}
$vcd = Join-Path $repo "task_d_full.vcd"
if (Test-Path $vcd) {
    $f = Get-Item $vcd
    Write-Host "OK: $($f.FullName) ($($f.Length) bytes)"
}
