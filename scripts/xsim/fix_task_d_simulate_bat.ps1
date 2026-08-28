# Fix Vivado-regenerated simulate.bat pointing at wrong waveform Tcl (Task D).
$root = "F:\Users\22563\Desktop\FPGAlearn\requantize_int8_rtl"
$bat  = Join-Path $root "requantize_int8_rtl.sim\sim_1\behav\xsim\simulate.bat"
$good = "F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/scripts/xsim/tb_dot_product_parallel_python_vectors.tcl"
$bad  = "tb_int8_dot_product_parallel.tcl"

if (-not (Test-Path $bat)) {
    Write-Error "simulate.bat not found: $bat"
}

$content = Get-Content $bat -Raw
if ($content -match [regex]::Escape($bad)) {
    $content = $content -replace [regex]::Escape($bad), (Split-Path $good -Leaf)
    $content = $content -replace "tb_dot_product_parallel_python_vectors\.tcl", (Split-Path $good -Leaf)
    # Ensure full path
    $content = $content -replace [regex]::Escape("scripts/xsim/tb_dot_product_parallel_python_vectors.tcl"), $good.Replace('\','/')
    $content = $content -replace [regex]::Escape("scripts/xsim/tb_int8_dot_product_parallel.tcl"), $good.Replace('\','/')
    Set-Content -Path $bat -Value $content -Encoding ASCII
    Write-Host "Patched simulate.bat -> Task D Tcl"
} else {
    Write-Host "simulate.bat already uses Task D Tcl (or unknown format)"
}

Write-Host "Expected tclbatch: $good"
