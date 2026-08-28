# Restore sim-dir waveform wrapper after Vivado regenerates add_wave /.
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$src  = Join-Path $PSScriptRoot "tb_int8_dot_product_parallel.tcl"
$dst  = Join-Path $root "requantize_int8_rtl.sim\sim_1\behav\xsim\tb_int8_dot_product_parallel.tcl"

$wrapper = @'
# Wrapper kept in sim dir. Canonical logic lives in scripts/xsim/ (git).
set _wave_script [file normalize [file join [file dirname [info script]] .. .. .. .. scripts xsim tb_int8_dot_product_parallel.tcl]]
if { ![file exists $_wave_script] } {
  send_msg_id Add_Wave-1 ERROR "Missing waveform script: $_wave_script"
  run all
  return
}
source $_wave_script
'@

Set-Content -Path $dst -Value $wrapper -Encoding ASCII
Write-Host "Updated $dst"
