# Curated waveform for D3 parallel GEMM.
# Auto-detects tb_gemm_parallel_python_vectors or tb_int8_gemm_parallel.
#
# Usage (sim must be running):
#   source F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/scripts/xsim/apply_waveform_gemm_parallel.tcl

if { ![info exists TB_TOP] } {
  set TB_TOP ""
}

source [file normalize [file join [file dirname [info script]] gemm_parallel_wave_common.tcl]]

if { [llength [get_objects]] == 0 } {
  puts "ERROR: no simulation objects. Launch simulation first, or type: restart"
  return
}

set TB_TOP [resolve_gemm_tb_top $TB_TOP]
if { $TB_TOP eq "" || [llength [get_objects -quiet /${TB_TOP}/clk]] == 0 } {
  puts "ERROR: cannot detect GEMM testbench top."
  return
}
puts "INFO: using TB_TOP = $TB_TOP"

foreach wc [get_wave_configs] {
  catch { close_wave_config -force $wc }
}
create_wave_config gemm_parallel_d3

set old_waves [get_waves *]
if { [llength $old_waves] > 0 } {
  delete_wave $old_waves
}

set added [add_gemm_parallel_waves $TB_TOP]
if { $added == 0 } {
  puts "ERROR: no waves added."
  return
}

set_property needs_save false [current_wave_config]

set wcfg_path [file normalize [file join [file dirname [info script]] tb_gemm_parallel_python_vectors_behav.wcfg]]
catch { save_wave_config $wcfg_path }

puts "INFO: GEMM parallel waveform ready, signals = [llength [get_waves *]]"
puts "INFO: saved wcfg -> $wcfg_path"
