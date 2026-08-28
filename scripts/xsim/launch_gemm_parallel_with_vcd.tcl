# D3 并行 GEMM：GUI 加波 + 录 VCD + run all（一体脚本）
# 用法：Vivado Launch 仿真后，xsim Tcl Console：
#   restart
#   source F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/scripts/xsim/launch_gemm_parallel_with_vcd.tcl
# Sim Top：tb_gemm_parallel_python_vectors 或 tb_int8_gemm_parallel

set repo_root [file normalize [file join [file dirname [info script]] .. ..]]
set vcd_path [file join $repo_root gemm_parallel_full.vcd]

source [file normalize [file join [file dirname [info script]] gemm_parallel_wave_common.tcl]]

if { [llength [get_objects]] == 0 } {
  puts "ERROR: no simulation objects."
  puts "  Do not source this after the simulator has stopped (\$finish)."
  puts "  Fix: Vivado GUI -> Flow -> Run Simulation -> Launch Behavioral Simulation"
  puts "  (after running scripts/setup_gemm_parallel_sim.tcl once from cmd)"
  puts "  Or from cmd: scripts\\run_gemm_parallel_export_vcd.bat"
  return
}

set TB_TOP [resolve_gemm_tb_top ""]
if { $TB_TOP eq "" || [llength [get_objects -quiet /${TB_TOP}/clk]] == 0 } {
  puts "ERROR: cannot detect GEMM testbench top."
  puts "  Expected: tb_gemm_parallel_python_vectors or tb_int8_gemm_parallel"
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
  puts "WARN: no GUI waves added (paths missing?)"
} else {
  set_property needs_save false [current_wave_config]
  puts "INFO: GUI waveform ready, signals = $added"
}

catch { close_vcd }
puts "INFO: opening VCD -> $vcd_path"
open_vcd $vcd_path
set logged [log_gemm_parallel_vcd $TB_TOP]
if { $logged == 0 } {
  puts "ERROR: no signals logged to VCD."
  catch { close_vcd }
  return
}

run all

close_vcd
set vcd_size [file size $vcd_path]
puts "INFO: VCD export done -> $vcd_path"
puts "INFO: size=$vcd_size bytes"
if { $vcd_size < 4096 } {
  puts "ERROR: VCD suspiciously small ($vcd_size bytes)."
} else {
  puts "INFO: open VCD with GTKWave; GUI waves are in the current wave window."
}
