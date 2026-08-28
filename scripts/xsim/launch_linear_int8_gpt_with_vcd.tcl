# D5 GPT Linear 层：GUI 加波 + 录 VCD + run all（一体脚本，无 bat）
# 用法：Vivado Launch 仿真后，xsim Tcl Console：
#   restart
#   source F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/scripts/xsim/launch_linear_int8_gpt_with_vcd.tcl
# Sim Top 应为 tb_linear_int8_gpt_vectors

set repo_root [file normalize [file join [file dirname [info script]] .. ..]]
set vcd_path [file join $repo_root gpt_linear_int8_full.vcd]

source [file normalize [file join [file dirname [info script]] gpt_linear_int8_wave_common.tcl]]

if { [llength [get_objects]] == 0 } {
  puts "ERROR: no simulation objects."
  puts "  Do not source this after the simulator has stopped (\$finish)."
  return
}

set TB_TOP [resolve_gpt_linear_tb_top ""]
if { $TB_TOP eq "" || [llength [get_objects -quiet /${TB_TOP}/clk]] == 0 } {
  puts "ERROR: cannot detect GPT linear INT8 testbench top."
  puts "  Expected: tb_linear_int8_gpt_vectors"
  return
}
puts "INFO: using TB_TOP = $TB_TOP"

foreach wc [get_wave_configs] {
  catch { close_wave_config -force $wc }
}
create_wave_config gpt_linear_int8_d5

set old_waves [get_waves *]
if { [llength $old_waves] > 0 } {
  delete_wave $old_waves
}

set added [add_gpt_linear_int8_waves $TB_TOP]
if { $added == 0 } {
  puts "WARN: no GUI waves added (paths missing?)"
} else {
  set_property needs_save false [current_wave_config]
  puts "INFO: GUI waveform ready, signals = $added"
}

catch { close_vcd }
puts "INFO: opening VCD -> $vcd_path"
open_vcd $vcd_path
set logged [log_gpt_linear_int8_vcd $TB_TOP]
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
