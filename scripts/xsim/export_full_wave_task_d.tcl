# Export full time-range VCD for Task D (every value change, not snapshots).
# Must run BEFORE or DURING simulation (WDB cannot be converted to VCD after the fact).
#
# GUI (sim already open):
#   close_sim
#   launch_simulation
#   source F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/scripts/xsim/export_full_wave_task_d.tcl
#
# Batch (from repo root, after compile/elab):
#   xsim tb_dot_product_parallel_python_vectors_behav -tclbatch scripts/xsim/export_full_wave_task_d.tcl

set TB_TOP tb_dot_product_parallel_python_vectors
set repo_root [file normalize [file join [file dirname [info script]] .. ..]]
set vcd_path [file join $repo_root task_d_full.vcd]

proc log_task_d_vcd {tb_top} {
  log_vcd /${tb_top}/clk
  log_vcd /${tb_top}/rst_n
  log_vcd /${tb_top}/cmd_valid
  log_vcd /${tb_top}/cmd_ready
  log_vcd /${tb_top}/cmd_length
  log_vcd /${tb_top}/s_valid
  log_vcd /${tb_top}/s_ready
  log_vcd /${tb_top}/s_a0
  log_vcd /${tb_top}/s_a1
  log_vcd /${tb_top}/s_a2
  log_vcd /${tb_top}/s_a3
  log_vcd /${tb_top}/s_b0
  log_vcd /${tb_top}/s_b1
  log_vcd /${tb_top}/s_b2
  log_vcd /${tb_top}/s_b3
  log_vcd /${tb_top}/s_keep
  log_vcd /${tb_top}/acc_clear
  log_vcd /${tb_top}/acc_enable
  log_vcd /${tb_top}/dut/state
  log_vcd /${tb_top}/dut/valid_s1
  log_vcd /${tb_top}/dut/valid_s2
  log_vcd /${tb_top}/dut/valid_s3
  log_vcd /${tb_top}/dut/valid_s4
  log_vcd /${tb_top}/dut/partial_valid
  log_vcd /${tb_top}/dut/partial_sum
  log_vcd /${tb_top}/dut/acc_valid
  log_vcd /${tb_top}/dut/acc_value
  log_vcd /${tb_top}/m_valid
  log_vcd /${tb_top}/m_ready
  log_vcd /${tb_top}/m_result
  log_vcd /${tb_top}/test_count
  log_vcd /${tb_top}/mismatch_count
}

if { [llength [get_objects]] == 0 } {
  puts "INFO: no live objects (sim finished or not launched); trying restart..."
  catch { restart }
}

if { [llength [get_objects]] == 0 } {
  puts "ERROR: still no objects."
  puts "  Fix: Flow -> Run Simulation (Task D top), then re-source this script."
  puts "  Or run from repo root: scripts\\run_task_d_export_vcd.bat"
  return
}

catch { close_vcd }

puts "INFO: opening VCD -> $vcd_path"
open_vcd $vcd_path
log_task_d_vcd $TB_TOP

restart
run all

close_vcd
puts "INFO: VCD export done -> $vcd_path"
puts "INFO: size=[file size $vcd_path] bytes"
puts "INFO: open with GTKWave, or share the file for review."
