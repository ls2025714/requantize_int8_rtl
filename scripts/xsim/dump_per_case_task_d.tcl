# Per-case result dump: one line per PASS/FAIL at m_valid (better than fixed-time snapshots).
#   source F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/scripts/xsim/dump_per_case_task_d.tcl
#
# Writes task_d_case_results.txt at repo root.

set TB /tb_dot_product_parallel_python_vectors
set repo_root [file normalize [file join [file dirname [info script]] .. ..]]
set out_path [file join $repo_root task_d_case_results.txt]

if { [llength [get_objects]] == 0 } {
  puts "INFO: sim finished; restarting..."
  catch { restart }
}
if { [llength [get_objects]] == 0 } {
  puts "ERROR: launch simulation first."
  return
}

set fp [open $out_path w]
puts $fp "# Task D per-case dump at each m_valid rising edge"
puts $fp "# columns: time_ns test_count m_result mismatch_count state"
puts $fp ""

set last_m_valid 0
set last_test_count -1
set t_ns 0

restart
while { $t_ns <= 500000 } {
  run 10ns
  set t_ns [expr {$t_ns + 10}]

  if { [catch { set m_valid [get_value ${TB}/m_valid] } err] } {
    break
  }
  set test_count [get_value ${TB}/test_count]
  set m_result [get_value ${TB}/m_result]
  set mismatch [get_value ${TB}/mismatch_count]
  set state [get_value ${TB}/dut/state]

  if { ($m_valid == 1) && ($last_m_valid == 0) } {
    puts $fp "$t_ns $test_count $m_result $mismatch $state"
    flush $fp
  }
  set last_m_valid $m_valid

  if { ($test_count == 120) && ($m_valid == 0) && ($last_test_count == 120) } {
    break
  }
  set last_test_count $test_count
}

close $fp
puts "INFO: wrote $out_path"
