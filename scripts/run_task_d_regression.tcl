# Batch regression: generate vectors, then run xsim directly.
# Usage (from project root):
#   vivado -mode batch -source scripts/run_task_d_regression.tcl
#
# Note: if Vivado GUI simulation is open, close it first so simulate.log is not locked.

set proj_dir [file normalize [file join [file dirname [info script]] ..]]
set bat_file [file join $proj_dir scripts run_task_d_xsim.bat]

exec python [file join $proj_dir scripts generate_dot_product_parallel_vectors.py]

set rc [catch { exec cmd /c $bat_file } output]
puts $output
if { $rc != 0 } {
  error "Task D xsim regression failed"
}

puts "INFO: Task D batch regression finished. See simulate_task_d.log for details."
