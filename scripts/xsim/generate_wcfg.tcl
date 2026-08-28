# Generate tb_int8_dot_product_parallel_behav.wcfg via Vivado (needs GUI wave APIs).
# Usage: vivado -mode batch -source scripts/xsim/generate_wcfg.tcl

set proj [file normalize [file join [file dirname [info script]] .. .. requantize_int8_rtl.xpr]]
open_project $proj
set sim_dir [file normalize [file join [file dirname $proj] requantize_int8_rtl.sim sim_1 behav xsim]]
set wcfg [file normalize [file join [file dirname [info script]] tb_int8_dot_product_parallel_behav.wcfg]]

close_sim -quiet
launch_simulation

set apply [file normalize [file join [file dirname [info script]] apply_waveform_only.tcl]]
source $apply

if { ![file exists $wcfg] } {
  puts "ERROR: failed to write $wcfg"
  exit 1
}

puts "INFO: wrote $wcfg"
close_sim -quiet
close_project
