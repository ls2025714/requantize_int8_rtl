# One-time Vivado project setup for tb_int8_dot_product_parallel simulation.
# Usage (from project root):
#   vivado -mode batch -source scripts/setup_vivado_sim.tcl

set proj_dir [file normalize [file join [file dirname [info script]] ..]]
set proj_file [file join $proj_dir requantize_int8_rtl.xpr]
set wave_tcl  [file normalize [file join $proj_dir scripts xsim tb_int8_dot_product_parallel.tcl]]

open_project $proj_file

set fs [get_filesets sim_1]

set_property xsim.simulate.custom_tcl $wave_tcl $fs
set_property xsim.simulate.runtime all $fs
catch { set_property xsim.view {} $fs }

foreach wcfg_file [get_files -of_objects $fs] {
  if { [string match *.wcfg $wcfg_file] } {
    remove_files -fileset sim_1 $wcfg_file
  }
}

puts "custom_tcl = [get_property xsim.simulate.custom_tcl $fs]"
puts "runtime    = [get_property xsim.simulate.runtime $fs]"

close_project

puts "INFO: Properties applied. Re-open project in Vivado GUI to persist if needed."
