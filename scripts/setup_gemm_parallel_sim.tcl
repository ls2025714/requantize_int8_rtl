# Switch Vivado sim_1 to D3 parallel GEMM.
# Sets custom_tcl = launch_gemm_parallel_with_vcd.tcl (GUI waves + VCD on Launch).
#
# Run from Windows CMD (NOT from xsim Tcl Console):
#   cd /d F:\Users\22563\Desktop\FPGAlearn\requantize_int8_rtl
#   vivado -mode batch -source scripts/setup_gemm_parallel_sim.tcl
#
# Then Vivado GUI: Run Simulation -> Launch Behavioral Simulation

set proj_dir [file normalize [file join [file dirname [info script]] ..]]
set proj_file [file join $proj_dir requantize_int8_rtl.xpr]
set wave_tcl  [file normalize [file join $proj_dir scripts xsim launch_gemm_parallel_with_vcd.tcl]]

open_project $proj_file

set fs [get_filesets sim_1]
set_property top tb_gemm_parallel_python_vectors $fs
set_property top_lib xil_defaultlib $fs
set_property xsim.simulate.custom_tcl $wave_tcl $fs
set_property xsim.simulate.runtime all $fs
catch { set_property xsim.view {} $fs }

puts "top         = [get_property top $fs]"
puts "custom_tcl  = [get_property xsim.simulate.custom_tcl $fs]"
puts "runtime     = [get_property xsim.simulate.runtime $fs]"

close_project
puts "INFO: D3 GEMM setup done."
puts "INFO: Launch Simulation -> auto waves + gemm_parallel_full.vcd"
puts "INFO: For inline TB: change top to tb_int8_gemm_parallel before Launch (same custom_tcl works)."
