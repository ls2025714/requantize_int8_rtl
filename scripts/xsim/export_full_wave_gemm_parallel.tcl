# Re-export VCD only (batch / re-run). Prefer launch_gemm_parallel_with_vcd.tcl for GUI.
# Usage: scripts\run_gemm_parallel_export_vcd.bat

set repo_root [file normalize [file join [file dirname [info script]] .. ..]]
set vcd_path [file join $repo_root gemm_parallel_full.vcd]

source [file normalize [file join [file dirname [info script]] gemm_parallel_wave_common.tcl]]

if { [llength [get_objects]] == 0 } {
  puts "INFO: no live objects; trying restart..."
  catch { restart }
}

if { [llength [get_objects]] == 0 } {
  puts "ERROR: simulator not active."
  puts "  Run: scripts\\run_gemm_parallel_export_vcd.bat"
  return
}

set TB_TOP [resolve_gemm_tb_top tb_gemm_parallel_python_vectors]
if { $TB_TOP eq "" || [llength [get_objects -quiet /${TB_TOP}/clk]] == 0 } {
  puts "ERROR: cannot detect GEMM testbench top."
  return
}
puts "INFO: using TB_TOP = $TB_TOP"

catch { close_vcd }
puts "INFO: opening VCD -> $vcd_path"
open_vcd $vcd_path
set logged [log_gemm_parallel_vcd $TB_TOP]
if { $logged == 0 } {
  puts "ERROR: no signals logged to VCD."
  catch { close_vcd }
  return
}

restart
run all

close_vcd
set vcd_size [file size $vcd_path]
puts "INFO: VCD export done -> $vcd_path"
puts "INFO: size=$vcd_size bytes"
if { $vcd_size < 4096 } {
  puts "ERROR: VCD suspiciously small ($vcd_size bytes)."
} else {
  puts "INFO: open with GTKWave."
}
