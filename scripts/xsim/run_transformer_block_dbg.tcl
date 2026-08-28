# D12 debug: heartbeat via TRANSFORMER_BLOCK_DBG + VCD snapshot
set repo_root [file normalize [file join [file dirname [info script]] .. ..]]
set vcd_path [file join $repo_root transformer_block_dbg.vcd]

set tb_top tb_int8_transformer_block
set dut /${tb_top}/dut

if { [llength [get_objects -quiet ${dut}/clk]] == 0 } {
  puts "ERROR: dut hierarchy not found"
  exit 1
}

catch { close_vcd }
puts "INFO: opening VCD -> $vcd_path"
open_vcd $vcd_path

set sigs {
  st ls head_r idx_r seq_r
  sm_cmd_v sm_cmd_r sm_in_v sm_in_r sm_out_v sm_out_r
  sm_in_row sm_in_col
  u_soft/state u_soft/seq_reg u_soft/work_row u_soft/col_i
  u_score/state
}
foreach obj $sigs {
  set path ${dut}/${obj}
  if { [llength [get_objects -quiet $path]] > 0 } {
    log_vcd [get_objects $path]
  } else {
    puts "WARN: missing $path"
  }
}

run 50000000 ns

close_vcd
set sz [file size $vcd_path]
puts "INFO: VCD done size=$sz bytes -> $vcd_path"
puts "INFO: check simulate_transformer_block_dbg.log for DBG lines"
exit
