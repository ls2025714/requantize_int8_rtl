# D12 Block：Vivado GUI 加精选波形 + run all
# 用法（必须先 Launch Simulation，且 Sim Top = tb_int8_transformer_block）：
#   restart
#   source F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/scripts/xsim/launch_transformer_block_waves.tcl
#
# 注意：完整 E2E ~2.7ms 仿真时间，GUI 可能要跑一两分钟，不要中途关窗口。

set TB_TOP tb_int8_transformer_block
set DUT /${TB_TOP}/dut

if { [llength [get_objects -quiet /${TB_TOP}/clk]] == 0 } {
  puts "ERROR: 找不到 /${TB_TOP}/clk"
  puts "  请先：Simulation Sources 右键 tb_int8_transformer_block -> Set as Top"
  puts "  再：Flow Navigator -> Run Simulation -> Run Behavioral Simulation"
  puts "  然后在 Tcl Console 里 source 本脚本（若已 \$finish，先输入 restart）"
  return
}

foreach wc [get_wave_configs] {
  catch { close_wave_config -force $wc }
}
create_wave_config transformer_block_d12

set old_waves [get_waves *]
if { [llength $old_waves] > 0 } {
  delete_wave $old_waves
}

set wave_list {
  /tb_int8_transformer_block/clk
  /tb_int8_transformer_block/rst_n
  /tb_int8_transformer_block/cmd_valid
  /tb_int8_transformer_block/cmd_ready
  /tb_int8_transformer_block/out_valid
  /tb_int8_transformer_block/out_ready
  /tb_int8_transformer_block/out_data
  /tb_int8_transformer_block/out_last
  /tb_int8_transformer_block/dut/st
  /tb_int8_transformer_block/dut/ls
  /tb_int8_transformer_block/dut/head_r
  /tb_int8_transformer_block/dut/seq_r
  /tb_int8_transformer_block/dut/idx_r
  /tb_int8_transformer_block/dut/preload_done
  /tb_int8_transformer_block/dut/l_cmd_v
  /tb_int8_transformer_block/dut/l_cmd_r
  /tb_int8_transformer_block/dut/sm_cmd_v
  /tb_int8_transformer_block/dut/sm_out_v
  /tb_int8_transformer_block/dut/act_score
  /tb_int8_transformer_block/dut/act_soft
  /tb_int8_transformer_block/dut/act_attn
  /tb_int8_transformer_block/dut/act_silu
  /tb_int8_transformer_block/dut/act_emul
}

set added 0
foreach p $wave_list {
  if { [llength [get_objects -quiet $p]] > 0 } {
    add_wave $p
    incr added
  } else {
    puts "WARN: missing $p"
  }
}

set_property needs_save false [current_wave_config]
puts "INFO: GUI waves added = $added"
puts "INFO: running all (E2E ~2716160 ns, please wait)..."
run all
puts "INFO: done. 看 Waveform 窗口里的 dut/st：PRELOAD -> Q/K/V -> SCORE -> ... -> OUTPUT"
puts "INFO: 缩放：鼠标滚轮；找变化：右键信号 -> Go to Transition"
