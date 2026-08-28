# Add curated 32-signal waveform for tb_int8_dot_product_parallel.
# Usage (sim must be running):
#   source F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/scripts/xsim/apply_waveform_only.tcl

set TB_TOP tb_int8_dot_product_parallel

proc add_dot_product_parallel_waves {tb_top} {
  add_wave /${tb_top}/clk
  add_wave /${tb_top}/rst_n
  add_wave /${tb_top}/cmd_valid
  add_wave /${tb_top}/cmd_ready
  add_wave /${tb_top}/cmd_length
  add_wave /${tb_top}/s_valid
  add_wave /${tb_top}/s_ready
  add_wave /${tb_top}/s_a0
  add_wave /${tb_top}/s_a1
  add_wave /${tb_top}/s_a2
  add_wave /${tb_top}/s_a3
  add_wave /${tb_top}/s_b0
  add_wave /${tb_top}/s_b1
  add_wave /${tb_top}/s_b2
  add_wave /${tb_top}/s_b3
  add_wave /${tb_top}/s_keep
  add_wave /${tb_top}/acc_clear
  add_wave /${tb_top}/acc_enable
  add_wave /${tb_top}/dut/state
  add_wave /${tb_top}/dut/valid_s1
  add_wave /${tb_top}/dut/valid_s2
  add_wave /${tb_top}/dut/valid_s3
  add_wave /${tb_top}/dut/valid_s4
  add_wave /${tb_top}/partial_valid
  add_wave /${tb_top}/partial_sum
  add_wave /${tb_top}/acc_valid
  add_wave /${tb_top}/acc_value
  add_wave /${tb_top}/m_valid
  add_wave /${tb_top}/m_ready
  add_wave /${tb_top}/m_result
  add_wave /${tb_top}/test_count
  add_wave /${tb_top}/mismatch_count
}

if { [llength [get_objects]] == 0 } {
  puts "ERROR: no simulation objects. Launch simulation first, or type: restart"
  return
}

foreach wc [get_wave_configs] {
  catch { close_wave_config -force $wc }
}
create_wave_config dot_product_parallel

set old_waves [get_waves *]
if { [llength $old_waves] > 0 } {
  delete_wave $old_waves
}

add_dot_product_parallel_waves $TB_TOP
set_property needs_save false [current_wave_config]

set wcfg_path [file normalize [file join [file dirname [info script]] tb_int8_dot_product_parallel_behav.wcfg]]
catch { save_wave_config $wcfg_path }

puts "INFO: waveform ready, signals = [llength [get_waves *]]"
