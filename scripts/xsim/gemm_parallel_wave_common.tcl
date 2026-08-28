# 共享：D3 并行 GEMM 波形 / VCD 信号列表（由 launch_gemm_parallel_with_vcd.tcl source）

proc detect_gemm_tb_top {} {
  foreach candidate {tb_gemm_parallel_python_vectors tb_int8_gemm_parallel} {
    if { [llength [get_objects -quiet /${candidate}/clk]] > 0 } {
      return $candidate
    }
  }
  return ""
}

proc resolve_gemm_tb_top {requested_top} {
  if { $requested_top ne "" && [llength [get_objects -quiet /${requested_top}/clk]] > 0 } {
    return $requested_top
  }
  set detected [detect_gemm_tb_top]
  if { $detected ne "" } {
    if { $requested_top ne "" && $requested_top ne $detected } {
      puts "WARN: requested TB_TOP=$requested_top not in sim; using detected $detected"
    }
    return $detected
  }
  return $requested_top
}

proc log_gemm_parallel_vcd {tb_top} {
  set signal_list {
    clk rst_n
    cmd_valid cmd_ready cmd_m cmd_n cmd_k
    a_valid a_ready a_data
    b_valid b_ready b_data
    c_valid c_ready c_data c_row c_col
    dut/state dut/row_index dut/col_index dut/beat_index
    dut/dot_s_valid dut/dot_s_ready dut/dot_s_keep
    dut/dot_m_valid dut/dot_m_ready dut/dot_m_result
    dut/u_dot_product/state
    test_count mismatch_count
  }
  set logged 0
  set missing {}
  foreach sig $signal_list {
    set path /${tb_top}/${sig}
    if { [llength [get_objects -quiet $path]] > 0 } {
      log_vcd $path
      incr logged
    } else {
      lappend missing $path
    }
  }
  puts "INFO: log_vcd registered $logged signals for /$tb_top"
  if { [llength $missing] > 0 } {
    puts "WARN: missing [llength $missing] paths (first 3): [lrange $missing 0 2]"
  }
  return $logged
}

proc add_gemm_parallel_waves {tb_top} {
  set signal_list {
    clk rst_n
    cmd_valid cmd_ready cmd_m cmd_n cmd_k
    a_valid a_ready a_data
    b_valid b_ready b_data
    c_valid c_ready c_data c_row c_col
    dut/state dut/row_index dut/col_index dut/beat_index
    dut/dot_s_valid dut/dot_s_ready dut/dot_s_keep
    dut/dot_m_valid dut/dot_m_ready dut/dot_m_result
    dut/u_dot_product/state
    test_count mismatch_count
  }
  set added 0
  foreach sig $signal_list {
    set path /${tb_top}/${sig}
    if { [llength [get_objects -quiet $path]] > 0 } {
      add_wave $path
      incr added
    }
  }
  puts "INFO: add_wave added $added signals for /$tb_top"
  return $added
}
