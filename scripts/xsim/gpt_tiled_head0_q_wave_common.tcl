# 共享：D6 Tiled head0 Q 波形 / VCD 信号列表

proc detect_tiled_head0_q_tb_top {} {
  foreach candidate {tb_linear_tiled_head0_q} {
    if { [llength [get_objects -quiet /${candidate}/clk]] > 0 } {
      return $candidate
    }
  }
  return ""
}

proc resolve_tiled_head0_q_tb_top {requested_top} {
  if { $requested_top ne "" && [llength [get_objects -quiet /${requested_top}/clk]] > 0 } {
    return $requested_top
  }
  set detected [detect_tiled_head0_q_tb_top]
  if { $detected ne "" } {
    if { $requested_top ne "" && $requested_top ne $detected } {
      puts "WARN: requested TB_TOP=$requested_top not in sim; using detected $detected"
    }
    return $detected
  }
  return $requested_top
}

proc log_tiled_head0_q_vcd {tb_top} {
  set signal_list {
    clk rst_n
    cmd_valid cmd_ready cmd_m cmd_n cmd_k
    a_valid a_ready a_data
    w_valid w_ready w_data
    mult_valid mult_ready mult_data
    c_valid c_ready c_data c_row c_col acc_debug
    dut/tile_state dut/n_tile dut/k_tile dut/tile_idx dut/k_base dut/n_base
    dut/replay_start dut/replay_base_addr dut/replay_length
    dut/held_acc dut/held_mult dut/out_idx
    dut/u_weight_loader/replay_active dut/u_weight_loader/r_valid dut/u_weight_loader/r_ready
    dut/u_weight_loader/r_addr dut/u_weight_loader/r_remaining
    dut/u_gemm/state dut/u_gemm/c_valid dut/u_gemm/c_ready dut/u_gemm/c_data
    dut/u_gemm/c_row dut/u_gemm/c_col
    dut/u_requant/in_valid dut/u_requant/out_valid dut/u_requant/out_i
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

proc add_tiled_head0_q_waves {tb_top} {
  set signal_list {
    clk rst_n
    cmd_valid cmd_ready cmd_m cmd_n cmd_k
    a_valid a_ready a_data
    w_valid w_ready w_data
    mult_valid mult_ready mult_data
    c_valid c_ready c_data c_row c_col acc_debug
    dut/tile_state dut/n_tile dut/k_tile dut/tile_idx dut/k_base dut/n_base
    dut/replay_start dut/replay_base_addr dut/replay_length
    dut/held_acc dut/held_mult dut/out_idx
    dut/u_weight_loader/replay_active dut/u_weight_loader/r_valid dut/u_weight_loader/r_ready
    dut/u_weight_loader/r_addr dut/u_weight_loader/r_remaining
    dut/u_gemm/state dut/u_gemm/c_valid dut/u_gemm/c_ready dut/u_gemm/c_data
    dut/u_gemm/c_row dut/u_gemm/c_col
    dut/u_requant/in_valid dut/u_requant/out_valid dut/u_requant/out_i
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
  return $added
}
