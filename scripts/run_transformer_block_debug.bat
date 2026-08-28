@echo off
REM D12 debug: compile with TRANSFORMER_BLOCK_DBG, run xsim + VCD, stop at 50ms sim time.
REM Usage: scripts\run_transformer_block_debug.bat

call F:\vivado\2026.1\Vivado\settings64.bat
cd /d F:\Users\22563\Desktop\FPGAlearn\requantize_int8_rtl\requantize_int8_rtl.sim\sim_1\behav\xsim

set SRC=F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/requantize_int8_rtl.srcs
set REPO=F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl

call xvlog --incr --relax -sv -d TRANSFORMER_BLOCK_DBG -work xil_defaultlib -i %SRC%/sources_1/new ^
  %SRC%/sources_1/new/int8_dot_product_parallel.sv ^
  %SRC%/sources_1/new/int8_gemm_parallel.sv ^
  %SRC%/sources_1/new/int8_weight_loader.sv ^
  %SRC%/sources_1/new/requantize_int8_pipeline.sv ^
  %SRC%/sources_1/new/int8_linear_tiled.sv ^
  %SRC%/sources_1/new/int8_score_gemm.sv ^
  %SRC%/sources_1/new/int8_softmax_causal.sv ^
  %SRC%/sources_1/new/int8_attn_v.sv ^
  %SRC%/sources_1/new/int8_residual_add.sv ^
  %SRC%/sources_1/new/int8_silu_lut.sv ^
  %SRC%/sources_1/new/int8_elem_mul.sv ^
  %SRC%/sources_1/new/int8_transformer_block.sv ^
  %SRC%/sim_1/new/tb_int8_transformer_block.sv ^
  glbl.v -log xvlog_transformer_block_dbg.log
if errorlevel 1 exit /b 1

call xelab --incr --debug typical --relax --mt 2 -L xil_defaultlib -L unisims_ver -L unimacro_ver -L secureip ^
  --snapshot tb_int8_transformer_block_dbg ^
  xil_defaultlib.tb_int8_transformer_block xil_defaultlib.glbl -log xelab_transformer_block_dbg.log
if errorlevel 1 exit /b 1

call xsim tb_int8_transformer_block_dbg -log simulate_transformer_block_dbg.log -tclbatch %REPO%/scripts/xsim/run_transformer_block_dbg.tcl
exit /b %errorlevel%
