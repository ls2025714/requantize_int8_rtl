@echo off
REM D11: compile and run layered FFN gate/up/down tiled regression.
REM Usage: scripts\run_ffn_layer_xsim.bat

call F:\vivado\2026.1\Vivado\settings64.bat
cd /d F:\Users\22563\Desktop\FPGAlearn\requantize_int8_rtl

set KMP_DUPLICATE_LIB_OK=TRUE
python scripts\generate_ffn_layer_vectors.py
if errorlevel 1 exit /b 1

cd /d F:\Users\22563\Desktop\FPGAlearn\requantize_int8_rtl\requantize_int8_rtl.sim\sim_1\behav\xsim

set SRC=F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/requantize_int8_rtl.srcs

call xvlog --incr --relax -sv -work xil_defaultlib ^
  %SRC%/sources_1/new/int8_dot_product_parallel.sv ^
  %SRC%/sources_1/new/int8_gemm_parallel.sv ^
  %SRC%/sources_1/new/int8_weight_loader.sv ^
  %SRC%/sources_1/new/requantize_int8_pipeline.sv ^
  %SRC%/sources_1/new/int8_linear_tiled.sv ^
  %SRC%/sim_1/new/tb_linear_tiled_ffn.sv ^
  glbl.v -log xvlog_ffn_layer.log
if errorlevel 1 exit /b 1

call xelab --incr --debug typical --relax --mt 2 -L xil_defaultlib -L unisims_ver -L unimacro_ver -L secureip ^
  --snapshot tb_linear_tiled_ffn_behav ^
  xil_defaultlib.tb_linear_tiled_ffn xil_defaultlib.glbl -log xelab_ffn_layer.log
if errorlevel 1 exit /b 1

call xsim tb_linear_tiled_ffn_behav -log simulate_ffn_layer.log -R
exit /b %errorlevel%
