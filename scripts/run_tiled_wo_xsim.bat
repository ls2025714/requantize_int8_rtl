@echo off
REM D10: compile and run Wo tiled N=64 K=64 regression.
REM Usage: scripts\run_tiled_wo_xsim.bat

call F:\vivado\2026.1\Vivado\settings64.bat
cd /d F:\Users\22563\Desktop\FPGAlearn\requantize_int8_rtl

set KMP_DUPLICATE_LIB_OK=TRUE
python scripts\generate_linear_tiled_wo_vectors.py
if errorlevel 1 exit /b 1

cd /d F:\Users\22563\Desktop\FPGAlearn\requantize_int8_rtl\requantize_int8_rtl.sim\sim_1\behav\xsim

set SRC=F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/requantize_int8_rtl.srcs

call xvlog --incr --relax -sv -work xil_defaultlib ^
  %SRC%/sources_1/new/int8_dot_product_parallel.sv ^
  %SRC%/sources_1/new/int8_gemm_parallel.sv ^
  %SRC%/sources_1/new/int8_weight_loader.sv ^
  %SRC%/sources_1/new/requantize_int8_pipeline.sv ^
  %SRC%/sources_1/new/int8_linear_tiled.sv ^
  %SRC%/sim_1/new/tb_linear_tiled_wo.sv ^
  glbl.v -log xvlog_tiled_wo.log
if errorlevel 1 exit /b 1

call xelab --incr --debug typical --relax --mt 2 -L xil_defaultlib -L unisims_ver -L unimacro_ver -L secureip ^
  --snapshot tb_linear_tiled_wo_behav ^
  xil_defaultlib.tb_linear_tiled_wo xil_defaultlib.glbl -log xelab_tiled_wo.log
if errorlevel 1 exit /b 1

call xsim tb_linear_tiled_wo_behav -log simulate_tiled_wo.log -R
exit /b %errorlevel%
