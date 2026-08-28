@echo off
REM D8: Score GEMM Q@K^T regression
call F:\vivado\2026.1\Vivado\settings64.bat
cd /d F:\Users\22563\Desktop\FPGAlearn\requantize_int8_rtl

python scripts\generate_score_gemm_vectors.py
if errorlevel 1 exit /b 1

cd /d F:\Users\22563\Desktop\FPGAlearn\requantize_int8_rtl\requantize_int8_rtl.sim\sim_1\behav\xsim
set SRC=F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/requantize_int8_rtl.srcs

call xvlog --incr --relax -sv -work xil_defaultlib ^
  %SRC%/sources_1/new/int8_dot_product_parallel.sv ^
  %SRC%/sources_1/new/int8_score_gemm.sv ^
  %SRC%/sim_1/new/tb_int8_score_gemm.sv ^
  glbl.v -log xvlog_score.log
if errorlevel 1 exit /b 1

call xelab --incr --debug typical --relax --mt 2 -L xil_defaultlib -L unisims_ver -L unimacro_ver -L secureip ^
  --snapshot tb_int8_score_gemm_behav xil_defaultlib.tb_int8_score_gemm xil_defaultlib.glbl -log xelab_score.log
if errorlevel 1 exit /b 1

call xsim tb_int8_score_gemm_behav -log simulate_score.log -R
exit /b %errorlevel%
