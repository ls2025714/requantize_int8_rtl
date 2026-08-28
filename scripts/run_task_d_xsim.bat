@echo off
REM Task D: compile and run python-vector regression (120 cases).
REM Usage: scripts\run_task_d_xsim.bat

call F:\vivado\2026.1\Vivado\settings64.bat
cd /d F:\Users\22563\Desktop\FPGAlearn\requantize_int8_rtl\requantize_int8_rtl.sim\sim_1\behav\xsim

set SRC=F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/requantize_int8_rtl.srcs
set TCL=F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/scripts/xsim/tb_dot_product_parallel_python_vectors.tcl

call xvlog --incr --relax -sv -L uvm -work xil_defaultlib ^
  %SRC%/sources_1/new/int8_dot_product_parallel.sv ^
  %SRC%/sim_1/new/tb_dot_product_parallel_python_vectors.sv ^
  glbl.v -log xvlog_task_d.log
if errorlevel 1 exit /b 1

call xelab --incr --debug typical --relax --mt 2 -L xil_defaultlib -L uvm -L unisims_ver -L unimacro_ver -L secureip ^
  --snapshot tb_dot_product_parallel_python_vectors_behav ^
  xil_defaultlib.tb_dot_product_parallel_python_vectors xil_defaultlib.glbl -log xelab_task_d.log
if errorlevel 1 exit /b 1

call xsim tb_dot_product_parallel_python_vectors_behav -log simulate_task_d.log -tclbatch %TCL%
exit /b %errorlevel%
