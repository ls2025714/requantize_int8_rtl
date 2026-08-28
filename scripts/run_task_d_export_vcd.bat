@echo off
REM Task D: compile + simulate + export full VCD -> repo_root\task_d_full.vcd
REM Usage: scripts\run_task_d_export_vcd.bat

echo === Task D VCD export ===
echo NOTE: Close Vivado Simulation / Vivado GUI before running, or xelab may fail with file lock.

tasklist /FI "IMAGENAME eq xsimk.exe" 2>nul | find /I "xsimk.exe" >nul
if not errorlevel 1 (
  echo ERROR: xsimk.exe is still running. Close Vivado sim or run: scripts\kill_xsim_locks.bat
  exit /b 1
)

if not exist "F:\vivado\2026.1\Vivado\settings64.bat" (
  echo ERROR: Vivado not found at F:\vivado\2026.1\Vivado\settings64.bat
  exit /b 1
)

call F:\vivado\2026.1\Vivado\settings64.bat
if errorlevel 1 (
  echo ERROR: Vivado settings64.bat failed
  exit /b 1
)

cd /d F:\Users\22563\Desktop\FPGAlearn\requantize_int8_rtl\requantize_int8_rtl.sim\sim_1\behav\xsim
if errorlevel 1 (
  echo ERROR: cannot cd to xsim directory
  exit /b 1
)

set SRC=F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/requantize_int8_rtl.srcs
set TCL=F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/scripts/xsim/export_full_wave_task_d.tcl
set OUT=F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/task_d_full.vcd

echo [1/3] xvlog compile...
call xvlog --incr --relax -sv -L uvm -work xil_defaultlib ^
  %SRC%/sources_1/new/int8_dot_product_parallel.sv ^
  %SRC%/sim_1/new/tb_dot_product_parallel_python_vectors.sv ^
  glbl.v -log xvlog_task_d.log
if errorlevel 1 (
  echo ERROR: xvlog failed. See xvlog_task_d.log
  exit /b 1
)

echo [2/3] xelab elaborate...
call xelab --incr --debug typical --relax --mt 2 -L xil_defaultlib -L uvm -L unisims_ver -L unimacro_ver -L secureip ^
  --snapshot tb_dot_product_parallel_python_vectors_behav ^
  xil_defaultlib.tb_dot_product_parallel_python_vectors xil_defaultlib.glbl -log xelab_task_d.log
if errorlevel 1 (
  echo ERROR: xelab failed. See xelab_task_d.log
  echo.
  echo Usually caused by Vivado/xsim still running and locking xsim.dir
  echo Fix: 1^) Close Vivado completely  2^) scripts\kill_xsim_locks.bat  3^) re-run this bat
  exit /b 1
)

echo [3/3] xsim run + VCD export...
call xsim tb_dot_product_parallel_python_vectors_behav -log simulate_task_d_vcd.log -tclbatch %TCL%
if errorlevel 1 (
  echo ERROR: xsim failed. See simulate_task_d_vcd.log
  exit /b 1
)

if exist %OUT% (
  echo.
  echo OK: %OUT%
  for %%A in (%OUT%) do echo     size=%%~zA bytes
) else (
  echo ERROR: VCD not found at %OUT%
  exit /b 1
)

exit /b 0
