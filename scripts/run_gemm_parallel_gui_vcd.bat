@echo off
REM D3 parallel GEMM: compile + GUI waveform + VCD (no Vivado project binding).
REM Usage:
REM   scripts\run_gemm_parallel_gui_vcd.bat          :: python-vector TB (default)
REM   scripts\run_gemm_parallel_gui_vcd.bat inline   :: inline 24-case TB
REM
REM Does NOT change requantize_int8_rtl.xpr sim settings.

setlocal EnableExtensions

set MODE=%~1
if /I "%MODE%"=="" set MODE=python
if /I not "%MODE%"=="python" if /I not "%MODE%"=="inline" (
  echo Usage: scripts\run_gemm_parallel_gui_vcd.bat [python^|inline]
  exit /b 1
)

echo === D3 GEMM: GUI waves + VCD (mode=%MODE%) ===
echo NOTE: Close other Vivado/xsim sessions first, or run scripts\kill_xsim_locks.bat

tasklist /FI "IMAGENAME eq xsimk.exe" 2>nul | find /I "xsimk.exe" >nul
if not errorlevel 1 (
  echo ERROR: xsimk.exe still running.
  exit /b 1
)

if not exist "F:\vivado\2026.1\Vivado\settings64.bat" (
  echo ERROR: Vivado not found at F:\vivado\2026.1\Vivado\settings64.bat
  exit /b 1
)

call F:\vivado\2026.1\Vivado\settings64.bat
if errorlevel 1 exit /b 1

cd /d F:\Users\22563\Desktop\FPGAlearn\requantize_int8_rtl
if errorlevel 1 exit /b 1

set SRC=F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/requantize_int8_rtl.srcs
set TCL=F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/scripts/xsim/launch_gemm_parallel_with_vcd.tcl
set OUT=F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/gemm_parallel_full.vcd

if /I "%MODE%"=="inline" (
  set TB_TOP=tb_int8_gemm_parallel
  set TB_SV=%SRC%/sim_1/new/tb_int8_gemm_parallel.sv
  set SNAPSHOT=tb_int8_gemm_parallel_behav
) else (
  set TB_TOP=tb_gemm_parallel_python_vectors
  set TB_SV=%SRC%/sim_1/new/tb_gemm_parallel_python_vectors.sv
  set SNAPSHOT=tb_gemm_parallel_python_vectors_behav
  python scripts\generate_gemm_parallel_vectors.py
  if errorlevel 1 exit /b 1
)

cd /d F:\Users\22563\Desktop\FPGAlearn\requantize_int8_rtl\requantize_int8_rtl.sim\sim_1\behav\xsim
if errorlevel 1 exit /b 1

echo [1/3] xvlog compile (%TB_TOP%)...
call xvlog --incr --relax -sv -work xil_defaultlib ^
  %SRC%/sources_1/new/int8_dot_product_parallel.sv ^
  %SRC%/sources_1/new/int8_gemm_parallel.sv ^
  %TB_SV% ^
  glbl.v -log xvlog_gemm_parallel_gui.log
if errorlevel 1 exit /b 1

echo [2/3] xelab elaborate...
call xelab --incr --debug typical --relax --mt 2 -L xil_defaultlib -L unisims_ver -L unimacro_ver -L secureip ^
  --snapshot %SNAPSHOT% ^
  xil_defaultlib.%TB_TOP% xil_defaultlib.glbl -log xelab_gemm_parallel_gui.log
if errorlevel 1 exit /b 1

echo [3/3] xsim GUI + waves + VCD...
call xsim %SNAPSHOT% -gui -tclbatch %TCL% -log simulate_gemm_parallel_gui.log
if errorlevel 1 exit /b 1

if exist %OUT% (
  echo.
  echo OK: %OUT%
  for %%A in (%OUT%) do echo     size=%%~zA bytes
) else (
  echo WARN: VCD not found at %OUT%
  exit /b 1
)

exit /b 0
