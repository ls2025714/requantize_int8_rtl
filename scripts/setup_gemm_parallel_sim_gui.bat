@echo off
REM OPTIONAL: bind D3 launch script to Vivado GUI "Launch Simulation" button.
REM You do NOT need this if you use run_gemm_parallel_gui_vcd.bat from cmd.
REM Usage: scripts\setup_gemm_parallel_sim_gui.bat

call F:\vivado\2026.1\Vivado\settings64.bat
cd /d F:\Users\22563\Desktop\FPGAlearn\requantize_int8_rtl
vivado -mode batch -source scripts\setup_gemm_parallel_sim.tcl
exit /b %errorlevel%
