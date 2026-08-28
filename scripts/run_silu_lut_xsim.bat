@echo off
call F:\vivado\2026.1\Vivado\settings64.bat
cd /d F:\Users\22563\Desktop\FPGAlearn\requantize_int8_rtl
python scripts\generate_silu_lut_vectors.py
python scripts\silu_fixed_ref.py requantize_int8_rtl.srcs\sources_1\new\silu_lut_params.svh
if errorlevel 1 exit /b 1
cd /d F:\Users\22563\Desktop\FPGAlearn\requantize_int8_rtl\requantize_int8_rtl.sim\sim_1\behav\xsim
set SRC=F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/requantize_int8_rtl.srcs
call xvlog --incr --relax -sv -work xil_defaultlib -i %SRC%/sources_1/new %SRC%/sources_1/new/int8_silu_lut.sv %SRC%/sim_1/new/tb_int8_silu_lut.sv glbl.v
if errorlevel 1 exit /b 1
call xelab --incr --relax --mt 2 -L xil_defaultlib -L unisims_ver -L secureip --snapshot tb_int8_silu_lut_behav xil_defaultlib.tb_int8_silu_lut xil_defaultlib.glbl
if errorlevel 1 exit /b 1
call xsim tb_int8_silu_lut_behav -R
exit /b %errorlevel%
