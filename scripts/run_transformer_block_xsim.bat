@echo off
REM D12: generate vectors, compile and run E2E transformer block regression.
REM Usage: scripts\run_transformer_block_xsim.bat
REM Optional: set SKIP_VECTOR_GEN=1 to reuse existing transformer_block_vectors.txt

cd /d F:\Users\22563\Desktop\FPGAlearn\requantize_int8_rtl

REM Prefer conda env llm-fpga (torch). Fallback: base anaconda, then PATH python.
set "PY=python"
if exist "F:\anaconda3\python.exe" set "PY=F:\anaconda3\python.exe"
if exist "F:\anaconda3\envs\llm-fpga\python.exe" set "PY=F:\anaconda3\envs\llm-fpga\python.exe"

set KMP_DUPLICATE_LIB_OK=TRUE
set "DO_GEN=1"
if "%SKIP_VECTOR_GEN%"=="1" set "DO_GEN=0"
if exist "transformer_block_vectors.txt" if not "%FORCE_VECTOR_GEN%"=="1" set "DO_GEN=0"

if "%DO_GEN%"=="0" (
  echo Reusing transformer_block_vectors.txt ^(set FORCE_VECTOR_GEN=1 to regenerate^)
  goto :after_gen
)

echo Generating vectors with: %PY%
"%PY%" -c "import torch" 1>nul 2>nul
if errorlevel 1 (
  echo ERROR: this Python has no torch: %PY%
  echo Fix: activate Anaconda, or set SKIP_VECTOR_GEN=1 to reuse existing vectors.
  exit /b 1
)
"%PY%" scripts\generate_transformer_block_vectors.py
if errorlevel 1 exit /b 1

:after_gen

call F:\vivado\2026.1\Vivado\settings64.bat
cd /d F:\Users\22563\Desktop\FPGAlearn\requantize_int8_rtl\requantize_int8_rtl.sim\sim_1\behav\xsim

set SRC=F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/requantize_int8_rtl.srcs

call xvlog --incr --relax -sv -work xil_defaultlib -i %SRC%/sources_1/new ^
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
  glbl.v -log xvlog_transformer_block.log
if errorlevel 1 exit /b 1

call xelab --incr --debug typical --relax --mt 2 -L xil_defaultlib -L unisims_ver -L unimacro_ver -L secureip ^
  --snapshot tb_int8_transformer_block_behav ^
  xil_defaultlib.tb_int8_transformer_block xil_defaultlib.glbl -log xelab_transformer_block.log
if errorlevel 1 exit /b 1

call xsim tb_int8_transformer_block_behav -log simulate_transformer_block.log -R
exit /b %errorlevel%
