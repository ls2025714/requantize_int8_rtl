@echo off
REM Build ZedBoard PL blink bitstream (non-project flow).
REM Usage (repo root): scripts\build_zedboard_blink.bat
REM Takes several minutes. Close other heavy Vivado runs if possible.

call F:\vivado\2026.1\Vivado\settings64.bat
cd /d F:\Users\22563\Desktop\FPGAlearn\requantize_int8_rtl

if not exist build\zedboard_pl_blink mkdir build\zedboard_pl_blink

echo Building zedboard_pl_blink bitstream...
call vivado -mode batch -source scripts\build_zedboard_blink.tcl -log build\zedboard_pl_blink\vivado_build.log -journal build\zedboard_pl_blink\vivado_build.jou
if errorlevel 1 (
  echo ERROR: build failed. See build\zedboard_pl_blink\vivado_build.log
  exit /b 1
)

if not exist build\zedboard_pl_blink\zedboard_pl_blink.bit (
  echo ERROR: bitstream not found
  exit /b 1
)

echo.
echo OK: build\zedboard_pl_blink\zedboard_pl_blink.bit
echo Next: power ZedBoard, plug PROG USB, then scripts\program_zedboard_blink.bat
exit /b 0
