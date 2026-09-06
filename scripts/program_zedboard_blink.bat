@echo off
REM Program ZedBoard with blink bitstream over JTAG.
REM Prerequisites:
REM   1) scripts\build_zedboard_blink.bat succeeded
REM   2) ZedBoard powered ON
REM   3) USB cable on PROG/JTAG port (not just UART)
REM   4) Close Vivado Hardware Manager if it already holds the cable

call F:\vivado\2026.1\Vivado\settings64.bat
cd /d F:\Users\22563\Desktop\FPGAlearn\requantize_int8_rtl

if not exist build\zedboard_pl_blink\zedboard_pl_blink.bit (
  echo ERROR: bitstream missing. Run scripts\build_zedboard_blink.bat first.
  exit /b 1
)

echo Programming ZedBoard...
call vivado -mode batch -source scripts\program_zedboard_blink.tcl -log build\zedboard_pl_blink\vivado_program.log -journal build\zedboard_pl_blink\vivado_program.jou
if errorlevel 1 (
  echo ERROR: program failed. See build\zedboard_pl_blink\vivado_program.log
  exit /b 1
)

echo OK: programmed. Watch LD0..LD3.
exit /b 0
