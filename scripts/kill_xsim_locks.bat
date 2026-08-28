@echo off
REM Kill leftover xsim processes that lock xsim.dir (close Vivado first if possible).
echo Checking for running xsim / Vivado simulator processes...

tasklist /FI "IMAGENAME eq xsimk.exe" 2>nul | find /I "xsimk.exe" >nul
if not errorlevel 1 (
  echo Killing xsimk.exe ...
  taskkill /F /IM xsimk.exe >nul 2>&1
)

tasklist /FI "IMAGENAME eq xsim.exe" 2>nul | find /I "xsim.exe" >nul
if not errorlevel 1 (
  echo Killing xsim.exe ...
  taskkill /F /IM xsim.exe >nul 2>&1
)

tasklist /FI "IMAGENAME eq xelab.exe" 2>nul | find /I "xelab.exe" >nul
if not errorlevel 1 (
  echo Killing xelab.exe ...
  taskkill /F /IM xelab.exe >nul 2>&1
)

echo Done. Now close Vivado GUI if still open, then re-run run_task_d_export_vcd.bat
exit /b 0
