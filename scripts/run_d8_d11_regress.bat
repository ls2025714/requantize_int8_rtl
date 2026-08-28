@echo off
cd /d F:\Users\22563\Desktop\FPGAlearn\requantize_int8_rtl
call scripts\kill_xsim_locks.bat
call scripts\run_score_gemm_xsim.bat > regress_d8.log 2>&1
echo D8 exit=%ERRORLEVEL%
call scripts\run_softmax_causal_xsim.bat > regress_d9.log 2>&1
echo D9 exit=%ERRORLEVEL%
call scripts\run_attn_v_xsim.bat > regress_d10a.log 2>&1
echo D10a exit=%ERRORLEVEL%
call scripts\run_tiled_wo_xsim.bat > regress_d10b.log 2>&1
echo D10b exit=%ERRORLEVEL%
call scripts\run_residual_add_xsim.bat > regress_d11a.log 2>&1
echo D11a exit=%ERRORLEVEL%
call scripts\run_silu_lut_xsim.bat > regress_d11b.log 2>&1
echo D11b exit=%ERRORLEVEL%
call scripts\run_ffn_layer_xsim.bat > regress_d11c.log 2>&1
echo D11c exit=%ERRORLEVEL%
echo ALL_REGRESS_DONE
