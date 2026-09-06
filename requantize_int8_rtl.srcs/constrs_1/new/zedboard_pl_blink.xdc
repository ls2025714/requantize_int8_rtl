# ============================================================================
# 文件: zedboard_pl_blink.xdc
# 板卡: Digilent ZedBoard (xc7z020clg484-1)
# Top:  zedboard_pl_blink
# ============================================================================

# 100 MHz oscillator -> PL (GCLK)
set_property PACKAGE_PIN Y9 [get_ports clk]
set_property IOSTANDARD LVCMOS33 [get_ports clk]
create_clock -period 10.000 -name sys_clk [get_ports clk]

# Bank 33 LEDs LD0..LD3
set_property PACKAGE_PIN T22 [get_ports {led[0]}]
set_property PACKAGE_PIN T21 [get_ports {led[1]}]
set_property PACKAGE_PIN U22 [get_ports {led[2]}]
set_property PACKAGE_PIN U21 [get_ports {led[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[*]}]
