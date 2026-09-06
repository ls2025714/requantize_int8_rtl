# Non-project flow: synth → impl → bitstream for ZedBoard PL blink smoke test.
# Usage (from Vivado Tcl or via build_zedboard_blink.bat):
#   source scripts/build_zedboard_blink.tcl

set repo [file normalize [file join [file dirname [info script]] ..]]
set src  [file join $repo requantize_int8_rtl.srcs/sources_1/new/zedboard_pl_blink.sv]
set xdc  [file join $repo requantize_int8_rtl.srcs/constrs_1/new/zedboard_pl_blink.xdc]
set outd [file join $repo build/zedboard_pl_blink]
set part xc7z020clg484-1

file mkdir $outd
cd $outd

puts "INFO: repo=$repo"
puts "INFO: building zedboard_pl_blink for $part"

read_verilog -sv $src
read_xdc $xdc

synth_design -top zedboard_pl_blink -part $part
write_checkpoint -force [file join $outd blink_synth.dcp]

opt_design
place_design
route_design
write_checkpoint -force [file join $outd blink_routed.dcp]

report_timing_summary -file [file join $outd timing_summary.rpt]
report_utilization -file [file join $outd utilization.rpt]

set bitfile [file join $outd zedboard_pl_blink.bit]
write_bitstream -force $bitfile

puts "INFO: bitstream ready -> $bitfile"
puts "INFO: next: scripts/program_zedboard_blink.bat"
