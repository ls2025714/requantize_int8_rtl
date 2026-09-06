# Program ZedBoard with PL blink bitstream via JTAG.
# Board must be powered; PROG/JTAG USB connected; hw_server reachable.

set repo [file normalize [file join [file dirname [info script]] ..]]
set bitfile [file join $repo build/zedboard_pl_blink/zedboard_pl_blink.bit]

if { ![file exists $bitfile] } {
  puts "ERROR: missing $bitfile"
  puts "  Run scripts/build_zedboard_blink.bat first."
  exit 1
}

open_hw_manager
connect_hw_server
open_hw_target

set devices [get_hw_devices]
if { [llength $devices] == 0 } {
  puts "ERROR: no hardware devices. Check power + PROG USB + drivers."
  exit 1
}

puts "INFO: devices = $devices"
# Zynq shows arm_dap_* + xc7z020_*; only the FPGA fabric device is programmable.
set fpga ""
foreach d $devices {
  if { [string match -nocase "*xc7z020*" $d] } {
    set fpga $d
    break
  }
}
if { $fpga eq "" } {
  puts "ERROR: no xc7z020 device among: $devices"
  exit 1
}
puts "INFO: programming $fpga with $bitfile"
current_hw_device $fpga
set_property PROGRAM.FILE $bitfile $fpga
program_hw_devices $fpga
refresh_hw_device $fpga

puts "INFO: done. LD0..LD3 should blink at different rates."
exit 0
