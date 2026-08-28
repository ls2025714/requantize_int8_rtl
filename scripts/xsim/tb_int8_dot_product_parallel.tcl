# Launch script: setup waves then run full testbench.
source [file normalize [file join [file dirname [info script]] apply_waveform_only.tcl]]
run all
