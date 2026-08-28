# Launch script: Task D curated waves then run full regression.
source [file normalize [file join [file dirname [info script]] apply_waveform_task_d.tcl]]
run all
