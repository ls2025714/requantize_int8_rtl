`timescale 1ns / 1ps
module tb_requantize_python_vectors;
logic signed [31:0] acc_i;
logic        [17:0] multiplier_i;
logic signed  [7:0] out_i;
integer exact_match_count;
integer mismatch_count;
requantize_int8 #(
    .SHIFT_BITS(24)
) dut (
    .acc_i(acc_i),
    .multiplier_i(multiplier_i),
    .out_i(out_i)
);
task automatic check_vector(
    input integer n,
    input integer m,
    input logic signed [31:0] test_acc,
    input logic        [17:0] test_multiplier,
    input logic signed  [7:0] expected_output
);
begin
    acc_i = test_acc;
    multiplier_i = test_multiplier;
    #10;
    if ($signed(out_i) === expected_output) begin
        exact_match_count = exact_match_count + 1;
        $display(
            "MATCH [%0d,%0d]: acc=%0d mult=%0d pre_clip=%0d rtl=%0d expected=%0d",
            n,
            m,
            test_acc,
            test_multiplier,
            $signed(dut.shifted_value),
            $signed(out_i),
            expected_output
        );
    end
    else begin
        mismatch_count = mismatch_count + 1;
        $error(
            "MISMATCH [%0d,%0d]: acc=%0d mult=%0d pre_clip=%0d rtl=%0d expected=%0d",
            n,
            m,
            test_acc,
            test_multiplier,
            $signed(dut.shifted_value),
            $signed(out_i),
            expected_output
        );
    end
end
endtask
initial begin
    exact_match_count = 0;
    mismatch_count = 0;
    acc_i = 32'sd0;
    multiplier_i = 18'd0;
    #10;
    $display("============================================================");
    $display("Python v2 versus RTL: 3x4 output comparison");
    $display("============================================================");
    check_vector(0, 0,  32'sd100,    18'd26979,   8'sd0);
    check_vector(0, 1,  32'sd23673,  18'd88887,   8'sd125);
    check_vector(0, 2, -32'sd10447,  18'd205831, -8'sd127);
    check_vector(0, 3,  32'sd24709,  18'd45891,   8'sd68);
    check_vector(1, 0,  32'sd11295,  18'd26979,   8'sd18);
    check_vector(1, 1,  32'sd1755,   18'd88887,   8'sd9);
    check_vector(1, 2, -32'sd3961,   18'd205831, -8'sd49);
    check_vector(1, 3,  32'sd3189,   18'd45891,   8'sd9);
    check_vector(2, 0,  32'sd11428,  18'd26979,   8'sd18);
    check_vector(2, 1,  32'sd4030,   18'd88887,   8'sd21);
    check_vector(2, 2,  32'sd281,    18'd205831,  8'sd3);
    check_vector(2, 3,  32'sd2340,   18'd45891,   8'sd6);
    $display("============================================================");
    $display("element_count     = 12");
    $display("exact_match_count = %0d", exact_match_count);
    $display("mismatch_count    = %0d", mismatch_count);
    $display("============================================================");
    if (mismatch_count == 0 && exact_match_count == 12) begin
        $display("PYTHON V2 AND RTL ARE 12/12 EXACT.");
        $finish;
    end
    else begin
        $fatal(
            1,
            "RTL-Python comparison failed: exact=%0d mismatch=%0d",
            exact_match_count,
            mismatch_count
        );
    end
end
endmodule