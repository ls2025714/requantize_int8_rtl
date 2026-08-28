// ============================================================================
// TB: tb_int8_mac.sv
// 测:  int8_mac
// 作用: 基础 MAC clear/enable/累加定向测试
// ============================================================================

// 定向测试 int8_mac：clear、enable、正负乘积、累加序列
`timescale 1ns / 1ps
module tb_int8_mac;
logic clk;
logic rst_n;
logic clear_i;
logic enable_i;
logic in_valid;
logic signed [7:0] a_i;
logic signed [7:0] b_i;
logic out_valid;
logic signed [31:0] acc_o;
integer pass_count;
integer fail_count;
int8_mac #(
    .DATA_W(8),
    .ACC_W(32)
) dut (
    .clk(clk),
    .rst_n(rst_n),
    .clear_i(clear_i),
    .enable_i(enable_i),
    .in_valid(in_valid),
    .a_i(a_i),
    .b_i(b_i),
    .out_valid(out_valid),
    .acc_o(acc_o)
);
initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
end
task automatic apply_and_check(
    input string test_name,
    input logic test_clear,
    input logic test_enable,
    input logic test_valid,
    input logic signed [7:0] test_a,
    input logic signed [7:0] test_b,
    input logic expected_valid,
    input logic signed [31:0] expected_acc
);
begin
    @(negedge clk);
    clear_i = test_clear;
    enable_i = test_enable;
    in_valid = test_valid;
    a_i = test_a;
    b_i = test_b;
    @(posedge clk);
    #1;
    if (out_valid !== expected_valid) begin
        fail_count = fail_count + 1;
        $error(
            "%s VALID FAIL: out_valid=%0b expected=%0b",
            test_name,
            out_valid,
            expected_valid
        );
    end
    else if ($signed(acc_o) !== expected_acc) begin
        fail_count = fail_count + 1;
        $error(
            "%s ACC FAIL: a=%0d b=%0d acc=%0d expected=%0d",
            test_name,
            test_a,
            test_b,
            $signed(acc_o),
            expected_acc
        );
    end
    else begin
        pass_count = pass_count + 1;
        $display(
            "%s PASS: clear=%0b enable=%0b valid=%0b a=%0d b=%0d product=%0d acc=%0d out_valid=%0b",
            test_name,
            test_clear,
            test_enable,
            test_valid,
            test_a,
            test_b,
            $signed(dut.product_comb),
            $signed(acc_o),
            out_valid
        );
    end
    clear_i = 1'b0;
    enable_i = 1'b0;
    in_valid = 1'b0;
    a_i = 8'sd0;
    b_i = 8'sd0;
end
endtask
initial begin
    rst_n = 1'b0;
    clear_i = 1'b0;
    enable_i = 1'b0;
    in_valid = 1'b0;
    a_i = 8'sd0;
    b_i = 8'sd0;
    pass_count = 0;
    fail_count = 0;
    repeat (3) @(posedge clk);
    @(negedge clk);
    rst_n = 1'b1;
    apply_and_check("TEST 1 CLEAR", 1'b1, 1'b0, 1'b0, 8'sd0, 8'sd0, 1'b1, 32'sd0);
    apply_and_check("TEST 2 POSITIVE", 1'b0, 1'b1, 1'b1, 8'sd3, 8'sd4, 1'b1, 32'sd12);
    apply_and_check("TEST 3 MIXED SIGN", 1'b0, 1'b1, 1'b1, -8'sd5, 8'sd6, 1'b1, -32'sd18);
    apply_and_check("TEST 4 BOTH NEGATIVE", 1'b0, 1'b1, 1'b1, -8'sd7, -8'sd8, 1'b1, 32'sd38);
    apply_and_check("TEST 5 POSITIVE MAX", 1'b0, 1'b1, 1'b1, 8'sd127, 8'sd127, 1'b1, 32'sd16167);
    apply_and_check("TEST 6 DISABLED", 1'b0, 1'b0, 1'b1, 8'sd100, 8'sd100, 1'b0, 32'sd16167);
    apply_and_check("TEST 7 INVALID", 1'b0, 1'b1, 1'b0, -8'sd100, 8'sd100, 1'b0, 32'sd16167);
    apply_and_check("TEST 8 CLEAR AGAIN", 1'b1, 1'b1, 1'b1, 8'sd127, 8'sd127, 1'b1, 32'sd0);
    apply_and_check("TEST 9 INT8 MIN SQUARE", 1'b0, 1'b1, 1'b1, -8'sd128, -8'sd128, 1'b1, 32'sd16384);
    apply_and_check("TEST 10 FINAL", 1'b0, 1'b1, 1'b1, -8'sd128, 8'sd127, 1'b1, 32'sd128);
    $display("============================================================");
    $display("INT8 MAC test summary");
    $display("PASS count = %0d", pass_count);
    $display("FAIL count = %0d", fail_count);
    $display("============================================================");
    if (fail_count == 0) begin
        $display("ALL INT8 MAC TESTS PASSED.");
        $finish;
    end
    else begin
        $fatal(1, "INT8 MAC TEST FAILED: %0d failure(s)", fail_count);
    end
end
endmodule