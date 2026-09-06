// ============================================================================
// 文件: tb_dot_product_int8.sv
// 阶段: 串行点积
// 作用: 定向+随机测试串行点积 m_result
// DUT: dot_product_int8
// 向量: 内联定向/随机 case
// ============================================================================
//
// 流程: send cmd_length → 流式 s_a/s_b → 接收 m_result 对拍；统计 mismatch
//
`timescale 1ns/1ps
module tb_dot_product_int8;
localparam int INPUT_WIDTH = 8;
localparam int ACC_WIDTH = 32;
localparam int MAX_K = 16;
localparam int LENGTH_WIDTH = $clog2(MAX_K + 1);
logic clk;
logic rst_n;
logic cmd_valid;
logic cmd_ready;
logic [LENGTH_WIDTH-1:0] cmd_length;
logic s_valid;
logic s_ready;
logic signed [INPUT_WIDTH-1:0] s_a;
logic signed [INPUT_WIDTH-1:0] s_b;
logic m_valid;
logic m_ready;
logic signed [ACC_WIDTH-1:0] m_result;
integer test_count;
integer mismatch_count;
integer seed;
dot_product_int8 #(
    .INPUT_WIDTH(INPUT_WIDTH),
    .ACC_WIDTH(ACC_WIDTH),
    .MAX_K(MAX_K),
    .LENGTH_WIDTH(LENGTH_WIDTH)
) dut (
    .clk(clk),
    .rst_n(rst_n),
    .cmd_valid(cmd_valid),
    .cmd_ready(cmd_ready),
    .cmd_length(cmd_length),
    .s_valid(s_valid),
    .s_ready(s_ready),
    .s_a(s_a),
    .s_b(s_b),
    .m_valid(m_valid),
    .m_ready(m_ready),
    .m_result(m_result)
);
initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
end
task automatic send_command(input integer length);
    begin
        @(negedge clk);
        cmd_length = length[LENGTH_WIDTH-1:0];
        cmd_valid = 1'b1;
        while (!cmd_ready) begin
            @(negedge clk);
        end
        @(posedge clk);
        @(negedge clk);
        cmd_valid = 1'b0;
    end
endtask
task automatic send_pair(input integer signed a_value, input integer signed b_value, input integer gap_cycles);
    integer gap_index;
    begin
        for (gap_index = 0; gap_index < gap_cycles; gap_index = gap_index + 1) begin
            @(negedge clk);
            s_valid = 1'b0;
            s_a = '0;
            s_b = '0;
        end
        @(negedge clk);
        s_a = a_value;
        s_b = b_value;
        s_valid = 1'b1;
        while (!s_ready) begin
            @(negedge clk);
        end
        @(posedge clk);
        @(negedge clk);
        s_valid = 1'b0;
        s_a = '0;
        s_b = '0;
    end
endtask
task automatic receive_and_check(input integer signed expected, input integer case_id, input integer length);
    integer hold_index;
    integer signed held_result;
    begin
        m_ready = 1'b0;
        while (!m_valid) begin
            @(negedge clk);
        end
        held_result = $signed(m_result);
        if (held_result !== expected) begin
            mismatch_count = mismatch_count + 1;
            $display("FAIL case=%0d K=%0d rtl=%0d expected=%0d", case_id, length, held_result, expected);
        end else begin
            $display("PASS case=%0d K=%0d result=%0d", case_id, length, held_result);
        end
        for (hold_index = 0; hold_index < 3; hold_index = hold_index + 1) begin
            @(negedge clk);
            if (!m_valid || ($signed(m_result) !== held_result)) begin
                mismatch_count = mismatch_count + 1;
                $display("FAIL backpressure case=%0d cycle=%0d m_valid=%0d result=%0d held=%0d", case_id, hold_index, m_valid, $signed(m_result), held_result);
            end
        end
        m_ready = 1'b1;
        @(posedge clk);
        @(negedge clk);
        m_ready = 1'b0;
        if (m_valid) begin
            mismatch_count = mismatch_count + 1;
            $display("FAIL output handshake case=%0d m_valid did not clear", case_id);
        end
    end
endtask
task automatic run_case(input integer length, input integer case_id);
    integer index;
    integer gap_cycles;
    integer signed a_value;
    integer signed b_value;
    integer signed expected;
    begin
        expected = 0;
        send_command(length);
        for (index = 0; index < length; index = index + 1) begin
            a_value = $urandom_range(0, 255) - 128;
            b_value = $urandom_range(0, 255) - 128;
            gap_cycles = $urandom_range(0, 3);
            expected = expected + a_value * b_value;
            send_pair(a_value, b_value, gap_cycles);
        end
        receive_and_check(expected, case_id, length);
        test_count = test_count + 1;
    end
endtask
integer case_index;
integer random_length;
initial begin
    rst_n = 1'b0;
    cmd_valid = 1'b0;
    cmd_length = 'd1;
    s_valid = 1'b0;
    s_a = '0;
    s_b = '0;
    m_ready = 1'b0;
    test_count = 0;
    mismatch_count = 0;
    seed = 32'h20260826;
    seed = $urandom(seed);
    repeat (4) @(posedge clk);
    @(negedge clk);
    rst_n = 1'b1;
    run_case(1, 1);
    run_case(2, 2);
    run_case(8, 3);
    run_case(MAX_K, 4);
    for (case_index = 5; case_index <= 24; case_index = case_index + 1) begin
        random_length = $urandom_range(1, MAX_K);
        run_case(random_length, case_index);
    end
    $display("--------------------------------------------------");
    $display("test_count     = %0d", test_count);
    $display("mismatch_count = %0d", mismatch_count);
    if ((test_count == 24) && (mismatch_count == 0)) begin
        $display("TEST RESULT: PASS");
    end else begin
        $display("TEST RESULT: FAIL");
    end
    $display("--------------------------------------------------");
    $finish;
end
initial begin
    #200000;
    $display("ERROR: simulation timeout");
    $finish;
end
endmodule
