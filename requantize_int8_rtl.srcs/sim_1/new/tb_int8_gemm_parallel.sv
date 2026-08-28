// ============================================================================
// TB: tb_int8_gemm_parallel.sv
// 测:  int8_gemm_parallel
// 作用: D3 并行 GEMM 24 case 内联回归，INT32 期望
// ============================================================================

// D3 内联 24 case，与 tb_int8_gemm_serial 同口径，DUT 为 int8_gemm_parallel
`timescale 1ns/1ps
module tb_int8_gemm_parallel;
localparam int INPUT_WIDTH = 8;
localparam int ACC_WIDTH = 32;
localparam int MAX_M = 4;
localparam int MAX_N = 4;
localparam int MAX_K = 16;
localparam int M_WIDTH = $clog2(MAX_M + 1);
localparam int N_WIDTH = $clog2(MAX_N + 1);
localparam int K_WIDTH = $clog2(MAX_K + 1);
logic clk;
logic rst_n;
logic cmd_valid;
logic cmd_ready;
logic [M_WIDTH-1:0] cmd_m;
logic [N_WIDTH-1:0] cmd_n;
logic [K_WIDTH-1:0] cmd_k;
logic a_valid;
logic a_ready;
logic signed [INPUT_WIDTH-1:0] a_data;
logic b_valid;
logic b_ready;
logic signed [INPUT_WIDTH-1:0] b_data;
logic c_valid;
logic c_ready;
logic signed [ACC_WIDTH-1:0] c_data;
logic [M_WIDTH-1:0] c_row;
logic [N_WIDTH-1:0] c_col;
integer signed a_ref [0:MAX_M*MAX_K-1];
integer signed b_ref [0:MAX_K*MAX_N-1];
integer signed c_ref [0:MAX_M*MAX_N-1];
integer test_count;
integer mismatch_count;
integer seed;
int8_gemm_parallel #(
    .INPUT_WIDTH(INPUT_WIDTH),
    .ACC_WIDTH(ACC_WIDTH),
    .MAX_M(MAX_M),
    .MAX_N(MAX_N),
    .MAX_K(MAX_K),
    .M_WIDTH(M_WIDTH),
    .N_WIDTH(N_WIDTH),
    .K_WIDTH(K_WIDTH)
) dut (
    .clk(clk),
    .rst_n(rst_n),
    .cmd_valid(cmd_valid),
    .cmd_ready(cmd_ready),
    .cmd_m(cmd_m),
    .cmd_n(cmd_n),
    .cmd_k(cmd_k),
    .a_valid(a_valid),
    .a_ready(a_ready),
    .a_data(a_data),
    .b_valid(b_valid),
    .b_ready(b_ready),
    .b_data(b_data),
    .c_valid(c_valid),
    .c_ready(c_ready),
    .c_data(c_data),
    .c_row(c_row),
    .c_col(c_col)
);
initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
end
task automatic send_command(input integer m_value, input integer n_value, input integer k_value);
    begin
        @(negedge clk);
        cmd_m = m_value[M_WIDTH-1:0];
        cmd_n = n_value[N_WIDTH-1:0];
        cmd_k = k_value[K_WIDTH-1:0];
        cmd_valid = 1'b1;
        while (!cmd_ready) begin
            @(negedge clk);
        end
        @(posedge clk);
        @(negedge clk);
        cmd_valid = 1'b0;
    end
endtask
task automatic send_a(input integer signed value, input integer gap_cycles);
    integer gap_index;
    begin
        for (gap_index = 0; gap_index < gap_cycles; gap_index = gap_index + 1) begin
            @(negedge clk);
            a_valid = 1'b0;
            a_data = '0;
        end
        @(negedge clk);
        a_data = value;
        a_valid = 1'b1;
        while (!a_ready) begin
            @(negedge clk);
        end
        @(posedge clk);
        @(negedge clk);
        a_valid = 1'b0;
        a_data = '0;
    end
endtask
task automatic send_b(input integer signed value, input integer gap_cycles);
    integer gap_index;
    begin
        for (gap_index = 0; gap_index < gap_cycles; gap_index = gap_index + 1) begin
            @(negedge clk);
            b_valid = 1'b0;
            b_data = '0;
        end
        @(negedge clk);
        b_data = value;
        b_valid = 1'b1;
        while (!b_ready) begin
            @(negedge clk);
        end
        @(posedge clk);
        @(negedge clk);
        b_valid = 1'b0;
        b_data = '0;
    end
endtask
task automatic receive_c(input integer expected_row, input integer expected_col, input integer signed expected_data, input integer case_id);
    integer hold_cycles;
    integer hold_index;
    integer signed held_data;
    integer held_row;
    integer held_col;
    begin
        c_ready = 1'b0;
        while (!c_valid) begin
            @(negedge clk);
        end
        held_data = $signed(c_data);
        held_row = c_row;
        held_col = c_col;
        if ((held_row != expected_row) || (held_col != expected_col) || (held_data !== expected_data)) begin
            mismatch_count = mismatch_count + 1;
            $display("FAIL case=%0d C[%0d][%0d] rtl_coord=[%0d][%0d] rtl=%0d expected=%0d", case_id, expected_row, expected_col, held_row, held_col, held_data, expected_data);
        end else begin
            $display("PASS case=%0d C[%0d][%0d]=%0d", case_id, expected_row, expected_col, held_data);
        end
        hold_cycles = $urandom_range(0, 3);
        for (hold_index = 0; hold_index < hold_cycles; hold_index = hold_index + 1) begin
            @(negedge clk);
            if (!c_valid || ($signed(c_data) !== held_data) || (c_row != held_row) || (c_col != held_col)) begin
                mismatch_count = mismatch_count + 1;
                $display("FAIL backpressure case=%0d C[%0d][%0d]", case_id, expected_row, expected_col);
            end
        end
        c_ready = 1'b1;
        @(posedge clk);
        @(negedge clk);
        c_ready = 1'b0;
    end
endtask
task automatic compute_reference(input integer m_value, input integer n_value, input integer k_value);
    integer i;
    integer j;
    integer k;
    integer signed sum;
    begin
        for (i = 0; i < m_value; i = i + 1) begin
            for (j = 0; j < n_value; j = j + 1) begin
                sum = 0;
                for (k = 0; k < k_value; k = k + 1) begin
                    sum = sum + a_ref[i*k_value+k] * b_ref[k*n_value+j];
                end
                c_ref[i*n_value+j] = sum;
            end
        end
    end
endtask
task automatic execute_loaded_case(input integer m_value, input integer n_value, input integer k_value, input integer case_id);
    integer index;
    integer i;
    integer j;
    begin
        compute_reference(m_value, n_value, k_value);
        send_command(m_value, n_value, k_value);
        for (index = 0; index < m_value*k_value; index = index + 1) begin
            send_a(a_ref[index], $urandom_range(0, 2));
        end
        for (index = 0; index < k_value*n_value; index = index + 1) begin
            send_b(b_ref[index], $urandom_range(0, 2));
        end
        for (i = 0; i < m_value; i = i + 1) begin
            for (j = 0; j < n_value; j = j + 1) begin
                receive_c(i, j, c_ref[i*n_value+j], case_id);
            end
        end
        test_count = test_count + 1;
    end
endtask
task automatic run_directed_case;
    begin
        a_ref[0] = 1;
        a_ref[1] = 2;
        a_ref[2] = 3;
        a_ref[3] = 4;
        a_ref[4] = 5;
        a_ref[5] = 6;
        b_ref[0] = 7;
        b_ref[1] = 8;
        b_ref[2] = 9;
        b_ref[3] = 10;
        b_ref[4] = 11;
        b_ref[5] = 12;
        execute_loaded_case(2, 2, 3, 1);
    end
endtask
task automatic run_random_case(input integer m_value, input integer n_value, input integer k_value, input integer case_id);
    integer index;
    begin
        for (index = 0; index < m_value*k_value; index = index + 1) begin
            a_ref[index] = $urandom_range(0, 255) - 128;
        end
        for (index = 0; index < k_value*n_value; index = index + 1) begin
            b_ref[index] = $urandom_range(0, 255) - 128;
        end
        execute_loaded_case(m_value, n_value, k_value, case_id);
    end
endtask
integer case_index;
integer random_m;
integer random_n;
integer random_k;
initial begin
    rst_n = 1'b0;
    cmd_valid = 1'b0;
    cmd_m = 'd1;
    cmd_n = 'd1;
    cmd_k = 'd1;
    a_valid = 1'b0;
    a_data = '0;
    b_valid = 1'b0;
    b_data = '0;
    c_ready = 1'b0;
    test_count = 0;
    mismatch_count = 0;
    seed = 32'h20260827;
    seed = $urandom(seed);
    repeat (4) @(posedge clk);
    @(negedge clk);
    rst_n = 1'b1;
    run_directed_case();
    run_random_case(1, 1, 1, 2);
    run_random_case(2, 3, 4, 3);
    run_random_case(MAX_M, MAX_N, MAX_K, 4);
    for (case_index = 5; case_index <= 24; case_index = case_index + 1) begin
        random_m = $urandom_range(1, MAX_M);
        random_n = $urandom_range(1, MAX_N);
        random_k = $urandom_range(1, MAX_K);
        run_random_case(random_m, random_n, random_k, case_index);
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
    #5000000;
    $display("ERROR: simulation timeout");
    $finish;
end
endmodule
