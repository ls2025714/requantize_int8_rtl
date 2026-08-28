// ============================================================================
// TB: tb_gemm_parallel_python_vectors.sv
// 测:  int8_gemm_parallel
// 作用: D3 — 读 gemm_parallel_vectors.txt，24/24 INT32 对拍
// ============================================================================

// D3 Python 向量: gemm_parallel_vectors.txt
// task: send_command, send_a, send_b, receive_c（INT32 期望）
`timescale 1ns/1ps

module tb_gemm_parallel_python_vectors;
    localparam int INPUT_WIDTH         = 8;
    localparam int ACC_WIDTH           = 32;
    localparam int MAX_M               = 4;
    localparam int MAX_N               = 4;
    localparam int MAX_K               = 16;
    localparam int M_WIDTH             = $clog2(MAX_M + 1);
    localparam int N_WIDTH             = $clog2(MAX_N + 1);
    localparam int K_WIDTH             = $clog2(MAX_K + 1);
    localparam int MAX_A_ELEMS         = MAX_M * MAX_K;
    localparam int MAX_B_ELEMS         = MAX_K * MAX_N;
    localparam int MAX_C_ELEMS         = MAX_M * MAX_N;
    localparam int EXPECTED_NUM_CASES  = 24;
    localparam int EXPECTED_SEED       = 20260827;
    localparam string VECTOR_FILE      =
        "F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/gemm_parallel_vectors.txt";

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

    integer vector_file;
    integer test_count;
    integer mismatch_count;
    integer header_num_cases;
    integer header_seed;
    string line_str;

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

    // -------------------------------------------------------------------------
    // TB 驱动 task：cmd → LOAD_A → LOAD_B → 收 INT32 C[row][col]
    // -------------------------------------------------------------------------
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

    task automatic receive_c(
        input integer expected_row,
        input integer expected_col,
        input integer signed expected_data,
        input integer case_id,
        input integer hold_cycles
    );
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
                $display(
                    "FAIL case=%0d C[%0d][%0d] rtl_coord=[%0d][%0d] rtl=%0d expected=%0d",
                    case_id, expected_row, expected_col, held_row, held_col, held_data, expected_data
                );
            end else begin
                $display("PASS case=%0d C[%0d][%0d]=%0d", case_id, expected_row, expected_col, held_data);
            end
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

    function automatic bit is_section_marker(input string line);
        begin
            is_section_marker =
                (line == "A\n") || (line == "B\n") || (line == "EXPECT\n") ||
                (line == "END\n") || (line == "END");
        end
    endfunction

    task automatic read_int_block(
        input integer expected_count,
        output integer signed values [0:MAX_A_ELEMS-1],
        output integer actual_count,
        output string next_line,
        input integer case_id,
        input string block_name
    );
        integer scan_count;
        integer signed val;
        begin
            actual_count = 0;
            next_line = "";
            while (!$feof(vector_file)) begin
                if ($fgets(line_str, vector_file) == 0) begin
                    continue;
                end
                if (line_str.len() == 0) begin
                    continue;
                end
                if (is_section_marker(line_str)) begin
                    next_line = line_str;
                    break;
                end
                scan_count = $sscanf(line_str, "%d", val);
                if (scan_count != 1) begin
                    mismatch_count = mismatch_count + 1;
                    $display("FAIL parse %s case=%0d line=%s", block_name, case_id, line_str);
                    return;
                end
                if (actual_count >= expected_count) begin
                    mismatch_count = mismatch_count + 1;
                    $display("FAIL extra %s value case=%0d", block_name, case_id);
                    return;
                end
                values[actual_count] = val;
                actual_count = actual_count + 1;
            end
            if (actual_count != expected_count) begin
                mismatch_count = mismatch_count + 1;
                $display(
                    "FAIL %s count case=%0d got=%0d expected=%0d",
                    block_name, case_id, actual_count, expected_count
                );
            end
        end
    endtask

    task automatic run_vector_case(
        input integer case_id,
        input integer m_value,
        input integer n_value,
        input integer k_value,
        input integer a_gap,
        input integer b_gap,
        input integer c_hold
    );
        integer a_count;
        integer b_count;
        integer c_count;
        integer idx;
        integer i;
        integer j;
        integer signed a_vals [0:MAX_A_ELEMS-1];
        integer signed b_vals [0:MAX_A_ELEMS-1];
        integer signed c_vals [0:MAX_A_ELEMS-1];
        string marker_line;
        begin
            read_int_block(m_value * k_value, a_vals, a_count, marker_line, case_id, "A");
            if (a_count != m_value * k_value) begin
                test_count = test_count + 1;
                return;
            end
            if (marker_line != "B\n") begin
                mismatch_count = mismatch_count + 1;
                $display("FAIL missing B marker case=%0d got=%s", case_id, marker_line);
                test_count = test_count + 1;
                return;
            end
            read_int_block(k_value * n_value, b_vals, b_count, marker_line, case_id, "B");
            if (b_count != k_value * n_value) begin
                test_count = test_count + 1;
                return;
            end
            if (marker_line != "EXPECT\n") begin
                mismatch_count = mismatch_count + 1;
                $display("FAIL missing EXPECT marker case=%0d got=%s", case_id, marker_line);
                test_count = test_count + 1;
                return;
            end
            read_int_block(m_value * n_value, c_vals, c_count, marker_line, case_id, "EXPECT");
            if (c_count != m_value * n_value) begin
                test_count = test_count + 1;
                return;
            end
            if ((marker_line != "END\n") && (marker_line != "END")) begin
                mismatch_count = mismatch_count + 1;
                $display("FAIL missing END marker case=%0d got=%s", case_id, marker_line);
                test_count = test_count + 1;
                return;
            end

            send_command(m_value, n_value, k_value);
            for (idx = 0; idx < m_value * k_value; idx = idx + 1) begin
                send_a(a_vals[idx], (idx == 0) ? 0 : a_gap);
            end
            for (idx = 0; idx < k_value * n_value; idx = idx + 1) begin
                send_b(b_vals[idx], (idx == 0) ? 0 : b_gap);
            end
            for (i = 0; i < m_value; i = i + 1) begin
                for (j = 0; j < n_value; j = j + 1) begin
                    receive_c(i, j, c_vals[i * n_value + j], case_id, c_hold);
                end
            end
            test_count = test_count + 1;
        end
    endtask

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        integer scan_count;
        integer case_id;
        integer m_value;
        integer n_value;
        integer k_value;
        integer a_gap;
        integer b_gap;
        integer c_hold;

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
        header_num_cases = -1;
        header_seed = -1;

        vector_file = $fopen(VECTOR_FILE, "r");
        if (vector_file == 0) begin
            $display("ERROR: cannot open vector file");
            $display("FILE: %s", VECTOR_FILE);
            $fatal;
        end

        $display("Parallel GEMM python-vector regression");
        $display("Vector file: %s", VECTOR_FILE);

        repeat (4) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        while (!$feof(vector_file)) begin
            if ($fgets(line_str, vector_file) == 0) begin
                continue;
            end
            if (line_str.len() == 0) begin
                continue;
            end
            if (line_str[0] == "#") begin
                scan_count = $sscanf(line_str, "# seed=%d num_cases=%d", header_seed, header_num_cases);
                continue;
            end
            if (line_str.substr(0, 3) == "CASE") begin
                scan_count = $sscanf(
                    line_str,
                    "CASE %d %d %d %d %d %d %d",
                    case_id, m_value, n_value, k_value, a_gap, b_gap, c_hold
                );
                if (scan_count != 7) begin
                    mismatch_count = mismatch_count + 1;
                    $display("FAIL parse CASE line: %s", line_str);
                    continue;
                end
                if ($fgets(line_str, vector_file) == 0 || line_str != "A\n") begin
                    mismatch_count = mismatch_count + 1;
                    $display("FAIL missing A marker case=%0d", case_id);
                    continue;
                end
                run_vector_case(case_id, m_value, n_value, k_value, a_gap, b_gap, c_hold);
            end
        end

        $fclose(vector_file);

        $display("--------------------------------------------------");
        $display("seed            = %0d", header_seed);
        $display("expected_cases  = %0d", EXPECTED_NUM_CASES);
        $display("header_cases    = %0d", header_num_cases);
        $display("test_count      = %0d", test_count);
        $display("mismatch_count  = %0d", mismatch_count);
        if ((header_seed == EXPECTED_SEED) &&
            (header_num_cases == EXPECTED_NUM_CASES) &&
            (test_count == EXPECTED_NUM_CASES) &&
            (mismatch_count == 0)) begin
            $display("TEST RESULT: PASS");
            $display("PYTHON AND RTL ARE %0d/%0d EXACT.", test_count, EXPECTED_NUM_CASES);
        end else begin
            $display("TEST RESULT: FAIL");
        end
        $display("--------------------------------------------------");
        $finish;
    end

    initial begin
        #10000000;
        $display("ERROR: simulation timeout");
        $display("test_count=%0d mismatch_count=%0d", test_count, mismatch_count);
        $finish;
    end
endmodule
