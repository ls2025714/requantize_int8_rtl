// ============================================================================
// TB: tb_linear_tiled_head0_q.sv
// 测:  int8_linear_tiled
// 作用: D6 — 读 tiled_head0_q_vectors.txt，head0 整层 Q (N=16 K=64)
// ============================================================================

`timescale 1ns/1ps

module tb_linear_tiled_head0_q;
    localparam int INPUT_WIDTH         = 8;
    localparam int ACC_WIDTH           = 32;
    localparam int MAX_M               = 4;
    localparam int FULL_N              = 16;
    localparam int FULL_K              = 64;
    localparam int WEIGHT_ELEMS        = FULL_N * FULL_K;
    localparam int M_WIDTH             = $clog2(MAX_M + 1);
    localparam int N_WIDTH             = $clog2(FULL_N + 1);
    localparam int K_WIDTH             = $clog2(FULL_K + 1);
    localparam int MAX_A_ELEMS         = MAX_M * FULL_K;
    localparam int MAX_C_ELEMS         = MAX_M * FULL_N;
    localparam int MAX_BLOCK_ELEMS     = WEIGHT_ELEMS;
    localparam int TILE_N              = 4;
    localparam int EXPECTED_NUM_CASES  = 4;
    localparam int EXPECTED_SEED       = 20260830;
    localparam string VECTOR_FILE      =
        "F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/tiled_head0_q_vectors.txt";

    logic clk;
    logic rst_n;
    logic cmd_valid;
    logic cmd_ready;
    logic [M_WIDTH-1:0] cmd_m;
    logic [N_WIDTH-1:0] cmd_n;
    logic [K_WIDTH-1:0] cmd_k;
    logic [1:0] cmd_op_type;
    logic [1:0] cmd_head_idx;
    logic a_valid;
    logic a_ready;
    logic signed [INPUT_WIDTH-1:0] a_data;
    logic w_valid;
    logic w_ready;
    logic signed [INPUT_WIDTH-1:0] w_data;
    logic mult_valid;
    logic mult_ready;
    logic [17:0] mult_data;
    logic c_valid;
    logic c_ready;
    logic signed [INPUT_WIDTH-1:0] c_data;
    logic [M_WIDTH-1:0] c_row;
    logic [N_WIDTH-1:0] c_col;
    logic signed [ACC_WIDTH-1:0] acc_debug;

    integer vector_file;
    integer test_count;
    integer mismatch_count;
    integer header_num_cases;
    integer header_seed;
    string line_str;

    int8_linear_tiled #(
        .INPUT_WIDTH(INPUT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .MAX_M(MAX_M),
        .FULL_N(FULL_N),
        .FULL_K(FULL_K)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .cmd_valid(cmd_valid),
        .cmd_ready(cmd_ready),
        .cmd_m(cmd_m),
        .cmd_n(cmd_n),
        .cmd_k(cmd_k),
        .cmd_op_type(cmd_op_type),
        .cmd_head_idx(cmd_head_idx),
        .a_valid(a_valid),
        .a_ready(a_ready),
        .a_data(a_data),
        .w_valid(w_valid),
        .w_ready(w_ready),
        .w_data(w_data),
        .mult_valid(mult_valid),
        .mult_ready(mult_ready),
        .mult_data(mult_data),
        .c_valid(c_valid),
        .c_ready(c_ready),
        .c_data(c_data),
        .c_row(c_row),
        .c_col(c_col),
        .acc_debug(acc_debug)
    );

    task automatic send_command(input integer m_value);
        begin
            @(negedge clk);
            cmd_m = m_value[M_WIDTH-1:0];
            cmd_n = FULL_N[N_WIDTH-1:0];
            cmd_k = FULL_K[K_WIDTH-1:0];
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

    task automatic send_weight(input integer signed value);
        begin
            @(negedge clk);
            w_data = value;
            w_valid = 1'b1;
            while (!w_ready) begin
                @(negedge clk);
            end
            @(posedge clk);
            @(negedge clk);
            w_valid = 1'b0;
            w_data = '0;
        end
    endtask

    task automatic send_mult(input integer value);
        begin
            @(negedge clk);
            mult_data = value[17:0];
            mult_valid = 1'b1;
            while (!mult_ready) begin
                @(negedge clk);
            end
            @(posedge clk);
            @(negedge clk);
            mult_valid = 1'b0;
            mult_data = '0;
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
                    "FAIL case=%0d C[%0d][%0d] rtl_coord=[%0d][%0d] rtl=%0d expected=%0d acc_debug=%0d",
                    case_id, expected_row, expected_col, held_row, held_col, held_data, expected_data, acc_debug
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
                (line == "A\n") || (line == "WEIGHT\n") || (line == "MULT\n") ||
                (line == "EXPECT\n") || (line == "END\n") || (line == "END");
        end
    endfunction

    task automatic read_int_block(
        input integer expected_count,
        output integer signed values [0:MAX_BLOCK_ELEMS-1],
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

    task automatic read_mult_block(
        input integer expected_count,
        output integer values [0:FULL_N-1],
        output integer actual_count,
        output string next_line,
        input integer case_id
    );
        integer scan_count;
        integer val;
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
                    $display("FAIL parse MULT case=%0d line=%s", case_id, line_str);
                    return;
                end
                if (actual_count >= expected_count) begin
                    mismatch_count = mismatch_count + 1;
                    $display("FAIL extra MULT value case=%0d", case_id);
                    return;
                end
                values[actual_count] = val;
                actual_count = actual_count + 1;
            end
            if (actual_count != expected_count) begin
                mismatch_count = mismatch_count + 1;
                $display(
                    "FAIL MULT count case=%0d got=%0d expected=%0d",
                    case_id, actual_count, expected_count
                );
            end
        end
    endtask

    task automatic run_vector_case(
        input integer case_id,
        input integer m_value,
        input integer a_gap,
        input integer c_hold
    );
        integer a_count;
        integer w_count;
        integer mult_count;
        integer c_count;
        integer idx;
        integer i;
        integer j;
        integer n_base;
        integer signed a_vals [0:MAX_BLOCK_ELEMS-1];
        integer signed w_vals [0:MAX_BLOCK_ELEMS-1];
        integer mult_vals [0:FULL_N-1];
        integer signed c_vals [0:MAX_BLOCK_ELEMS-1];
        string marker_line;
        begin
            read_int_block(m_value * FULL_K, a_vals, a_count, marker_line, case_id, "A");
            if (a_count != m_value * FULL_K) begin
                test_count = test_count + 1;
                return;
            end
            if (marker_line != "WEIGHT\n") begin
                mismatch_count = mismatch_count + 1;
                $display("FAIL missing WEIGHT marker case=%0d got=%s", case_id, marker_line);
                test_count = test_count + 1;
                return;
            end
            read_int_block(WEIGHT_ELEMS, w_vals, w_count, marker_line, case_id, "WEIGHT");
            if (w_count != WEIGHT_ELEMS) begin
                test_count = test_count + 1;
                return;
            end
            if (marker_line != "MULT\n") begin
                mismatch_count = mismatch_count + 1;
                $display("FAIL missing MULT marker case=%0d got=%s", case_id, marker_line);
                test_count = test_count + 1;
                return;
            end
            read_mult_block(FULL_N, mult_vals, mult_count, marker_line, case_id);
            if (mult_count != FULL_N) begin
                test_count = test_count + 1;
                return;
            end
            if (marker_line != "EXPECT\n") begin
                mismatch_count = mismatch_count + 1;
                $display("FAIL missing EXPECT marker case=%0d got=%s", case_id, marker_line);
                test_count = test_count + 1;
                return;
            end
            read_int_block(m_value * FULL_N, c_vals, c_count, marker_line, case_id, "EXPECT");
            if (c_count != m_value * FULL_N) begin
                test_count = test_count + 1;
                return;
            end
            if ((marker_line != "END\n") && (marker_line != "END")) begin
                mismatch_count = mismatch_count + 1;
                $display("FAIL missing END marker case=%0d got=%s", case_id, marker_line);
                test_count = test_count + 1;
                return;
            end

            for (idx = 0; idx < WEIGHT_ELEMS; idx = idx + 1) begin
                send_weight(w_vals[idx]);
            end
            for (idx = 0; idx < FULL_N; idx = idx + 1) begin
                send_mult(mult_vals[idx]);
            end
            send_command(m_value);
            for (idx = 0; idx < m_value * FULL_K; idx = idx + 1) begin
                send_a(a_vals[idx], (idx == 0) ? 0 : a_gap);
            end
            // RTL 按 N-tile 输出：每个 tile 先吐完 M×4（全局 col = n_base+local）
            for (n_base = 0; n_base < FULL_N; n_base = n_base + TILE_N) begin
                for (i = 0; i < m_value; i = i + 1) begin
                    for (j = 0; j < TILE_N; j = j + 1) begin
                        receive_c(
                            i,
                            n_base + j,
                            c_vals[i * FULL_N + n_base + j],
                            case_id,
                            c_hold
                        );
                    end
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
        integer a_gap;
        integer c_hold;

        rst_n = 1'b0;
        cmd_valid = 1'b0;
        cmd_m = 'd1;
        cmd_n = FULL_N[N_WIDTH-1:0];
        cmd_k = FULL_K[K_WIDTH-1:0];
        cmd_op_type = 2'b00;
        cmd_head_idx = 2'b00;
        a_valid = 1'b0;
        a_data = '0;
        w_valid = 1'b0;
        w_data = '0;
        mult_valid = 1'b0;
        mult_data = '0;
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

        $display("D6 Tiled head0 Q regression");
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
                    "CASE %d %d %d %d",
                    case_id, m_value, a_gap, c_hold
                );
                if (scan_count != 4) begin
                    mismatch_count = mismatch_count + 1;
                    $display("FAIL parse CASE line: %s", line_str);
                    continue;
                end
                if ($fgets(line_str, vector_file) == 0 || line_str != "A\n") begin
                    mismatch_count = mismatch_count + 1;
                    $display("FAIL missing A marker case=%0d", case_id);
                    continue;
                end
                run_vector_case(case_id, m_value, a_gap, c_hold);
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
        #500000000;
        $display("ERROR: simulation timeout");
        $display("test_count=%0d mismatch_count=%0d", test_count, mismatch_count);
        $finish;
    end
endmodule
