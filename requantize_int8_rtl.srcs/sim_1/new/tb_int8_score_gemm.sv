// ============================================================================
// TB: tb_int8_score_gemm.sv
// 测:  int8_score_gemm
// 作用: D8 — 读 score_gemm_vectors.txt，Q@K^T INT32 对拍
// ============================================================================

`timescale 1ns/1ps

module tb_int8_score_gemm;
    localparam int INPUT_WIDTH = 8;
    localparam int ACC_WIDTH   = 32;
    localparam int MAX_SEQ     = 4;
    localparam int HEAD_DIM    = 16;
    localparam int SEQ_WIDTH   = $clog2(MAX_SEQ + 1);
    localparam int MAX_ELEMS   = MAX_SEQ * HEAD_DIM;
    localparam int MAX_S_ELEMS = MAX_SEQ * MAX_SEQ;
    localparam int EXPECTED_NUM_CASES = 8;
    localparam int EXPECTED_SEED = 20260901;
    localparam string VECTOR_FILE =
        "F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/score_gemm_vectors.txt";

    logic clk, rst_n;
    logic cmd_valid, cmd_ready;
    logic [SEQ_WIDTH-1:0] cmd_seq;
    logic q_valid, q_ready;
    logic signed [INPUT_WIDTH-1:0] q_data;
    logic k_valid, k_ready;
    logic signed [INPUT_WIDTH-1:0] k_data;
    logic s_valid, s_ready;
    logic signed [ACC_WIDTH-1:0] s_data;
    logic [SEQ_WIDTH-1:0] s_row, s_col;

    integer vector_file, test_count, mismatch_count, header_num_cases, header_seed;
    string line_str;

    int8_score_gemm dut (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready), .cmd_seq(cmd_seq),
        .q_valid(q_valid), .q_ready(q_ready), .q_data(q_data),
        .k_valid(k_valid), .k_ready(k_ready), .k_data(k_data),
        .s_valid(s_valid), .s_ready(s_ready), .s_data(s_data),
        .s_row(s_row), .s_col(s_col)
    );

    task automatic send_cmd(input integer seq_v);
        begin
            @(negedge clk);
            cmd_seq = seq_v[SEQ_WIDTH-1:0];
            cmd_valid = 1'b1;
            while (!cmd_ready) @(negedge clk);
            @(posedge clk); @(negedge clk);
            cmd_valid = 1'b0;
        end
    endtask

    task automatic send_q(input integer signed value, input integer gap);
        integer g;
        begin
            for (g = 0; g < gap; g = g + 1) begin
                @(negedge clk); q_valid = 1'b0; q_data = '0;
            end
            @(negedge clk);
            q_data = value; q_valid = 1'b1;
            while (!q_ready) @(negedge clk);
            @(posedge clk); @(negedge clk);
            q_valid = 1'b0; q_data = '0;
        end
    endtask

    task automatic send_k(input integer signed value, input integer gap);
        integer g;
        begin
            for (g = 0; g < gap; g = g + 1) begin
                @(negedge clk); k_valid = 1'b0; k_data = '0;
            end
            @(negedge clk);
            k_data = value; k_valid = 1'b1;
            while (!k_ready) @(negedge clk);
            @(posedge clk); @(negedge clk);
            k_valid = 1'b0; k_data = '0;
        end
    endtask

    task automatic recv_s(
        input integer erow, input integer ecol, input integer signed expv,
        input integer case_id, input integer hold
    );
        integer h;
        integer signed held;
        integer hr, hc;
        begin
            s_ready = 1'b0;
            while (!s_valid) @(negedge clk);
            held = $signed(s_data); hr = s_row; hc = s_col;
            if ((hr != erow) || (hc != ecol) || (held !== expv)) begin
                mismatch_count = mismatch_count + 1;
                $display("FAIL case=%0d S[%0d][%0d] rtl=%0d/%0d/%0d exp=%0d",
                    case_id, erow, ecol, hr, hc, held, expv);
            end else begin
                $display("PASS case=%0d S[%0d][%0d]=%0d", case_id, erow, ecol, held);
            end
            for (h = 0; h < hold; h = h + 1) begin
                @(negedge clk);
                if (!s_valid || ($signed(s_data) !== held) || (s_row != hr) || (s_col != hc)) begin
                    mismatch_count = mismatch_count + 1;
                    $display("FAIL backpressure case=%0d", case_id);
                end
            end
            s_ready = 1'b1;
            @(posedge clk); @(negedge clk);
            s_ready = 1'b0;
        end
    endtask

    function automatic bit is_marker(input string line);
        is_marker = (line == "Q\n") || (line == "K\n") || (line == "EXPECT\n") ||
                    (line == "END\n") || (line == "END");
    endfunction

    task automatic read_block(
        input integer nexpect,
        output integer signed vals [0:MAX_ELEMS-1],
        output integer nactual,
        output string next_line,
        input integer case_id,
        input string name
    );
        integer sc; integer signed v;
        begin
            nactual = 0; next_line = "";
            while (!$feof(vector_file)) begin
                if ($fgets(line_str, vector_file) == 0) continue;
                if (line_str.len() == 0) continue;
                if (is_marker(line_str)) begin next_line = line_str; break; end
                if (line_str[0] == "#") continue;
                sc = $sscanf(line_str, "%d", v);
                if (sc != 1) begin
                    mismatch_count++; $display("FAIL parse %s case=%0d", name, case_id); return;
                end
                if (nactual >= nexpect) begin
                    mismatch_count++; $display("FAIL extra %s case=%0d", name, case_id); return;
                end
                vals[nactual] = v; nactual++;
            end
            if (nactual != nexpect) begin
                mismatch_count++;
                $display("FAIL %s count case=%0d got=%0d exp=%0d", name, case_id, nactual, nexpect);
            end
        end
    endtask

    task automatic read_expect(
        input integer nexpect,
        output integer signed vals [0:MAX_S_ELEMS-1],
        output integer nactual,
        output string next_line,
        input integer case_id
    );
        integer sc; integer signed v;
        begin
            nactual = 0; next_line = "";
            while (!$feof(vector_file)) begin
                if ($fgets(line_str, vector_file) == 0) continue;
                if (line_str.len() == 0) continue;
                if (is_marker(line_str)) begin next_line = line_str; break; end
                if (line_str[0] == "#") continue;
                sc = $sscanf(line_str, "%d", v);
                if (sc != 1) begin mismatch_count++; return; end
                vals[nactual] = v; nactual++;
                if (nactual > nexpect) begin mismatch_count++; return; end
            end
            if (nactual != nexpect) begin
                mismatch_count++;
                $display("FAIL EXPECT count case=%0d got=%0d exp=%0d", case_id, nactual, nexpect);
            end
        end
    endtask

    task automatic run_case(
        input integer case_id, input integer head_v, input integer seq_v,
        input integer q_gap, input integer k_gap, input integer s_hold
    );
        integer qn, kn, sn, i, r, c;
        integer signed qv [0:MAX_ELEMS-1];
        integer signed kv [0:MAX_ELEMS-1];
        integer signed sv [0:MAX_S_ELEMS-1];
        string marker;
        begin
            $display("CASE %0d head=%0d seq=%0d", case_id, head_v, seq_v);
            read_block(seq_v * HEAD_DIM, qv, qn, marker, case_id, "Q");
            if (qn != seq_v * HEAD_DIM) begin test_count++; return; end
            if (marker != "K\n") begin mismatch_count++; test_count++; return; end
            read_block(seq_v * HEAD_DIM, kv, kn, marker, case_id, "K");
            if (kn != seq_v * HEAD_DIM) begin test_count++; return; end
            if (marker != "EXPECT\n") begin mismatch_count++; test_count++; return; end
            read_expect(seq_v * seq_v, sv, sn, marker, case_id);
            if (sn != seq_v * seq_v) begin test_count++; return; end

            send_cmd(seq_v);
            for (i = 0; i < seq_v * HEAD_DIM; i = i + 1)
                send_q(qv[i], (i == 0) ? 0 : q_gap);
            for (i = 0; i < seq_v * HEAD_DIM; i = i + 1)
                send_k(kv[i], (i == 0) ? 0 : k_gap);
            for (r = 0; r < seq_v; r = r + 1)
                for (c = 0; c < seq_v; c = c + 1)
                    recv_s(r, c, sv[r * seq_v + c], case_id, s_hold);
            test_count++;
        end
    endtask

    initial begin clk = 0; forever #5 clk = ~clk; end

    initial begin
        integer sc, case_id, head_v, seq_v, q_gap, k_gap, s_hold;
        rst_n = 0; cmd_valid = 0; cmd_seq = 1;
        q_valid = 0; q_data = 0; k_valid = 0; k_data = 0; s_ready = 0;
        test_count = 0; mismatch_count = 0; header_num_cases = -1; header_seed = -1;
        vector_file = $fopen(VECTOR_FILE, "r");
        if (vector_file == 0) begin $display("ERROR open %s", VECTOR_FILE); $fatal; end
        $display("D8 Score GEMM regression");
        repeat (4) @(posedge clk); @(negedge clk); rst_n = 1;
        while (!$feof(vector_file)) begin
            if ($fgets(line_str, vector_file) == 0) continue;
            if (line_str.len() == 0) continue;
            if (line_str[0] == "#") begin
                sc = $sscanf(line_str, "# seed=%d num_cases=%d", header_seed, header_num_cases);
                continue;
            end
            if (line_str.substr(0, 3) == "CASE") begin
                sc = $sscanf(line_str, "CASE %d %d %d %d %d %d",
                    case_id, head_v, seq_v, q_gap, k_gap, s_hold);
                if (sc != 6) begin mismatch_count++; continue; end
                if ($fgets(line_str, vector_file) == 0 || line_str != "Q\n") begin
                    mismatch_count++; continue;
                end
                run_case(case_id, head_v, seq_v, q_gap, k_gap, s_hold);
            end
        end
        $fclose(vector_file);
        $display("--------------------------------------------------");
        $display("seed=%0d test_count=%0d mismatch_count=%0d", header_seed, test_count, mismatch_count);
        if ((header_seed == EXPECTED_SEED) && (test_count == EXPECTED_NUM_CASES) &&
            (header_num_cases == EXPECTED_NUM_CASES) && (mismatch_count == 0)) begin
            $display("TEST RESULT: PASS");
            $display("PYTHON AND RTL ARE %0d/%0d EXACT.", test_count, EXPECTED_NUM_CASES);
        end else $display("TEST RESULT: FAIL");
        $finish;
    end

    initial begin #200000000; $display("TIMEOUT"); $finish; end
endmodule
