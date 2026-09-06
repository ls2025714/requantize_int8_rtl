// ============================================================================
// 文件: tb_int8_softmax_causal.sv
// 阶段: D9
// 作用: 对拍因果 Softmax（scale + mask + 定点 P）
// DUT: int8_softmax_causal
// 向量: softmax_causal_vectors.txt（seed=20260902，6 case）
// CLI: scripts/run_softmax_causal_xsim.bat
// ============================================================================
//
// 流程: send_cmd(seq) → stream score(in_row/col) → receive out_data/out_row/out_col 对拍
//
`timescale 1ns/1ps

module tb_int8_softmax_causal;
    localparam int ACC_WIDTH = 32;
    localparam int MAX_SEQ = 4;
    localparam int P_WIDTH = 16;
    localparam int SEQ_WIDTH = $clog2(MAX_SEQ + 1);
    localparam int MAX_ELEMS = MAX_SEQ * MAX_SEQ;
    localparam int EXPECTED_NUM_CASES = 6;
    localparam int EXPECTED_SEED = 20260902;
    localparam string VECTOR_FILE =
        "F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/softmax_causal_vectors.txt";

    logic clk, rst_n;
    logic cmd_valid, cmd_ready;
    logic [SEQ_WIDTH-1:0] cmd_seq;
    logic in_valid, in_ready;
    logic signed [ACC_WIDTH-1:0] in_data;
    logic [SEQ_WIDTH-1:0] in_row, in_col;
    logic out_valid, out_ready;
    logic [P_WIDTH-1:0] out_data;
    logic [SEQ_WIDTH-1:0] out_row, out_col;

    integer vector_file, test_count, mismatch_count, header_seed, header_cases;
    string line_str;

    int8_softmax_causal dut (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready), .cmd_seq(cmd_seq),
        .in_valid(in_valid), .in_ready(in_ready), .in_data(in_data),
        .in_row(in_row), .in_col(in_col),
        .out_valid(out_valid), .out_ready(out_ready), .out_data(out_data),
        .out_row(out_row), .out_col(out_col)
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

    task automatic send_score(input integer signed val, input integer r, input integer c);
        begin
            @(negedge clk);
            in_data = val; in_row = r[SEQ_WIDTH-1:0]; in_col = c[SEQ_WIDTH-1:0];
            in_valid = 1'b1;
            while (!in_ready) @(negedge clk);
            @(posedge clk); @(negedge clk);
            in_valid = 1'b0;
        end
    endtask

    task automatic recv_p(input integer er, input integer ec, input integer expv,
                          input integer case_id, input integer hold);
        integer h; integer held; integer hr, hc;
        begin
            out_ready = 1'b0;
            while (!out_valid) @(negedge clk);
            held = out_data; hr = out_row; hc = out_col;
            if ((hr != er) || (hc != ec) || (held !== expv)) begin
                mismatch_count++;
                $display("FAIL case=%0d P[%0d][%0d] rtl=%0d/%0d/%0d exp=%0d",
                    case_id, er, ec, hr, hc, held, expv);
            end else $display("PASS case=%0d P[%0d][%0d]=%0d", case_id, er, ec, held);
            for (h = 0; h < hold; h++) begin
                @(negedge clk);
                if (!out_valid || (out_data !== held)) begin
                    mismatch_count++; $display("FAIL hold case=%0d", case_id);
                end
            end
            out_ready = 1'b1;
            @(posedge clk); @(negedge clk);
            out_ready = 1'b0;
        end
    endtask

    function automatic bit is_marker(input string line);
        is_marker = (line == "SCORE\n") || (line == "EXPECT\n") || (line == "END\n") || (line == "END");
    endfunction

    task automatic read_ints(input integer nexp, output integer signed vals[0:MAX_ELEMS-1],
                             output integer nact, output string next_line, input string name);
        integer sc; integer signed v;
        begin
            nact = 0; next_line = "";
            while (!$feof(vector_file)) begin
                if ($fgets(line_str, vector_file) == 0) continue;
                if (line_str.len() == 0) continue;
                if (line_str[0] == "#") continue;
                if (is_marker(line_str)) begin next_line = line_str; break; end
                sc = $sscanf(line_str, "%d", v);
                if (sc != 1) begin mismatch_count++; return; end
                if (nact >= nexp) begin
                    mismatch_count++;
                    $display("FAIL extra %s", name);
                    return;
                end
                vals[nact] = v; nact++;
            end
            if (nact != nexp) begin
                mismatch_count++;
                $display("FAIL %s count got=%0d exp=%0d", name, nact, nexp);
            end
        end
    endtask

    task automatic run_case(input integer case_id, input integer seq_v, input integer hold);
        integer sn, en, i, r, c;
        integer signed sv[0:MAX_ELEMS-1];
        integer signed ev[0:MAX_ELEMS-1];
        string marker;
        begin
            $display("CASE %0d seq=%0d", case_id, seq_v);
            read_ints(seq_v * seq_v, sv, sn, marker, "SCORE");
            if (sn != seq_v * seq_v) begin test_count++; return; end
            if (marker != "EXPECT\n") begin
                // consume until EXPECT if needed
                if ($fgets(line_str, vector_file) == 0 || line_str != "EXPECT\n") begin
                    // try reading again after SCORE values stopped early
                end
            end
            // After read_ints hits EXPECT marker it stops; marker should be EXPECT
            if (marker != "EXPECT\n") begin
                mismatch_count++; test_count++; return;
            end
            read_ints(seq_v * seq_v, ev, en, marker, "EXPECT");
            if (en != seq_v * seq_v) begin test_count++; return; end

            send_cmd(seq_v);
            for (r = 0; r < seq_v; r++)
                for (c = 0; c < seq_v; c++)
                    send_score(sv[r * seq_v + c], r, c);
            for (r = 0; r < seq_v; r++)
                for (c = 0; c < seq_v; c++)
                    recv_p(r, c, ev[r * seq_v + c], case_id, hold);
            test_count++;
        end
    endtask

    initial begin clk = 0; forever #5 clk = ~clk; end

    initial begin
        integer sc, case_id, seq_v, hold;
        rst_n = 0; cmd_valid = 0; cmd_seq = 1;
        in_valid = 0; in_data = 0; in_row = 0; in_col = 0; out_ready = 0;
        test_count = 0; mismatch_count = 0; header_seed = -1; header_cases = -1;
        vector_file = $fopen(VECTOR_FILE, "r");
        if (!vector_file) begin $display("ERROR open vector"); $fatal; end
        $display("D9 Softmax causal regression");
        repeat (4) @(posedge clk); @(negedge clk); rst_n = 1;
        while (!$feof(vector_file)) begin
            if ($fgets(line_str, vector_file) == 0) continue;
            if (line_str.len() == 0) continue;
            if (line_str[0] == "#") begin
                sc = $sscanf(line_str, "# seed=%d num_cases=%d", header_seed, header_cases);
                continue;
            end
            if (line_str.substr(0, 3) == "CASE") begin
                sc = $sscanf(line_str, "CASE %d %d %d", case_id, seq_v, hold);
                if (sc != 3) begin mismatch_count++; continue; end
                if ($fgets(line_str, vector_file) == 0 || line_str != "SCORE\n") begin
                    mismatch_count++; continue;
                end
                run_case(case_id, seq_v, hold);
            end
        end
        $fclose(vector_file);
        $display("--------------------------------------------------");
        $display("seed=%0d test_count=%0d mismatch_count=%0d", header_seed, test_count, mismatch_count);
        if ((header_seed == EXPECTED_SEED) && (test_count == EXPECTED_NUM_CASES) &&
            (header_cases == EXPECTED_NUM_CASES) && (mismatch_count == 0)) begin
            $display("TEST RESULT: PASS");
            $display("PYTHON AND RTL ARE %0d/%0d EXACT.", test_count, EXPECTED_NUM_CASES);
        end else $display("TEST RESULT: FAIL");
        $finish;
    end
    initial begin #500000000; $display("TIMEOUT"); $finish; end
endmodule
