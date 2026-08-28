// ============================================================================
// TB: tb_int8_transformer_block.sv — D12 E2E block (seq=1)
// ============================================================================
`timescale 1ns/1ps

module tb_int8_transformer_block;
    localparam int EXPECTED_SEED      = 20260908;
    localparam int EXPECTED_NUM_CASES = 1;
    localparam int EMBD               = 64;
    localparam int QKV_W              = 12288;
    localparam int QKV_M              = 192;
    localparam int WO_W               = 4096;
    localparam int WO_M               = 64;
    localparam int WIDE_W             = 16384;
    localparam int WIDE_M             = 256;
    localparam int DOWN_W             = 16384;
    localparam int DOWN_M             = 64;
    localparam string VECTOR_FILE     =
        "F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/transformer_block_vectors.txt";

    logic clk, rst_n;
    logic cmd_valid, cmd_ready;
    logic [2:0] cmd_seq;
    logic w_valid, w_ready;
    logic signed [7:0] w_data;
    logic mult_valid, mult_ready;
    logic [17:0] mult_data;
    logic x_valid, x_ready;
    logic signed [7:0] x_data;
    logic out_valid, out_ready;
    logic signed [7:0] out_data;
    logic [9:0] out_idx;

    integer vf, tc, mc, hseed, hcases;
    string line;

    int8_transformer_block dut (.*);

    task send_w(input int signed v);
        begin @(negedge clk); w_data = v; w_valid = 1'b1;
            while (!w_ready) @(negedge clk);
            @(posedge clk); @(negedge clk); w_valid = 1'b0; end
    endtask

    task send_mult(input int v);
        begin @(negedge clk); mult_data = v[17:0]; mult_valid = 1'b1;
            while (!mult_ready) @(negedge clk);
            @(posedge clk); @(negedge clk); mult_valid = 1'b0; end
    endtask

    task send_x(input int signed v);
        begin @(negedge clk); x_data = v; x_valid = 1'b1;
            while (!x_ready) @(negedge clk);
            @(posedge clk); @(negedge clk); x_valid = 1'b0; end
    endtask

    task recv_out(input int ei, input int signed ev, input int cid);
        begin out_ready = 1'b0;
            while (!out_valid) @(negedge clk);
            if (out_idx !== ei || $signed(out_data) !== ev) begin
                mc++; $display("FAIL c=%0d i=%0d rtl=%0d exp=%0d", cid, ei, out_data, ev);
            end else $display("PASS c=%0d i=%0d out=%0d", cid, ei, out_data);
            out_ready = 1'b1; @(posedge clk); @(negedge clk); out_ready = 1'b0;
        end
    endtask

    function automatic bit is_marker(input string s, input string tag);
        is_marker = (s.len() >= tag.len()) && (s.substr(0, tag.len() - 1) == tag);
    endfunction

    task read_int_lines(input integer n, input bit is_weight, input bit is_mult, input string tag);
        integer i, sc, v;
        begin
            $display("Preload %s (%0d items)...", tag, n);
            for (i = 0; i < n; i = i + 1) begin
                if ($fgets(line, vf) == 0) begin mc++; return; end
                sc = $sscanf(line, "%d", v);
                if (sc != 1) begin mc++; continue; end
                if (is_weight) send_w(v);
                else if (is_mult) send_mult(v);
                if ((i + 1) % 4096 == 0) $display("  %s %0d/%0d", tag, i + 1, n);
            end
            $display("Preload %s done.", tag);
        end
    endtask

    task run_case(input int cid, input int seq);
        integer i, sc, v;
        integer signed xv[0:255];
        integer signed ev[0:255];
        begin
            if ($fgets(line, vf) == 0 || !is_marker(line, "X")) begin mc++; return; end
            for (i = 0; i < seq * EMBD; i = i + 1) begin
                if ($fgets(line, vf) == 0) begin mc++; return; end
                sc = $sscanf(line, "%d", xv[i]);
                if (sc != 1) mc++;
            end
            if ($fgets(line, vf) == 0 || !is_marker(line, "EXPECT")) begin mc++; return; end
            for (i = 0; i < seq * EMBD; i = i + 1) begin
                if ($fgets(line, vf) == 0) begin mc++; return; end
                sc = $sscanf(line, "%d", ev[i]);
                if (sc != 1) mc++;
            end
            if ($fgets(line, vf) == 0 || !is_marker(line, "END")) mc++;

            @(negedge clk);
            cmd_seq = seq[2:0]; cmd_valid = 1'b1;
            while (!cmd_ready) @(negedge clk);
            @(posedge clk); @(negedge clk); cmd_valid = 1'b0;

            for (i = 0; i < seq * EMBD; i = i + 1)
                send_x(xv[i]);

            for (i = 0; i < seq * EMBD; i = i + 1)
                recv_out(i, ev[i], cid);

            tc++;
        end
    endtask

    initial clk = 0;
    always #5 clk = ~clk;

    initial begin
        integer cid, seq;
        tc = 0; mc = 0; hseed = 0; hcases = 0;
        rst_n = 0; cmd_valid = 0; w_valid = 0; mult_valid = 0; x_valid = 0; out_ready = 0;
        vf = $fopen(VECTOR_FILE, "r");
        repeat(4) @(posedge clk);
        rst_n = 1;

        while (!$feof(vf)) begin
            if ($fgets(line, vf) == 0) continue;
            if (line.len() == 0) continue;
            if (line[0] == "#") begin
                void'($sscanf(line, "# seed=%d num_cases=%d", hseed, hcases));
                continue;
            end
            if (is_marker(line, "PRELOAD_QKV_WEIGHT"))
                read_int_lines(QKV_W, 1, 0, "QKV_WEIGHT");
            else if (is_marker(line, "PRELOAD_QKV_MULT"))
                read_int_lines(QKV_M, 0, 1, "QKV_MULT");
            else if (is_marker(line, "PRELOAD_WO_WEIGHT"))
                read_int_lines(WO_W, 1, 0, "WO_WEIGHT");
            else if (is_marker(line, "PRELOAD_WO_MULT"))
                read_int_lines(WO_M, 0, 1, "WO_MULT");
            else if (is_marker(line, "PRELOAD_FFN_GATE_WEIGHT"))
                read_int_lines(WIDE_W, 1, 0, "FFN_GATE_WEIGHT");
            else if (is_marker(line, "PRELOAD_FFN_GATE_MULT"))
                read_int_lines(WIDE_M, 0, 1, "FFN_GATE_MULT");
            else if (is_marker(line, "PRELOAD_FFN_UP_WEIGHT"))
                read_int_lines(WIDE_W, 1, 0, "FFN_UP_WEIGHT");
            else if (is_marker(line, "PRELOAD_FFN_UP_MULT"))
                read_int_lines(WIDE_M, 0, 1, "FFN_UP_MULT");
            else if (is_marker(line, "PRELOAD_FFN_DOWN_WEIGHT"))
                read_int_lines(DOWN_W, 1, 0, "FFN_DOWN_WEIGHT");
            else if (is_marker(line, "PRELOAD_FFN_DOWN_MULT"))
                read_int_lines(DOWN_M, 0, 1, "FFN_DOWN_MULT");
            else if (line.substr(0, 3) == "CASE") begin
                void'($sscanf(line, "CASE %d %d", cid, seq));
            $display("CASE %0d seq=%0d — cmd/x/out...", cid, seq);
            run_case(cid, seq);
            end
        end
        $fclose(vf);

        if (hseed == EXPECTED_SEED && tc == EXPECTED_NUM_CASES && mc == 0)
            $display("TEST RESULT: PASS %0d/%0d EXACT", tc, EXPECTED_NUM_CASES);
        else
            $display("TEST RESULT: FAIL tc=%0d mc=%0d seed=%0d exp_seed=%0d",
                tc, mc, hseed, EXPECTED_SEED);
        $finish;
    end

    initial begin #2000000000; $display("TIMEOUT"); $finish; end

`ifdef TRANSFORMER_BLOCK_DBG
    initial begin
        forever begin
            #500000;
            if ($time > 500000)
                $display("DBG t=%0t st=%0d ls=%0d head=%0d idx=%0d ra=%0d ra_x=%b/%b ra_y=%b/%b ra_z=%b/%b wo_ts=%0d",
                    $time, dut.st, dut.ls, dut.head_r, dut.idx_r,
                    dut.u_res.state,
                    dut.u_res.x_valid, dut.u_res.x_ready,
                    dut.u_res.y_valid, dut.u_res.y_ready,
                    dut.u_res.z_valid, dut.u_res.z_ready,
                    dut.u_wo.tile_state);
        end
    end
`endif
endmodule
