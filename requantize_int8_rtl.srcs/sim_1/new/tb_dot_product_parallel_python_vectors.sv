// ============================================================================
// TB: tb_dot_product_parallel_python_vectors.sv
// 测:  int8_dot_product_parallel
// 作用: Task D — 读 dot_product_parallel_vectors.txt，120 case 对拍
// ============================================================================

// Task D: 读 dot_product_parallel_vectors.txt
// 格式 CASE id K gap hold expected + BEAT a0..b3 keep + END
`timescale 1ns/1ps

module tb_dot_product_parallel_python_vectors;
    localparam int INPUT_WIDTH           = 8;
    localparam int PARTIAL_WIDTH         = 18;
    localparam int ACC_WIDTH             = 32;
    localparam int MAX_K                 = 256;
    localparam int LENGTH_WIDTH          = $clog2(MAX_K + 1);
    localparam int PARTIAL_LATENCY       = 2;
    localparam int ACC_LATENCY           = 3;
    localparam int CASE_TIMEOUT_CYCLES   = 256;
    localparam int EXPECTED_NUM_CASES    = 120;
    localparam int EXPECTED_SEED         = 42;
    localparam string VECTOR_FILE        =
        "F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/dot_product_parallel_vectors.txt";

    logic clk;
    logic rst_n;
    logic cmd_valid;
    logic cmd_ready;
    logic [LENGTH_WIDTH-1:0] cmd_length;
    logic s_valid;
    logic s_ready;
    logic signed [INPUT_WIDTH-1:0] s_a0;
    logic signed [INPUT_WIDTH-1:0] s_a1;
    logic signed [INPUT_WIDTH-1:0] s_a2;
    logic signed [INPUT_WIDTH-1:0] s_a3;
    logic signed [INPUT_WIDTH-1:0] s_b0;
    logic signed [INPUT_WIDTH-1:0] s_b1;
    logic signed [INPUT_WIDTH-1:0] s_b2;
    logic signed [INPUT_WIDTH-1:0] s_b3;
    logic [3:0] s_keep;
    logic acc_clear;
    logic acc_enable;
    logic m_valid;
    logic m_ready;
    logic signed [ACC_WIDTH-1:0] m_result;

    integer vector_file;
    integer test_count;
    integer mismatch_count;
    integer header_num_cases;
    integer header_seed;
    string line_str;

    int8_dot_product_parallel #(
        .INPUT_WIDTH(INPUT_WIDTH),
        .PARTIAL_WIDTH(PARTIAL_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .MAX_K(MAX_K),
        .LENGTH_WIDTH(LENGTH_WIDTH),
        .PARTIAL_LATENCY(PARTIAL_LATENCY),
        .ACC_LATENCY(ACC_LATENCY)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .cmd_valid(cmd_valid),
        .cmd_ready(cmd_ready),
        .cmd_length(cmd_length),
        .s_valid(s_valid),
        .s_ready(s_ready),
        .s_a0(s_a0),
        .s_a1(s_a1),
        .s_a2(s_a2),
        .s_a3(s_a3),
        .s_b0(s_b0),
        .s_b1(s_b1),
        .s_b2(s_b2),
        .s_b3(s_b3),
        .s_keep(s_keep),
        .acc_clear(acc_clear),
        .acc_enable(acc_enable),
        .m_valid(m_valid),
        .m_ready(m_ready),
        .m_result(m_result),
        .partial_valid(),
        .partial_sum(),
        .acc_valid(),
        .acc_value()
    );

    task automatic clear_inputs;
        begin
            s_valid = 1'b0;
            s_a0 = '0;
            s_a1 = '0;
            s_a2 = '0;
            s_a3 = '0;
            s_b0 = '0;
            s_b1 = '0;
            s_b2 = '0;
            s_b3 = '0;
            s_keep = 4'b0000;
        end
    endtask

    task automatic pulse_clear;
        begin
            @(negedge clk);
            cmd_valid = 1'b0;
            cmd_length = '0;
            acc_clear = 1'b1;
            clear_inputs();
            @(posedge clk);
            #1;
            @(negedge clk);
            acc_clear = 1'b0;
        end
    endtask

    task automatic recover_dut;
        begin
            m_ready = 1'b1;
            @(posedge clk);
            #1;
            @(negedge clk);
            m_ready = 1'b0;
            acc_clear = 1'b1;
            @(posedge clk);
            #1;
            @(negedge clk);
            acc_clear = 1'b0;
        end
    endtask

    task automatic send_command(input integer K, output bit ok);
        begin
            ok = 1'b1;
            @(negedge clk);
            cmd_length = K[LENGTH_WIDTH-1:0];
            cmd_valid = 1'b0;
            @(negedge clk);
            cmd_valid = 1'b1;
            if (!cmd_ready) begin
                ok = 1'b0;
                cmd_valid = 1'b0;
                mismatch_count = mismatch_count + 1;
                $display("FAIL send_command not ready K=%0d state=%0d", K, dut.state);
                recover_dut();
                return;
            end
            @(posedge clk);
            #1;
            if (dut.state != 1) begin
                ok = 1'b0;
                cmd_valid = 1'b0;
                mismatch_count = mismatch_count + 1;
                $display("FAIL send_command no accept K=%0d state=%0d", K, dut.state);
                recover_dut();
                return;
            end
            @(negedge clk);
            cmd_valid = 1'b0;
        end
    endtask

    task automatic send_beat(
        input integer signed a0,
        input integer signed a1,
        input integer signed a2,
        input integer signed a3,
        input integer signed b0,
        input integer signed b1,
        input integer signed b2,
        input integer signed b3,
        input bit [3:0] keep,
        input integer gap_cycles
    );
        integer gap;
        integer wait_cycles;
        begin
            for (gap = 0; gap < gap_cycles; gap = gap + 1) begin
                @(negedge clk);
                clear_inputs();
            end
            @(negedge clk);
            s_a0 = a0;
            s_a1 = a1;
            s_a2 = a2;
            s_a3 = a3;
            s_b0 = b0;
            s_b1 = b1;
            s_b2 = b2;
            s_b3 = b3;
            s_keep = keep;
            s_valid = 1'b1;
            wait_cycles = 0;
            while (!s_ready && (wait_cycles < CASE_TIMEOUT_CYCLES)) begin
                @(negedge clk);
                wait_cycles = wait_cycles + 1;
            end
            if (!s_ready) begin
                mismatch_count = mismatch_count + 1;
                $display("FAIL send_beat timeout keep=%b", keep);
                clear_inputs();
                recover_dut();
                return;
            end
            @(posedge clk);
            #1;
            @(negedge clk);
            clear_inputs();
        end
    endtask

    task automatic receive_result(
        input integer signed expected,
        input integer case_id,
        input integer K,
        input integer hold_cycles
    );
        integer hold;
        integer wait_cycles;
        integer signed held_result;
        begin
            m_ready = 1'b0;
            wait_cycles = 0;
            while (!m_valid && (wait_cycles < CASE_TIMEOUT_CYCLES)) begin
                @(posedge clk);
                #1;
                wait_cycles = wait_cycles + 1;
            end
            if (!m_valid) begin
                mismatch_count = mismatch_count + 1;
                $display("FAIL timeout case=%0d K=%0d waiting for m_valid", case_id, K);
                recover_dut();
                return;
            end
            held_result = $signed(m_result);
            if (held_result !== expected) begin
                mismatch_count = mismatch_count + 1;
                $display(
                    "FAIL case=%0d K=%0d rtl=%0d expected=%0d",
                    case_id, K, held_result, expected
                );
            end else begin
                $display("PASS case=%0d K=%0d result=%0d", case_id, K, held_result);
            end
            for (hold = 0; hold < hold_cycles; hold = hold + 1) begin
                @(negedge clk);
                if (!m_valid || ($signed(m_result) !== held_result)) begin
                    mismatch_count = mismatch_count + 1;
                    $display(
                        "FAIL backpressure case=%0d cycle=%0d m_valid=%0d result=%0d held=%0d",
                        case_id, hold, m_valid, $signed(m_result), held_result
                    );
                end
            end
            m_ready = 1'b1;
            @(posedge clk);
            #1;
            @(negedge clk);
            m_ready = 1'b0;
            if (m_valid) begin
                mismatch_count = mismatch_count + 1;
                $display("FAIL handshake case=%0d m_valid did not clear", case_id);
            end
        end
    endtask

    function automatic bit [3:0] parse_keep(input integer keep_value);
        begin
            parse_keep = keep_value[3:0];
        end
    endfunction

    task automatic run_vector_case(
        input integer case_id,
        input integer K,
        input integer gap_cycles,
        input integer hold_cycles,
        input integer signed expected
    );
        integer scan_count;
        integer a0;
        integer a1;
        integer a2;
        integer a3;
        integer b0;
        integer b1;
        integer b2;
        integer b3;
        integer keep_raw;
        bit [3:0] keep;
        bit cmd_ok;
        integer beat_idx;
        begin
            beat_idx = 0;
            send_command(K, cmd_ok);
            if (!cmd_ok) begin
                test_count = test_count + 1;
                return;
            end
            while (!$feof(vector_file)) begin
                if ($fgets(line_str, vector_file) == 0) begin
                    continue;
                end
                if (line_str.len() == 0) begin
                    continue;
                end
                if (line_str.substr(0, 2) == "END") begin
                    break;
                end
                if (line_str.substr(0, 3) == "BEAT") begin
                    scan_count = $sscanf(
                        line_str,
                        "BEAT %d %d %d %d %d %d %d %d %b",
                        a0, a1, a2, a3, b0, b1, b2, b3, keep_raw
                    );
                    if (scan_count != 9) begin
                        mismatch_count = mismatch_count + 1;
                        $display("FAIL parse BEAT case=%0d beat=%0d", case_id, beat_idx);
                        recover_dut();
                        test_count = test_count + 1;
                        return;
                    end
                    keep = parse_keep(keep_raw);
                    send_beat(
                        a0, a1, a2, a3,
                        b0, b1, b2, b3,
                        keep,
                        (beat_idx == 0) ? 0 : gap_cycles
                    );
                    beat_idx = beat_idx + 1;
                end else begin
                    mismatch_count = mismatch_count + 1;
                    $display("FAIL unexpected line in case=%0d: %s", case_id, line_str);
                    recover_dut();
                    test_count = test_count + 1;
                    return;
                end
            end
            receive_result(expected, case_id, K, hold_cycles);
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
        integer K;
        integer gap_cycles;
        integer hold_cycles;
        integer signed expected;

        rst_n = 1'b0;
        cmd_valid = 1'b0;
        cmd_length = '0;
        clear_inputs();
        acc_clear = 1'b0;
        acc_enable = 1'b1;
        m_ready = 1'b0;
        test_count = 0;
        mismatch_count = 0;

        vector_file = $fopen(VECTOR_FILE, "r");
        if (vector_file == 0) begin
            $display("ERROR: cannot open vector file");
            $display("FILE: %s", VECTOR_FILE);
            $fatal;
        end

        header_num_cases = -1;
        header_seed = -1;

        $display("Task D python-vector regression");
        $display("Vector file: %s", VECTOR_FILE);

        repeat (2) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        pulse_clear();

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
                    "CASE %d %d %d %d %d",
                    case_id, K, gap_cycles, hold_cycles, expected
                );
                if (scan_count != 5) begin
                    mismatch_count = mismatch_count + 1;
                    $display("FAIL parse CASE line: %s", line_str);
                    continue;
                end
                pulse_clear();
                run_vector_case(case_id, K, gap_cycles, hold_cycles, expected);
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
        #5000000;
        $display("ERROR: simulation timeout");
        $display("test_count=%0d mismatch_count=%0d", test_count, mismatch_count);
        $finish;
    end
endmodule
