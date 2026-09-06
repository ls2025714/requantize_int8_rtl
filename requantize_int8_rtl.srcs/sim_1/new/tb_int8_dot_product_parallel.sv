// ============================================================================
// 文件: tb_int8_dot_product_parallel.sv
// 阶段: D1/D2（Task B/C）
// 作用: 定向测试并行点积 partial/acc 时序、FSM 与随机结果
// DUT: int8_dot_product_parallel
// 向量: 内联（Task B 时序 + Task C 定向 + 随机）
// ============================================================================
//
// 流程: Task B 查 partial/acc 延迟 → Task C FSM 定向 → 随机 case 对拍 m_result
// task: send_command / send_beat / receive_result / pulse_clear
//
`timescale 1ns/1ps

module tb_int8_dot_product_parallel;
    localparam int INPUT_WIDTH      = 8;
    localparam int PARTIAL_WIDTH    = 18;
    localparam int ACC_WIDTH        = 32;
    localparam int MAX_K            = 256;
    localparam int LENGTH_WIDTH     = $clog2(MAX_K + 1);
    // Accept at T0 -> S1@T0 -> S2@T1 -> S3@T2. Sample after NBA with #1,
    // so first partial_valid is observed on cycle 2 after accept.
    localparam int PARTIAL_LATENCY  = 2;
    // Acc updates on the S3 cycle; acc_valid is registered one cycle later.
    localparam int ACC_LATENCY      = 3;
    localparam int CONT_BEATS       = 8;
    localparam int CASE_TIMEOUT_CYCLES = 64;

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
    logic partial_valid;
    logic signed [PARTIAL_WIDTH-1:0] partial_sum;
    logic acc_valid;
    logic signed [ACC_WIDTH-1:0] acc_value;

    integer test_count;
    integer mismatch_count;

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
        .partial_valid(partial_valid),
        .partial_sum(partial_sum),
        .acc_valid(acc_valid),
        .acc_value(acc_value)
    );

    function automatic integer signed lane_product(
        input integer signed a_value,
        input integer signed b_value,
        input bit keep
    );
        if (!keep) begin
            return 0;
        end
        return a_value * b_value;
    endfunction

    function automatic integer signed expected_partial_sum(
        input integer signed a0,
        input integer signed a1,
        input integer signed a2,
        input integer signed a3,
        input integer signed b0,
        input integer signed b1,
        input integer signed b2,
        input integer signed b3,
        input bit [3:0] keep
    );
        integer signed sum_value;
        begin
            sum_value = 0;
            sum_value = sum_value + lane_product(a0, b0, keep[0]);
            sum_value = sum_value + lane_product(a1, b1, keep[1]);
            sum_value = sum_value + lane_product(a2, b2, keep[2]);
            sum_value = sum_value + lane_product(a3, b3, keep[3]);
            return sum_value;
        end
    endfunction

    // -------------------------------------------------------------------------
    // 通用 helper：清输入、采样、发 cmd/beat、收 m_result
    // -------------------------------------------------------------------------
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

    // Sample after NBA settles to avoid posedge race with DUT.
    task automatic sample_cycle;
        begin
            @(posedge clk);
            #1;
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

    // Present one beat for exactly one rising edge, then drop valid.
    task automatic accept_one_beat(
        input integer signed a0,
        input integer signed a1,
        input integer signed a2,
        input integer signed a3,
        input integer signed b0,
        input integer signed b1,
        input integer signed b2,
        input integer signed b3,
        input bit [3:0] keep
    );
        begin
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
            if (!s_ready) begin
                mismatch_count = mismatch_count + 1;
                $display("FAIL accept_one_beat: s_ready=0");
            end
            @(posedge clk);
            #1;
            @(negedge clk);
            clear_inputs();
        end
    endtask

    task automatic check_partial_latency(
        input integer case_id,
        input string case_name,
        input integer signed a0,
        input integer signed a1,
        input integer signed a2,
        input integer signed a3,
        input integer signed b0,
        input integer signed b1,
        input integer signed b2,
        input integer signed b3,
        input bit [3:0] keep
    );
        integer signed expected;
        integer latency;
        integer saw_valid;
        begin
            expected = expected_partial_sum(a0, a1, a2, a3, b0, b1, b2, b3, keep);
            pulse_clear();
            accept_one_beat(a0, a1, a2, a3, b0, b1, b2, b3, keep);

            latency = 0;
            saw_valid = 0;
            while ((latency < CASE_TIMEOUT_CYCLES) && (saw_valid == 0)) begin
                sample_cycle();
                latency = latency + 1;
                if (partial_valid) begin
                    saw_valid = 1;
                    if (latency !== PARTIAL_LATENCY) begin
                        mismatch_count = mismatch_count + 1;
                        $display(
                            "FAIL case=%0d %s latency=%0d expected_latency=%0d",
                            case_id, case_name, latency, PARTIAL_LATENCY
                        );
                    end else if ($signed(partial_sum) !== expected) begin
                        mismatch_count = mismatch_count + 1;
                        $display(
                            "FAIL case=%0d %s rtl=%0d expected=%0d keep=%b",
                            case_id, case_name, $signed(partial_sum), expected, keep
                        );
                    end else begin
                        $display(
                            "PASS case=%0d %s result=%0d latency=%0d keep=%b",
                            case_id, case_name, $signed(partial_sum), latency, keep
                        );
                    end
                end
            end

            if (saw_valid == 0) begin
                mismatch_count = mismatch_count + 1;
                $display("FAIL case=%0d %s missing partial_valid", case_id, case_name);
            end else begin
                sample_cycle();
                if (partial_valid) begin
                    mismatch_count = mismatch_count + 1;
                    $display("FAIL case=%0d %s partial_valid stuck high", case_id, case_name);
                end
            end

            test_count = test_count + 1;
        end
    endtask

    function automatic integer beat_count_for_k(input integer K);
        return (K + 3) / 4;
    endfunction

    function automatic bit [3:0] beat_keep_mask(input integer K, input integer beat_idx);
        integer base_idx;
        integer lane;
        begin
            beat_keep_mask = 4'b0000;
            base_idx = beat_idx * 4;
            for (lane = 0; lane < 4; lane = lane + 1) begin
                if ((base_idx + lane) < K) begin
                    beat_keep_mask[lane] = 1'b1;
                end
            end
        end
    endfunction

    function automatic integer signed expected_dot_product_k(
        input integer K,
        input integer signed a_vec [0:15],
        input integer signed b_vec [0:15]
    );
        integer idx;
        integer signed sum_value;
        begin
            sum_value = 0;
            for (idx = 0; idx < K; idx = idx + 1) begin
                sum_value = sum_value + a_vec[idx] * b_vec[idx];
            end
            return sum_value;
        end
    endfunction

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
                $display(
                    "FAIL send_command not ready K=%0d state=%0d cmd_length=%0d",
                    K, dut.state, cmd_length
                );
                recover_dut();
                return;
            end
            @(posedge clk);
            #1;
            if (dut.state != 1) begin
                ok = 1'b0;
                cmd_valid = 1'b0;
                mismatch_count = mismatch_count + 1;
                $display(
                    "FAIL send_command no accept K=%0d state=%0d",
                    K, dut.state
                );
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
                $display("FAIL fsm timeout case=%0d K=%0d waiting for m_valid", case_id, K);
                recover_dut();
                return;
            end
            held_result = $signed(m_result);
            if (held_result !== expected) begin
                mismatch_count = mismatch_count + 1;
                $display("FAIL fsm case=%0d K=%0d rtl=%0d expected=%0d", case_id, K, held_result, expected);
            end else begin
                $display("PASS fsm case=%0d K=%0d result=%0d", case_id, K, held_result);
            end
            for (hold = 0; hold < hold_cycles; hold = hold + 1) begin
                @(negedge clk);
                if (!m_valid || ($signed(m_result) !== held_result)) begin
                    mismatch_count = mismatch_count + 1;
                    $display(
                        "FAIL fsm backpressure case=%0d cycle=%0d m_valid=%0d result=%0d held=%0d",
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
                $display("FAIL fsm handshake case=%0d m_valid did not clear", case_id);
            end
        end
    endtask

    // 跑一组完整 FSM case：cmd → 若干 beat → receive_result
    task automatic run_fsm_case(
        input integer case_id,
        input integer K,
        input integer signed a_vec [0:15],
        input integer signed b_vec [0:15],
        input integer gap_cycles,
        input integer hold_cycles
    );
        integer beats;
        integer beat;
        integer base;
        integer lane;
        integer signed a0;
        integer signed a1;
        integer signed a2;
        integer signed a3;
        integer signed b0;
        integer signed b1;
        integer signed b2;
        integer signed b3;
        bit [3:0] keep;
        bit cmd_ok;
        integer signed expected;
        begin
            $display("FSM case=%0d K=%0d beats=%0d", case_id, K, beat_count_for_k(K));
            expected = expected_dot_product_k(K, a_vec, b_vec);
            send_command(K, cmd_ok);
            if (!cmd_ok) begin
                test_count = test_count + 1;
                return;
            end
            beats = beat_count_for_k(K);
            for (beat = 0; beat < beats; beat = beat + 1) begin
                base = beat * 4;
                a0 = (base + 0 < K) ? a_vec[base + 0] : 0;
                a1 = (base + 1 < K) ? a_vec[base + 1] : 0;
                a2 = (base + 2 < K) ? a_vec[base + 2] : 0;
                a3 = (base + 3 < K) ? a_vec[base + 3] : 0;
                b0 = (base + 0 < K) ? b_vec[base + 0] : 0;
                b1 = (base + 1 < K) ? b_vec[base + 1] : 0;
                b2 = (base + 2 < K) ? b_vec[base + 2] : 0;
                b3 = (base + 3 < K) ? b_vec[base + 3] : 0;
                keep = beat_keep_mask(K, beat);
                send_beat(a0, a1, a2, a3, b0, b1, b2, b3, keep, (beat == 0) ? 0 : gap_cycles);
            end
            receive_result(expected, case_id, K, hold_cycles);
            test_count = test_count + 1;
        end
    endtask

    // -------------------------------------------------------------------------
    // Task C：FSM 定向 case（K=1..16、尾 beat keep、随机 K）
    // -------------------------------------------------------------------------
    task automatic check_fsm_task_c;
        integer signed a_vec [0:15];
        integer signed b_vec [0:15];
        integer idx;
        integer random_k;
        integer j;
        begin
            // K=1, lane 0 only
            a_vec[0] = 7;  b_vec[0] = 5;
            run_fsm_case(101, 1, a_vec, b_vec, 0, 2);

            // K=3, single-beat tail mask
            a_vec[0] = 2;  b_vec[0] = 3;
            a_vec[1] = -4; b_vec[1] = 5;
            a_vec[2] = 6;  b_vec[2] = -1;
            run_fsm_case(102, 3, a_vec, b_vec, 0, 2);

            // K=4, one full beat
            a_vec[0] = 1;  b_vec[0] = 2;
            a_vec[1] = 3;  b_vec[1] = 4;
            a_vec[2] = -5; b_vec[2] = 6;
            a_vec[3] = 7;  b_vec[3] = -8;
            run_fsm_case(103, 4, a_vec, b_vec, 0, 2);

            // K=5, one full beat + one lane
            for (idx = 0; idx < 5; idx = idx + 1) begin
                a_vec[idx] = idx + 1;
                b_vec[idx] = (idx % 2 == 0) ? 2 : -3;
            end
            run_fsm_case(104, 5, a_vec, b_vec, 0, 2);

            // K=7, last beat keep=0111
            for (idx = 0; idx < 7; idx = idx + 1) begin
                a_vec[idx] = 10 + idx;
                b_vec[idx] = 1;
            end
            run_fsm_case(105, 7, a_vec, b_vec, 0, 3);

            // K=8, two full beats
            for (idx = 0; idx < 8; idx = idx + 1) begin
                a_vec[idx] = (idx % 2 == 0) ? 3 : -2;
                b_vec[idx] = idx + 1;
            end
            run_fsm_case(106, 8, a_vec, b_vec, 0, 2);

            // K=16, continuous full load
            for (idx = 0; idx < 16; idx = idx + 1) begin
                a_vec[idx] = idx - 8;
                b_vec[idx] = (idx % 3 == 0) ? 4 : -1;
            end
            run_fsm_case(107, 16, a_vec, b_vec, 0, 2);

            // K=17, multi-beat with tail
            for (idx = 0; idx < 17; idx = idx + 1) begin
                a_vec[idx] = 17 - idx;
                b_vec[idx] = 2;
            end
            run_fsm_case(108, 17, a_vec, b_vec, 0, 2);

            // -128 x -128 boundary inside K=4 beat
            a_vec[0] = -128; b_vec[0] = -128;
            a_vec[1] = 127;  b_vec[1] = 127;
            a_vec[2] = -1;   b_vec[2] = 1;
            a_vec[3] = 0;    b_vec[3] = 0;
            run_fsm_case(109, 4, a_vec, b_vec, 0, 2);

            // Input bubbles between beats, K=7
            run_fsm_case(110, 7, a_vec, b_vec, 2, 2);

            // Consecutive commands without overlapping output
            for (idx = 0; idx < 4; idx = idx + 1) begin
                a_vec[idx] = idx + 2;
                b_vec[idx] = 3;
            end
            run_fsm_case(111, 4, a_vec, b_vec, 0, 1);
            a_vec[0] = -5; b_vec[0] = 6;
            run_fsm_case(112, 1, a_vec, b_vec, 0, 1);

            // 10 random K cases with bubbles and backpressure
            for (idx = 0; idx < 10; idx = idx + 1) begin
                random_k = $urandom_range(1, 16);
                for (j = 0; j < random_k; j = j + 1) begin
                    a_vec[j] = $urandom_range(0, 255) - 128;
                    b_vec[j] = $urandom_range(0, 255) - 128;
                end
                run_fsm_case(200 + idx, random_k, a_vec, b_vec,
                             $urandom_range(0, 2), $urandom_range(1, 3));
            end
        end
    endtask

    task automatic check_continuous_and_bubbles;
        integer signed expected_q [0:CONT_BEATS-1];
        integer signed a0_q [0:CONT_BEATS-1];
        integer signed a1_q [0:CONT_BEATS-1];
        integer signed a2_q [0:CONT_BEATS-1];
        integer signed a3_q [0:CONT_BEATS-1];
        integer signed b0_q [0:CONT_BEATS-1];
        integer signed b1_q [0:CONT_BEATS-1];
        integer signed b2_q [0:CONT_BEATS-1];
        integer signed b3_q [0:CONT_BEATS-1];
        bit [3:0] keep_q [0:CONT_BEATS-1];
        integer signed expected_acc;
        integer signed held_acc;
        integer i;
        integer out_index;
        integer wait_cycles;
        integer got_acc;
        begin
            expected_acc = 0;
            for (i = 0; i < CONT_BEATS; i = i + 1) begin
                a0_q[i] = i + 1;
                a1_q[i] = i + 2;
                a2_q[i] = -(i + 3);
                a3_q[i] = i + 4;
                b0_q[i] = 2;
                b1_q[i] = -1;
                b2_q[i] = 3;
                b3_q[i] = 1;
                keep_q[i] = (i == (CONT_BEATS - 1)) ? 4'b0111 : 4'b1111;
                expected_q[i] = expected_partial_sum(
                    a0_q[i], a1_q[i], a2_q[i], a3_q[i],
                    b0_q[i], b1_q[i], b2_q[i], b3_q[i],
                    keep_q[i]
                );
                expected_acc = expected_acc + expected_q[i];
            end

            // Continuous 1 beat/cycle after fill.
            pulse_clear();
            out_index = 0;
            @(negedge clk);
            for (i = 0; i < CONT_BEATS; i = i + 1) begin
                s_a0 = a0_q[i];
                s_a1 = a1_q[i];
                s_a2 = a2_q[i];
                s_a3 = a3_q[i];
                s_b0 = b0_q[i];
                s_b1 = b1_q[i];
                s_b2 = b2_q[i];
                s_b3 = b3_q[i];
                s_keep = keep_q[i];
                s_valid = 1'b1;
                if (!s_ready) begin
                    mismatch_count = mismatch_count + 1;
                    $display("FAIL continuous: s_ready low at beat=%0d", i);
                end
                @(posedge clk);
                #1;
                if (partial_valid) begin
                    if (out_index >= CONT_BEATS) begin
                        mismatch_count = mismatch_count + 1;
                        $display("FAIL continuous: unexpected extra partial");
                    end else if ($signed(partial_sum) !== expected_q[out_index]) begin
                        mismatch_count = mismatch_count + 1;
                        $display(
                            "FAIL continuous partial idx=%0d rtl=%0d expected=%0d",
                            out_index, $signed(partial_sum), expected_q[out_index]
                        );
                    end
                    out_index = out_index + 1;
                end
                @(negedge clk);
            end
            clear_inputs();

            wait_cycles = 0;
            while ((out_index < CONT_BEATS) && (wait_cycles < CASE_TIMEOUT_CYCLES)) begin
                sample_cycle();
                wait_cycles = wait_cycles + 1;
                if (partial_valid) begin
                    if ($signed(partial_sum) !== expected_q[out_index]) begin
                        mismatch_count = mismatch_count + 1;
                        $display(
                            "FAIL continuous drain idx=%0d rtl=%0d expected=%0d",
                            out_index, $signed(partial_sum), expected_q[out_index]
                        );
                    end
                    out_index = out_index + 1;
                end
            end

            if (out_index !== CONT_BEATS) begin
                mismatch_count = mismatch_count + 1;
                $display("FAIL continuous: got %0d / %0d partials", out_index, CONT_BEATS);
            end else begin
                $display(
                    "PASS continuous: %0d beats, 1 beat/cycle, PARTIAL_LATENCY=%0d",
                    CONT_BEATS, PARTIAL_LATENCY
                );
            end
            test_count = test_count + 1;

            // Acc should already hold final sum after last S3 cycle; allow one drain cycle.
            wait_cycles = 0;
            got_acc = 0;
            while ((wait_cycles < CASE_TIMEOUT_CYCLES) && (got_acc == 0)) begin
                sample_cycle();
                wait_cycles = wait_cycles + 1;
                if (!partial_valid && ($signed(acc_value) === expected_acc)) begin
                    got_acc = 1;
                end
            end
            if (got_acc == 0) begin
                mismatch_count = mismatch_count + 1;
                $display("FAIL continuous acc rtl=%0d expected=%0d", $signed(acc_value), expected_acc);
            end else begin
                $display("PASS continuous acc result=%0d", $signed(acc_value));
            end
            test_count = test_count + 1;

            // Bubble test: idle gaps between beats; collect every partial as it appears.
            pulse_clear();
            out_index = 0;
            for (i = 0; i < 3; i = i + 1) begin
                accept_one_beat(
                    a0_q[i], a1_q[i], a2_q[i], a3_q[i],
                    b0_q[i], b1_q[i], b2_q[i], b3_q[i],
                    keep_q[i]
                );
                // Two idle cycles; collect any in-flight partials during the gap.
                repeat (2) begin
                    sample_cycle();
                    if (partial_valid) begin
                        if ($signed(partial_sum) !== expected_q[out_index]) begin
                            mismatch_count = mismatch_count + 1;
                            $display(
                                "FAIL bubble partial idx=%0d rtl=%0d expected=%0d",
                                out_index, $signed(partial_sum), expected_q[out_index]
                            );
                        end
                        out_index = out_index + 1;
                    end
                end
            end

            wait_cycles = 0;
            while ((out_index < 3) && (wait_cycles < CASE_TIMEOUT_CYCLES)) begin
                sample_cycle();
                wait_cycles = wait_cycles + 1;
                if (partial_valid) begin
                    if ($signed(partial_sum) !== expected_q[out_index]) begin
                        mismatch_count = mismatch_count + 1;
                        $display(
                            "FAIL bubble drain idx=%0d rtl=%0d expected=%0d",
                            out_index, $signed(partial_sum), expected_q[out_index]
                        );
                    end
                    out_index = out_index + 1;
                end
            end

            sample_cycle();
            if (out_index !== 3) begin
                mismatch_count = mismatch_count + 1;
                $display("FAIL bubble: got %0d / 3 partials", out_index);
            end else if (partial_valid) begin
                mismatch_count = mismatch_count + 1;
                $display("FAIL bubble: unexpected extra partial_valid");
            end else begin
                $display("PASS bubble: 3 beats with idle gaps, valid aligned");
            end
            test_count = test_count + 1;

            // Single-beat accumulator latency.
            pulse_clear();
            held_acc = expected_q[0];
            accept_one_beat(
                a0_q[0], a1_q[0], a2_q[0], a3_q[0],
                b0_q[0], b1_q[0], b2_q[0], b3_q[0],
                keep_q[0]
            );
            got_acc = 0;
            for (i = 1; i <= CASE_TIMEOUT_CYCLES; i = i + 1) begin
                sample_cycle();
                if (acc_valid && (got_acc == 0)) begin
                    got_acc = 1;
                    if (i !== ACC_LATENCY) begin
                        mismatch_count = mismatch_count + 1;
                        $display("FAIL acc latency=%0d expected=%0d", i, ACC_LATENCY);
                    end else if ($signed(acc_value) !== held_acc) begin
                        mismatch_count = mismatch_count + 1;
                        $display("FAIL acc value rtl=%0d expected=%0d", $signed(acc_value), held_acc);
                    end else begin
                        $display("PASS acc latency=%0d value=%0d", i, $signed(acc_value));
                    end
                end
            end
            if (got_acc == 0) begin
                mismatch_count = mismatch_count + 1;
                $display("FAIL acc_valid never asserted");
            end
            test_count = test_count + 1;
        end
    endtask

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    // 主测试：Task B partial/acc → Task C FSM → 汇总 PASS/FAIL
    initial begin
        rst_n = 1'b0;
        cmd_valid = 1'b0;
        cmd_length = '0;
        clear_inputs();
        acc_clear = 1'b0;
        acc_enable = 1'b1;
        m_ready = 1'b0;
        test_count = 0;
        mismatch_count = 0;

        repeat (2) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        check_partial_latency(1, "all_positive_full", 3, 4, 5, 6, 2, 3, 4, 5, 4'b1111);
        check_partial_latency(2, "mixed_sign_full", 3, -4, 5, -6, 2, 3, -4, 5, 4'b1111);
        check_partial_latency(3, "max_positive", 127, 127, 127, 127, 127, 127, 127, 127, 4'b1111);
        check_partial_latency(4, "min_negative_product", -128, -128, 0, 0, -128, -128, 1, 1, 4'b0011);
        check_partial_latency(5, "single_lane0", 7, 9, 11, 13, 5, 6, 7, 8, 4'b0001);
        check_partial_latency(6, "tail_keep_0111", 10, 20, 30, 99, 2, 3, 4, 9, 4'b0111);
        check_partial_latency(7, "invalid_lanes_zeroed", 10, 20, 30, 40, 2, 3, 4, 5, 4'b0101);
        check_partial_latency(8, "all_invalid", 10, 20, 30, 40, 2, 3, 4, 5, 4'b0000);
        check_partial_latency(9, "negative_only", -1, -2, -3, -4, -5, -6, -7, -8, 4'b1111);
        check_partial_latency(10, "single_negative_edge", -128, 1, 1, 1, -128, 1, 1, 1, 4'b0001);

        check_continuous_and_bubbles();
        pulse_clear();
        $display("--- Task C FSM tests begin ---");
        check_fsm_task_c();

        $display("--------------------------------------------------");
        $display("PARTIAL_LATENCY = %0d", PARTIAL_LATENCY);
        $display("ACC_LATENCY     = %0d", ACC_LATENCY);
        $display("test_count      = %0d", test_count);
        $display("mismatch_count  = %0d", mismatch_count);
        if ((test_count == 36) && (mismatch_count == 0)) begin
            $display("TEST RESULT: PASS");
        end else begin
            $display("TEST RESULT: FAIL");
        end
        $display("--------------------------------------------------");
        $finish;
    end

    // Safety watchdog only; normal completion is $finish from the test sequence (~10-20 us).
    initial begin
        #20000;
        $display("ERROR: simulation timeout");
        $display("test_count=%0d mismatch_count=%0d", test_count, mismatch_count);
        $finish;
    end
endmodule
