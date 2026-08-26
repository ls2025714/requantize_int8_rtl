`timescale 1ns / 1ps

module tb_requantize_int8;

    logic signed [31:0] acc_i;
    logic        [17:0] multiplier_i;
    logic signed  [7:0] out_i;

    integer pass_count;
    integer fail_count;

    requantize_int8 #(
        .SHIFT_BITS(24)
    ) dut (
        .acc_i        (acc_i),
        .multiplier_i (multiplier_i),
        .out_i        (out_i)
    );

    // ========================================================
    // 检查宽乘积
    // ========================================================

    task automatic check_product(
        input string               test_name,
        input logic signed [31:0]  test_acc,
        input logic        [17:0]  test_multiplier,
        input logic signed [49:0]  expected_product
    );
        begin
            acc_i        = test_acc;
            multiplier_i = test_multiplier;
            #10;

            if ($signed(dut.product) !== expected_product) begin
                $error(
                    "%s FAIL: product = %0d, expected = %0d",
                    test_name,
                    $signed(dut.product),
                    expected_product
                );
                fail_count = fail_count + 1;
            end
            else begin
                $display(
                    "%s PASS: product = %0d",
                    test_name,
                    $signed(dut.product)
                );
                pass_count = pass_count + 1;
            end
        end
    endtask

    // ========================================================
    // 检查舍入和右移后的宽结果
    // ========================================================

    task automatic check_shifted(
        input string               test_name,
        input logic signed [31:0]  test_acc,
        input logic        [17:0]  test_multiplier,
        input logic signed [49:0]  expected_shifted
    );
        begin
            acc_i        = test_acc;
            multiplier_i = test_multiplier;
            #10;

            if ($signed(dut.shifted_value) !== expected_shifted) begin
                $error(
                    "%s FAIL: shifted_value = %0d, expected = %0d",
                    test_name,
                    $signed(dut.shifted_value),
                    expected_shifted
                );
                fail_count = fail_count + 1;
            end
            else begin
                $display(
                    "%s PASS: shifted_value = %0d",
                    test_name,
                    $signed(dut.shifted_value)
                );
                pass_count = pass_count + 1;
            end
        end
    endtask

    // ========================================================
    // 检查最终 INT8 输出
    // ========================================================

    task automatic check_output(
        input string               test_name,
        input logic signed [31:0]  test_acc,
        input logic        [17:0]  test_multiplier,
        input logic signed  [7:0]  expected_output
    );
        begin
            acc_i        = test_acc;
            multiplier_i = test_multiplier;
            #10;

            if ($signed(out_i) !== expected_output) begin
                $error(
                    "%s FAIL: out_i = %0d, expected = %0d",
                    test_name,
                    $signed(out_i),
                    expected_output
                );
                fail_count = fail_count + 1;
            end
            else begin
                $display(
                    "%s PASS: out_i = %0d",
                    test_name,
                    $signed(out_i)
                );
                pass_count = pass_count + 1;
            end
        end
    endtask

    // ========================================================
    // 测试过程
    // ========================================================

    initial begin
        pass_count = 0;
        fail_count = 0;

        acc_i        = 32'sd0;
        multiplier_i = 18'd0;
        #10;

        $display("========================================");
        $display("Part 1: Wide multiplication");
        $display("========================================");

        // 测试 1：正数乘法
        check_product(
            "TEST 1",
            32'sd1000,
            18'd5000,
            50'sd5000000
        );

        // 测试 2：负数乘法
        check_product(
            "TEST 2",
            -32'sd1000,
            18'd5000,
            -50'sd5000000
        );

        // 测试 3：实验 08 的最大实际 Multiplier。
        // 205831 超过了 17-bit unsigned 的最大值 131071，
        // 用于验证当前 18-bit 接口没有发生截断。
        check_product(
            "TEST 3",
            32'sd1,
            18'd205831,
            50'sd205831
        );

        $display("========================================");
        $display("Part 2: Positive rounding");
        $display("========================================");

        // 2^23 - 1：略小于 +0.5，结果应为 0。
        check_shifted(
            "TEST 4",
            32'sd8388607,
            18'd1,
            50'sd0
        );

        // 2^23：正好 +0.5，half away from zero 得到 +1。
        check_shifted(
            "TEST 5",
            32'sd8388608,
            18'd1,
            50'sd1
        );

        // 3 × 2^23：正好 +1.5，结果应为 +2。
        check_shifted(
            "TEST 6",
            32'sd25165824,
            18'd1,
            50'sd2
        );

        $display("========================================");
        $display("Part 3: Negative rounding");
        $display("========================================");

        // 绝对值略小于 0.5，结果应为 0。
        check_shifted(
            "TEST 7",
            -32'sd8388607,
            18'd1,
            50'sd0
        );

        // 正好 -0.5，half away from zero 得到 -1。
        check_shifted(
            "TEST 8",
            -32'sd8388608,
            18'd1,
            -50'sd1
        );

        // 正好 -1.5，结果应为 -2。
        check_shifted(
            "TEST 9",
            -32'sd25165824,
            18'd1,
            -50'sd2
        );

        $display("========================================");
        $display("Part 4: INT8 saturation");
        $display("========================================");

        // +127 边界，不应被改变。
        check_output(
            "TEST 10",
            32'sd2130706432,
            18'd1,
            8'sd127
        );

        // 右移后得到 +128，应饱和到 +127。
        check_output(
            "TEST 11",
            32'sd1073741824,
            18'd2,
            8'sd127
        );

        // -127 边界，不应被改变。
        check_output(
            "TEST 12",
            -32'sd2130706432,
            18'd1,
            -8'sd127
        );

        // 右移后得到 -128，应饱和到 -127。
        check_output(
            "TEST 13",
            -32'sd1073741824,
            18'd2,
            -8'sd127
        );

        // 范围内普通值，应直接输出 42。
        check_output(
            "TEST 14",
            32'sd704643072,
            18'd1,
            8'sd42
        );

        $display("========================================");
        $display("Test summary");
        $display("PASS count = %0d", pass_count);
        $display("FAIL count = %0d", fail_count);
        $display("========================================");

        if (fail_count == 0) begin
            $display("ALL REQUANTIZATION TESTS PASSED.");
            $finish;
        end
        else begin
            $fatal(
                1,
                "REQUANTIZATION TEST FAILED: %0d failure(s)",
                fail_count
            );
        end
    end

endmodule