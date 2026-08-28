// ============================================================================
// 文件: requantize_int8.sv
// 阶段: 量化
// 作用: 组合逻辑 — (INT32 acc × 18-bit scale) >>> 24 → 对称饱和 INT8
// 验证: tb_requantize_int8_pipeline.sv
// ============================================================================
//
// 公式: out = saturate( round( acc_i × multiplier_i ) >> SHIFT_BITS )
//   acc_i — INT32 累加；multiplier_i — 18-bit 无符号 scale；饱和 ±127
//
module requantize_int8 #(
    parameter int SHIFT_BITS = 24
) (
    input  logic signed [31:0] acc_i,
    input  logic        [17:0] multiplier_i,
    output logic signed  [7:0] out_i
);

    // 18-bit unsigned multiplier 前面补一个 0，
    // 转换成非负的 19-bit signed 数。
    logic signed [18:0] multiplier_signed;

    // signed 32-bit accumulator × unsigned 18-bit multiplier
    // 使用 signed 50-bit 保存完整宽乘积。
    logic signed [49:0] product;
    logic signed [49:0] rounded_product;
    logic signed [49:0] shifted_value;

    // SHIFT_BITS = 24 时：
    // HALF_LSB = 2^(24-1) = 2^23 = 8,388,608
    localparam logic signed [49:0] HALF_LSB =
        50'sd1 <<< (SHIFT_BITS - 1);

    // 保持 multiplier_i 的非负数值，再参与 signed 乘法。
    assign multiplier_signed = $signed({1'b0, multiplier_i});

    // 宽乘法。
    assign product = acc_i * multiplier_signed;

    // Round half away from zero，并执行算术右移。
    always_comb begin
        if (product >= 0) begin
            // 正数：加 0.5 LSB。
            rounded_product = product + HALF_LSB;
        end
        else begin
            // 负数：修正 arithmetic right shift 的向负无穷取整。
            rounded_product = product + (HALF_LSB - 50'sd1);
        end

        // 算术右移，保留负数符号。
        shifted_value = rounded_product >>> SHIFT_BITS;
    end

    // Symmetric INT8 saturation，合法范围为 [-127, 127]。
    always_comb begin
        if (shifted_value > 50'sd127)
            out_i = 8'sd127;
        else if (shifted_value < -50'sd127)
            out_i = -8'sd127;
        else
            out_i = $signed(shifted_value[7:0]);
    end

endmodule