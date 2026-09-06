// ============================================================================
// 文件: requantize_int8_pipeline.sv
// 学习阶段: D4 量化 — 2 拍流水（Linear / tiled / RQ 状态机对接的版本）
// ----------------------------------------------------------------------------
// 【为何流水】给上层 FSM 稳定的 in_valid→out_valid 延迟（固定 2 拍）
//   S1: product = acc * scale
//   S2: round half-away-from-zero + >>> SHIFT_BITS + 饱和 [-127,127]
// 【谁用】int8_linear_layer、int8_linear_tiled 的 TILE_REQ_* / OUT_* 
// 【验证】tb_requantize_int8_pipeline.sv
// ============================================================================
//
module requantize_int8_pipeline #(
    parameter int SHIFT_BITS = 24
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               in_valid,
    input  logic signed [31:0] acc_i,
    input  logic        [17:0] multiplier_i,
    output logic               out_valid,
    output logic signed  [7:0] out_i
);
localparam logic signed [49:0] HALF_LSB = 50'sd1 <<< (SHIFT_BITS - 1);
logic signed [18:0] multiplier_signed_in;
logic signed [49:0] product_s1;
logic               valid_s1;
logic signed [49:0] rounded_comb;
logic signed [49:0] shifted_comb;
logic signed [49:0] rounded_s2;
logic signed [49:0] shifted_s2;
logic               valid_s2;
assign multiplier_signed_in = $signed({1'b0, multiplier_i});
// --- S2 组合：round + 算术右移（基于 S1 product）---
always_comb begin
    if (product_s1 >= 0)
        rounded_comb = product_s1 + HALF_LSB;
    else
        rounded_comb = product_s1 + (HALF_LSB - 50'sd1);
    shifted_comb = rounded_comb >>> SHIFT_BITS;
end
// --- 2 拍流水：S1 乘积 → S2 round/shift → 饱和输出 ---
always_ff @(posedge clk) begin
    if (!rst_n) begin
        product_s1 <= 50'sd0;
        valid_s1 <= 1'b0;
        rounded_s2 <= 50'sd0;
        shifted_s2 <= 50'sd0;
        valid_s2 <= 1'b0;
        out_i <= 8'sd0;
        out_valid <= 1'b0;
    end
    else begin
        product_s1 <= acc_i * multiplier_signed_in;
        valid_s1 <= in_valid;
        rounded_s2 <= rounded_comb;
        shifted_s2 <= shifted_comb;
        valid_s2 <= valid_s1;
        if (shifted_s2 > 50'sd127)
            out_i <= 8'sd127;
        else if (shifted_s2 < -50'sd127)
            out_i <= -8'sd127;
        else
            out_i <= $signed(shifted_s2[7:0]);
        out_valid <= valid_s2;
    end
end
endmodule
