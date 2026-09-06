// ============================================================================
// 文件: int8_mac.sv
// 学习阶段: D1 基线 — 单拍 MAC（乘加累加器）
// ----------------------------------------------------------------------------
// 【在整条链的位置】最底层算子：acc += a×b（有符号 INT8）
//   后续串行点积 / 流水 MAC 都建立在「乘积符号扩展到 INT32 再累加」上
//
// 【行为】
//   clear_i=1 → 清零 acc，并打一拍 out_valid
//   enable_i && in_valid → acc += sign_ext16to32(a_i * b_i)
//   位宽: INT8×INT8 → 16-bit 乘积 → 扩展到 32-bit 累加
// 【验证】tb_int8_mac.sv
// ============================================================================
//
module int8_mac #(
    parameter int DATA_W = 8,
    parameter int ACC_W = 32
) (
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         clear_i,
    input  logic                         enable_i,
    input  logic                         in_valid,
    input  logic signed [DATA_W-1:0]     a_i,
    input  logic signed [DATA_W-1:0]     b_i,
    output logic                         out_valid,
    output logic signed [ACC_W-1:0]      acc_o
);
// --- 组合乘法 + 符号扩展 ---
localparam int PRODUCT_W = 2 * DATA_W;
logic signed [PRODUCT_W-1:0] product_comb;
logic signed [ACC_W-1:0] product_extended;
assign product_comb = a_i * b_i;
assign product_extended = {{(ACC_W-PRODUCT_W){product_comb[PRODUCT_W-1]}}, product_comb};
// --- 累加寄存器：clear / enable+valid ---
always_ff @(posedge clk) begin
    if (!rst_n) begin
        acc_o <= '0;
        out_valid <= 1'b0;
    end
    else begin
        out_valid <= 1'b0;
        if (clear_i) begin
            acc_o <= '0;
            out_valid <= 1'b1;
        end
        else if (enable_i && in_valid) begin
            acc_o <= acc_o + product_extended;
            out_valid <= 1'b1;
        end
    end
end
endmodule