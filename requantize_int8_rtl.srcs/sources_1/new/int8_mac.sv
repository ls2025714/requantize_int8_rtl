// ============================================================================
// 文件: int8_mac.sv
// 阶段: 基础
// 作用: 单 MAC 单元 — activation × weight，可 clear/enable 控制累加器
// 验证: tb_int8_mac.sv
// ============================================================================

// 接口说明:
//   clear_i=1  清零累加器并打一拍 out_valid
//   enable_i=1 且 in_valid=1 时执行 acc += sign_ext(a*b)
//   a_i 通常接 activation，b_i 接 weight（命名习惯，乘法可交换）
// 位宽: INT8×INT8 → 16-bit 乘积，符号扩展到 32-bit 再累加
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