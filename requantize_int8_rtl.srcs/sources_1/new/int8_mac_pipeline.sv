// ============================================================================
// 文件: int8_mac_pipeline.sv
// 阶段: 基础
// 作用: 1 级流水线 MAC，每拍可喂一对 activation/weight
// 验证: tb_int8_mac_pipeline.sv
// ============================================================================

// 流水线: S1 寄存乘积 + valid_s1；下一拍 valid_s1 为真时累加到 acc_out
// clear 时清零乘积寄存器、valid、acc_out
// enable=0 时保持 acc，out_valid 拉低
module int8_mac_pipeline #(
    parameter int INPUT_WIDTH = 8,
    parameter int ACC_WIDTH   = 32
)(
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         clear,
    input  logic                         enable,
    input  logic                         in_valid,
    input  logic signed [INPUT_WIDTH-1:0] activation,
    input  logic signed [INPUT_WIDTH-1:0] weight,
    output logic                         out_valid,
    output logic signed [ACC_WIDTH-1:0]   acc_out
);
// --- S1: 乘积寄存 ---
localparam int PRODUCT_WIDTH = 2 * INPUT_WIDTH;
logic signed [PRODUCT_WIDTH-1:0] product_s1;
logic                                  valid_s1;
// valid_s1 延迟 1 拍驱动 out_valid 与累加
always_ff @(posedge clk) begin
    if (!rst_n) begin
        product_s1 <= '0;
        valid_s1   <= 1'b0;
        acc_out    <= '0;
        out_valid  <= 1'b0;
    end else if (clear) begin
        product_s1 <= '0;
        valid_s1   <= 1'b0;
        acc_out    <= '0;
        out_valid  <= 1'b0;
    end else if (enable) begin
        valid_s1  <= in_valid;
        out_valid <= valid_s1;
        if (in_valid) begin
            product_s1 <= activation * weight;
        end
        if (valid_s1) begin
            acc_out <= $signed(acc_out) + $signed(product_s1);
        end
    end else begin
        product_s1 <= product_s1;
        valid_s1   <= valid_s1;
        acc_out    <= acc_out;
        out_valid  <= 1'b0;
    end
end
endmodule