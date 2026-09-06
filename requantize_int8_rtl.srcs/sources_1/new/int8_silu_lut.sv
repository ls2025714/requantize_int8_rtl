// ============================================================================
// 文件: int8_silu_lut.sv
// 学习阶段: D11 SiLU（SwiGLU 里对 gate 做激活）
// ----------------------------------------------------------------------------
// 【在整条链的位置】
//   D12: GATE(Linear) → 本模块 SiLU(gate) → 再与 UP 做 elem_mul → DOWN
//   不是经典 FFN 的 “Linear→Act→Linear”，而是 SwiGLU 五段之一
//
// 【实现】256 项冻结 LUT；索引 = in_data + 128（把 [-128,127] 映到 [0,255]）
//   1 拍流水寄存；背压: in_ready = !pipe_valid || out_ready
//   学习注意：一进一出，勿在 FEED 时长期拉低 out_ready 堵死
//
// 【参考】silu_lut_params.svh（`include）；scripts/silu_fixed_ref.py
// 【验证】tb_int8_silu_lut.sv
// ============================================================================
//
module int8_silu_lut #(
    parameter int DATA_WIDTH = 8,
    parameter int IDX_WIDTH  = 10
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          in_valid,
    output logic                          in_ready,
    input  logic signed [DATA_WIDTH-1:0]  in_data,
    input  logic [IDX_WIDTH-1:0]          in_idx,
    output logic                          out_valid,
    input  logic                          out_ready,
    output logic signed [DATA_WIDTH-1:0]  out_data,
    output logic [IDX_WIDTH-1:0]          out_idx
);
    logic signed [DATA_WIDTH-1:0] lut [0:255];
    logic pipe_valid;
    logic signed [DATA_WIDTH-1:0] pipe_data;
    logic [IDX_WIDTH-1:0] pipe_idx;
    logic [7:0] lut_idx;

    // --- LUT 装载（auto-generated，勿手改数组体）---
    initial begin
        `include "silu_lut_params.svh"
    end

    // --- 握手 ---
    assign in_ready  = !pipe_valid || out_ready;
    assign out_valid = pipe_valid;
    assign out_data  = pipe_data;
    assign out_idx   = pipe_idx;

    // --- 1 拍查表流水 ---
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_valid <= 1'b0;
            pipe_data  <= '0;
            pipe_idx   <= '0;
        end else begin
            if (out_ready)
                pipe_valid <= 1'b0;
            if (in_valid && in_ready) begin
                lut_idx    = 8'(in_data + 8'sd128);
                pipe_data  <= lut[lut_idx];
                pipe_idx   <= in_idx;
                pipe_valid <= 1'b1;
            end
        end
    end
endmodule
