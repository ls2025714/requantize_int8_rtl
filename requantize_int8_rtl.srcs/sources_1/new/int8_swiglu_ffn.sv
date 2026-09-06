// ============================================================================
// 文件: int8_swiglu_ffn.sv
// 学习阶段: D11 SwiGLU FFN（文档/占位；真编排在 D12）
// ----------------------------------------------------------------------------
// 【重要】本文件是 stub，不包含可综合数据通路。
//   真正的 gate/up/SiLU/elem_mul/down 由 int8_transformer_block 实例化并调度：
//     Res1 → GATE(tiled) → UP(tiled) → SiLU(gate) → elem_mul → DOWN(tiled) → Res2
//   shape: n_embd=64 → d_ff=256 → n_embd=64
//
// 【学习时】对照 D12 的 ST_GATE / ST_UP / ST_SILU / ST_ELMUL / ST_DOWN，
//   不要期望在本文件里找到完整 FSM。
// 【验证】tb_linear_tiled_ffn.sv（分层）/ tb_int8_transformer_block.sv（E2E）
// ============================================================================
//
module int8_swiglu_ffn #(
    parameter int EMBD = 64,
    parameter int D_FF = 256,
    parameter int MAX_M = 4
)(
    input logic clk,
    input logic rst_n
);
    // Placeholder: D12 block top wires gate/up/down tiled linears + silu + elem_mul.
endmodule
