// ============================================================================
// int8_swiglu_ffn.sv — D11/D12 SwiGLU FFN wrapper (gate/up SiLU*mul/down)
// Top FSM in int8_transformer_block orchestrates; this module documents ports.
// ============================================================================
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
