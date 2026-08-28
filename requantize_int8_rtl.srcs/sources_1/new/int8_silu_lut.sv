// ============================================================================
// int8_silu_lut.sv — D11 SiLU via frozen 256-entry LUT (INT8 in/out)
// ============================================================================
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

    initial begin
        `include "silu_lut_params.svh"
    end

    assign in_ready  = !pipe_valid || out_ready;
    assign out_valid = pipe_valid;
    assign out_data  = pipe_data;
    assign out_idx   = pipe_idx;

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
