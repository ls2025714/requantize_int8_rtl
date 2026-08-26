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
always_comb begin
    if (product_s1 >= 0)
        rounded_comb = product_s1 + HALF_LSB;
    else
        rounded_comb = product_s1 + (HALF_LSB - 50'sd1);
    shifted_comb = rounded_comb >>> SHIFT_BITS;
end
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