// ============================================================================
// int8_elem_mul.sv — D11 element-wise INT8 multiply with >>7 rounding/sat
// ============================================================================
module int8_elem_mul #(
    parameter int DATA_WIDTH = 8,
    parameter int IDX_WIDTH  = 10,
    parameter int SHIFT_BITS = 7
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          cmd_valid,
    output logic                          cmd_ready,
    input  logic [IDX_WIDTH-1:0]          cmd_len,
    input  logic                          a_valid,
    output logic                          a_ready,
    input  logic signed [DATA_WIDTH-1:0]  a_data,
    input  logic                          b_valid,
    output logic                          b_ready,
    input  logic signed [DATA_WIDTH-1:0]  b_data,
    output logic                          z_valid,
    input  logic                          z_ready,
    output logic signed [DATA_WIDTH-1:0]  z_data,
    output logic [IDX_WIDTH-1:0]          z_idx
);
    typedef enum logic [1:0] {IDLE, WAIT_A, WAIT_B, EMIT} state_t;
    state_t state;
    logic [IDX_WIDTH-1:0] len_reg, idx;
    logic signed [DATA_WIDTH-1:0] a_hold, b_hold, z_hold;
    logic signed [2*DATA_WIDTH-1:0] prod_w, rounded;
    localparam logic signed [2*DATA_WIDTH-1:0] HALF = (1 << (SHIFT_BITS - 1));

    assign cmd_ready = (state == IDLE);
    assign a_ready   = (state == WAIT_A);
    assign b_ready   = (state == WAIT_B);
    assign z_data    = z_hold;
    assign z_idx     = idx;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state   <= IDLE;
            len_reg <= '0;
            idx     <= '0;
            z_hold  <= '0;
            z_valid <= 1'b0;
        end else begin
            z_valid <= 1'b0;
            case (state)
                IDLE: begin
                    idx <= '0;
                    if (cmd_valid && cmd_ready) begin
                        len_reg <= cmd_len;
                        state   <= WAIT_A;
                    end
                end
                WAIT_A: begin
                    if (a_valid && a_ready) begin
                        a_hold <= a_data;
                        state  <= WAIT_B;
                    end
                end
                WAIT_B: begin
                    if (b_valid && b_ready) begin
                        b_hold <= b_data;
                        prod_w = $signed(a_hold) * $signed(b_data);
                        if (prod_w >= 0)
                            rounded = prod_w + HALF;
                        else
                            rounded = prod_w + (HALF - 1);
                        if ((rounded >>> SHIFT_BITS) > 127)
                            z_hold <= 8'sd127;
                        else if ((rounded >>> SHIFT_BITS) < -127)
                            z_hold <= -8'sd127;
                        else
                            z_hold <= $signed(rounded >>> SHIFT_BITS);
                        state <= EMIT;
                    end
                end
                EMIT: begin
                    z_valid <= 1'b1;
                    if (z_ready) begin
                        if (idx == len_reg - 1'b1)
                            state <= IDLE;
                        else begin
                            idx   <= idx + 1'b1;
                            state <= WAIT_A;
                        end
                    end
                end
            endcase
        end
    end
endmodule
