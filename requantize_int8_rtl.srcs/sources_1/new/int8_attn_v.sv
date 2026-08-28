// ============================================================================
// 文件: int8_attn_v.sv
// 阶段: D10 Attention Probability @ V
// 作用: Y[i][d] = saturate(round(sum_j P[i][j]*V[j][d]) >> 15)
//       P = UQ1.15, V = INT8, Y = INT8
// 验证: tb_int8_attn_v.sv
// ============================================================================

module int8_attn_v #(
    parameter int INPUT_WIDTH = 8,
    parameter int P_WIDTH     = 16,
    parameter int ACC_WIDTH   = 40,
    parameter int MAX_SEQ     = 4,
    parameter int HEAD_DIM    = 16,
    parameter int SEQ_WIDTH   = $clog2(MAX_SEQ + 1),
    parameter int D_WIDTH     = $clog2(HEAD_DIM + 1),
    parameter int SHIFT_BITS  = 15
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          cmd_valid,
    output logic                          cmd_ready,
    input  logic [SEQ_WIDTH-1:0]          cmd_seq,
    input  logic                          p_valid,
    output logic                          p_ready,
    input  logic [P_WIDTH-1:0]            p_data,
    input  logic [SEQ_WIDTH-1:0]          p_row,
    input  logic [SEQ_WIDTH-1:0]          p_col,
    input  logic                          v_valid,
    output logic                          v_ready,
    input  logic signed [INPUT_WIDTH-1:0] v_data,
    output logic                          y_valid,
    input  logic                          y_ready,
    output logic signed [INPUT_WIDTH-1:0] y_data,
    output logic [SEQ_WIDTH-1:0]          y_row,
    output logic [D_WIDTH-1:0]            y_col
);
    localparam int V_COUNT_W = $clog2(MAX_SEQ * HEAD_DIM + 1);
    localparam logic signed [ACC_WIDTH-1:0] HALF = ACC_WIDTH'(1) <<< (SHIFT_BITS - 1);

    typedef enum logic [2:0] {IDLE, LOAD_P, LOAD_V, MAC, SETTLE, QUANT, EMIT} state_t;

    state_t state;
    logic [SEQ_WIDTH-1:0] seq_reg;
    logic [SEQ_WIDTH-1:0] i_idx;
    logic [SEQ_WIDTH-1:0] j_idx;
    logic [D_WIDTH-1:0] d_idx;
    logic [V_COUNT_W-1:0] v_count;
    logic [V_COUNT_W-1:0] v_total;
    logic [P_WIDTH-1:0] p_mem [0:MAX_SEQ-1][0:MAX_SEQ-1];
    logic signed [INPUT_WIDTH-1:0] v_mem [0:MAX_SEQ-1][0:HEAD_DIM-1];
    logic signed [ACC_WIDTH-1:0] acc;
    logic signed [INPUT_WIDTH-1:0] y_hold;
    logic signed [ACC_WIDTH-1:0] prod;
    logic signed [ACC_WIDTH-1:0] rounded;
    logic signed [ACC_WIDTH-1:0] shifted;

    assign cmd_ready = (state == IDLE);
    assign p_ready   = (state == LOAD_P);
    assign v_ready   = (state == LOAD_V);
    assign y_valid   = (state == EMIT);
    assign y_data    = y_hold;
    assign y_row     = i_idx;
    assign y_col     = d_idx;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state   <= IDLE;
            seq_reg <= '0;
            i_idx   <= '0;
            j_idx   <= '0;
            d_idx   <= '0;
            v_count <= '0;
            v_total <= '0;
            acc     <= '0;
            y_hold  <= '0;
        end else begin
            unique case (state)
                IDLE: begin
                    if (cmd_valid && cmd_ready) begin
                        seq_reg <= cmd_seq;
                        v_total <= V_COUNT_W'(cmd_seq) * V_COUNT_W'(HEAD_DIM);
                        v_count <= '0;
                        state   <= LOAD_P;
                    end
                end

                LOAD_P: begin
                    if (p_valid && p_ready) begin
                        p_mem[p_row][p_col] <= p_data;
                        if ((p_row == seq_reg - 1'b1) && (p_col == seq_reg - 1'b1)) begin
                            v_count <= '0;
                            state   <= LOAD_V;
                        end
                    end
                end

                LOAD_V: begin
                    if (v_valid && v_ready) begin
                        v_mem[v_count / V_COUNT_W'(HEAD_DIM)]
                             [v_count % V_COUNT_W'(HEAD_DIM)] <= v_data;
                        if (v_count == v_total - 1'b1) begin
                            i_idx <= '0;
                            d_idx <= '0;
                            j_idx <= '0;
                            acc   <= '0;
                            state <= MAC;
                        end else begin
                            v_count <= v_count + 1'b1;
                        end
                    end
                end

                MAC: begin
                    prod = $signed({1'b0, p_mem[i_idx][j_idx]}) * $signed(v_mem[j_idx][d_idx]);
                    acc  <= acc + prod;
                    if (j_idx == seq_reg - 1'b1) begin
                        state <= SETTLE;
                    end else begin
                        j_idx <= j_idx + 1'b1;
                    end
                end

                SETTLE: begin
                    state <= QUANT;
                end

                QUANT: begin
                    if (acc >= 0)
                        rounded = acc + HALF;
                    else
                        rounded = acc + (HALF - ACC_WIDTH'(1));
                    shifted = rounded >>> SHIFT_BITS;
                    if (shifted > ACC_WIDTH'(127))
                        y_hold <= 8'sd127;
                    else if (shifted < -ACC_WIDTH'(127))
                        y_hold <= -8'sd127;
                    else
                        y_hold <= $signed(shifted[7:0]);
                    state <= EMIT;
                end

                EMIT: begin
                    if (y_valid && y_ready) begin
                        if (d_idx == HEAD_DIM - 1) begin
                            if (i_idx == seq_reg - 1'b1) begin
                                state <= IDLE;
                            end else begin
                                i_idx <= i_idx + 1'b1;
                                d_idx <= '0;
                                j_idx <= '0;
                                acc   <= '0;
                                state <= MAC;
                            end
                        end else begin
                            d_idx <= d_idx + 1'b1;
                            j_idx <= '0;
                            acc   <= '0;
                            state <= MAC;
                        end
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end
endmodule
