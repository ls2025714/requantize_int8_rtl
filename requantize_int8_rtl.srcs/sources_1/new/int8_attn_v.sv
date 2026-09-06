// ============================================================================
// 文件: int8_attn_v.sv
// 学习阶段: D10 Attention @ V（概率加权求和）
// ----------------------------------------------------------------------------
// 【在整条链的位置】
//   D9 Softmax 产出 P[seq,seq]（UQ1.15）；本模块再乘 V[seq,head_dim] INT8
//   单 head 输出 Y[seq,head_dim] INT8；多 head 在 D12 里 CONCAT 后再走 Wo
//   Wo 本身复用 int8_linear_tiled（N=K=64），不在本文件
//
// 【数学】
//   Y[i][d] = saturate( round( Σ_j P[i][j] * V[j][d] ) >>> 15 )
//   P 无符号扩展为有符号再乘；累加用较宽 ACC（40-bit）防溢出
//
// 【FSM】IDLE→LOAD_P→LOAD_V→对每个 (i,d): MAC(沿 j)→SETTLE→QUANT→EMIT
// 【验证】tb_int8_attn_v.sv
// ============================================================================
//
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

    // IDLE: 等 seq；LOAD_P/V: 装概率与 V
    // MAC: 固定 (i,d)，j 从 0..seq-1 累加乘积
    // SETTLE: 空一拍对齐；QUANT: round>>>15+饱和；EMIT: 吐 Y[i][d]，推进 d 或 i
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

    // --- 主 FSM：LOAD_P/V → MAC 累加 → QUANT → EMIT ---
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

                // 按 p_row/p_col 写入整张概率表
                LOAD_P: begin
                    if (p_valid && p_ready) begin
                        p_mem[p_row][p_col] <= p_data;
                        if ((p_row == seq_reg - 1'b1) && (p_col == seq_reg - 1'b1)) begin
                            v_count <= '0;
                            state   <= LOAD_V;
                        end
                    end
                end

                // 行优先灌 V；满后从 (i,d,j)=(0,0,0) 开始 MAC
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

                // 一拍累加一项 P[i][j]*V[j][d]；j 走完 → SETTLE
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

                // 与 requant 同类的 round + 算术右移 + 对称饱和 → INT8
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

                // 取走后：同 token 下一通道 d++；否则下一 token i++；全完回 IDLE
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
