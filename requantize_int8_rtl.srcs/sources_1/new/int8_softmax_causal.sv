// ============================================================================
// 文件: int8_softmax_causal.sv
// 学习阶段: D9 Scale + Causal Mask + 定点 Softmax
// ----------------------------------------------------------------------------
// 【在整条链的位置】
//   D8 score_gemm 吐出 Score[seq,seq] INT32
//   D9 本模块：按行做 scale → causal mask → 减 max → exp LUT → 归一化
//   输出 UQ1.15 概率，供 D10 Attn@V 使用
//
// 【务必对齐的参考】
//   Python/RTL 同一套定点：scripts/softmax_fixed_ref.py（bit-exact）
//   不要另开“通用 Softmax 近似”来学；以本仓库冻结算法为准
//
// 【每行算法人话】
//   1) SCALE:  s' = (score * SOFT_SCALE_MULT) >>> SOFT_SCALE_SHIFT
//   2) MASK:   若 col > row（未来 token）→ 置成极负 SOFT_MASK_VAL
//   3) MAX:    只在有效列上找行最大值（数值稳定，不改变 Softmax 相对关系）
//   4) EXP:    clamp(s'-max) 查 SOFT_EXP_LUT，累加 sum_e；上三角 exp=0
//   5) RECIP:  inv_q ≈ 2^31 / sum_e
//   6) EMIT:   P = (exp * inv_q) >> 16；上三角强制 0
//
// 【FSM】IDLE→LOAD→每行(ROW_INIT→SCALE→MAX→EXP→RECIP→EMIT)→下一行或 IDLE
// 【验证】tb_int8_softmax_causal.sv；seed=20260902，6/6 PASS
// ============================================================================
//
module int8_softmax_causal #(
    parameter int ACC_WIDTH = 32,
    parameter int MAX_SEQ   = 4,
    parameter int P_WIDTH   = 16,
    parameter int SEQ_WIDTH = $clog2(MAX_SEQ + 1)
)(
    input  logic                        clk,
    input  logic                        rst_n,
    input  logic                        cmd_valid,
    output logic                        cmd_ready,
    input  logic [SEQ_WIDTH-1:0]        cmd_seq,
    input  logic                        in_valid,
    output logic                        in_ready,
    input  logic signed [ACC_WIDTH-1:0] in_data,
    input  logic [SEQ_WIDTH-1:0]        in_row,
    input  logic [SEQ_WIDTH-1:0]        in_col,
    output logic                        out_valid,
    input  logic                        out_ready,
    output logic [P_WIDTH-1:0]          out_data,
    output logic [SEQ_WIDTH-1:0]        out_row,
    output logic [SEQ_WIDTH-1:0]        out_col
);
    localparam int SOFT_SCALE_MULT  = 1;
    localparam int SOFT_SCALE_SHIFT = 2;
    localparam logic signed [31:0] SOFT_MASK_VAL = -32'sd1073741824;
    localparam logic signed [31:0] SOFT_EXP_MIN = -32'sd32;
    localparam logic signed [31:0] SOFT_EXP_MAX = 32'sd0;
    localparam logic [14:0] SOFT_EXP_LUT [0:32] = '{
        15'd0, 15'd0, 15'd0, 15'd0, 15'd0, 15'd0, 15'd0, 15'd0, 15'd0, 15'd0,
        15'd0, 15'd0, 15'd0, 15'd0, 15'd0, 15'd0, 15'd0, 15'd0, 15'd0, 15'd0,
        15'd0, 15'd1, 15'd1, 15'd4, 15'd11, 15'd30, 15'd81, 15'd221, 15'd600,
        15'd1631, 15'd4435, 15'd12054, 15'd32767
    };

    // IDLE: 等 cmd_seq
    // LOAD: 按 in_row/in_col 写入整张 Score
    // ROW_*: 对 work_row 做完一行 Softmax 后 work_row++，直到最后一行
    typedef enum logic [3:0] {
        IDLE, LOAD, ROW_INIT, ROW_SCALE, ROW_MAX, ROW_EXP, ROW_RECIP, ROW_EMIT
    } state_t;

    state_t state;
    logic [SEQ_WIDTH-1:0] seq_reg;
    logic [SEQ_WIDTH-1:0] work_row;
    logic [SEQ_WIDTH-1:0] col_i;
    logic signed [ACC_WIDTH-1:0] score_mem [0:MAX_SEQ-1][0:MAX_SEQ-1];
    logic signed [ACC_WIDTH-1:0] scaled_mem [0:MAX_SEQ-1];
    logic [14:0] exp_mem [0:MAX_SEQ-1];
    logic signed [ACC_WIDTH-1:0] row_max;
    logic [31:0] sum_e;
    logic [31:0] inv_q;
    logic [P_WIDTH-1:0] out_data_comb;
    logic signed [ACC_WIDTH-1:0] tmp_scaled;
    logic signed [ACC_WIDTH-1:0] x_diff;
    logic [5:0] lut_idx;
    logic max_valid;

    assign cmd_ready = (state == IDLE);
    assign in_ready  = (state == LOAD);
    assign out_valid = (state == ROW_EMIT);
    assign out_row   = work_row;
    assign out_col   = col_i;
    assign out_data  = out_data_comb;

    // --- EMIT 组合：因果上三角置 0，否则 (exp*inv)>>16 ---
    always_comb begin
        if (state != ROW_EMIT)
            out_data_comb = '0;
        else if (col_i > work_row)
            out_data_comb = '0;
        else
            out_data_comb = P_WIDTH'((32'(exp_mem[col_i]) * inv_q) >> 16);
    end

    // --- 主 FSM：LOAD + 按行 Softmax ---
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state     <= IDLE;
            seq_reg   <= '0;
            work_row  <= '0;
            col_i     <= '0;
            row_max   <= '0;
            sum_e     <= '0;
            inv_q     <= '0;
            max_valid <= 1'b0;
        end else begin
            unique case (state)
                // 锁存序列长度，开始灌 Score
                IDLE: begin
                    if (cmd_valid && cmd_ready) begin
                        seq_reg <= cmd_seq;
                        state   <= LOAD;
                    end
                end

                // 收满 Score[seq-1][seq-1] 后从第 0 行 Softmax
                LOAD: begin
                    if (in_valid && in_ready) begin
                        score_mem[in_row][in_col] <= in_data;
                        if ((in_row == seq_reg - 1'b1) && (in_col == seq_reg - 1'b1)) begin
                            work_row <= '0;
                            state    <= ROW_INIT;
                        end
                    end
                end

                // 清本行工作变量，进入 SCALE
                ROW_INIT: begin
                    col_i     <= '0;
                    sum_e     <= '0;
                    max_valid <= 1'b0;
                    row_max   <= '0;
                    state     <= ROW_SCALE;
                end

                // 逐列：未来位置写 MASK；否则乘 scale 右移。扫完 → ROW_MAX
                ROW_SCALE: begin
                    if (col_i > work_row) begin
                        scaled_mem[col_i] <= SOFT_MASK_VAL;
                    end else begin
                        tmp_scaled = (score_mem[work_row][col_i] * ACC_WIDTH'(SOFT_SCALE_MULT))
                                     >>> SOFT_SCALE_SHIFT;
                        scaled_mem[col_i] <= tmp_scaled;
                    end
                    if (col_i == seq_reg - 1'b1) begin
                        col_i <= '0;
                        state <= ROW_MAX;
                    end else begin
                        col_i <= col_i + 1'b1;
                    end
                end

                // 只在因果可见列 (col<=row) 上找最大值
                ROW_MAX: begin
                    if (col_i <= work_row) begin
                        if (!max_valid || (scaled_mem[col_i] > row_max)) begin
                            row_max   <= scaled_mem[col_i];
                            max_valid <= 1'b1;
                        end
                    end
                    if (col_i == seq_reg - 1'b1) begin
                        col_i <= '0;
                        sum_e <= '0;
                        state <= ROW_EXP;
                    end else begin
                        col_i <= col_i + 1'b1;
                    end
                end

                // 查 LUT 得 exp，累加 sum_e；未来列 exp=0
                ROW_EXP: begin
                    if (col_i > work_row) begin
                        exp_mem[col_i] <= '0;
                    end else begin
                        x_diff = scaled_mem[col_i] - row_max;
                        if (x_diff < SOFT_EXP_MIN)
                            x_diff = SOFT_EXP_MIN;
                        if (x_diff > SOFT_EXP_MAX)
                            x_diff = SOFT_EXP_MAX;
                        lut_idx = 6'(x_diff - SOFT_EXP_MIN);
                        exp_mem[col_i] <= SOFT_EXP_LUT[lut_idx];
                        sum_e <= sum_e + 32'(SOFT_EXP_LUT[lut_idx]);
                    end
                    if (col_i == seq_reg - 1'b1) begin
                        state <= ROW_RECIP;
                    end else begin
                        col_i <= col_i + 1'b1;
                    end
                end

                // 一行一个倒数；再逐列发射概率
                ROW_RECIP: begin
                    if (sum_e == 32'd0)
                        inv_q <= 32'd0;
                    else
                        inv_q <= (32'h8000_0000) / sum_e;
                    col_i <= '0;
                    state <= ROW_EMIT;
                end

                // 下游握手吐出本行各列 P；行完 → 下一行 ROW_INIT；全完 → IDLE
                ROW_EMIT: begin
                    if (out_valid && out_ready) begin
                        if (col_i == seq_reg - 1'b1) begin
                            if (work_row == seq_reg - 1'b1) begin
                                state <= IDLE;
                            end else begin
                                work_row <= work_row + 1'b1;
                                state    <= ROW_INIT;
                            end
                        end else begin
                            col_i <= col_i + 1'b1;
                        end
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end
endmodule
