// ============================================================================
// 文件: int8_dot_product_parallel.sv
// 学习阶段: D2 四路并行点积（Task B/C/D 核心）
// ----------------------------------------------------------------------------
// 【相对串行】每拍最多 4 对 INT8；beat_total=ceil(K/4)；尾拍用 s_keep 掩无效 lane
// 【位宽树】必须会推导:
//   4×16-bit Product → 2×17-bit Pair Sum（先符号扩展）→ 18-bit Partial → INT32 Acc
// 【流水】S1乘 → S2 pair → S3 partial → S4 acc；DRAIN 等 pipeline 清空后再 OUTPUT
// 【FSM】IDLE → RUN → DRAIN → OUTPUT（输出阻塞时 m_valid/m_result 保持）
// 【谁用】int8_gemm_parallel、int8_score_gemm 等
// 【验证】tb_int8_dot_product_parallel / tb_dot_product_parallel_python_vectors（120/120）
// ============================================================================
//
// debug_mode: IDLE 且无 cmd 时可用 acc_enable 单步灌 partial（Task B）
// partial_*/acc_* 为 debug 口，功能口只需 m_*
//
module int8_dot_product_parallel #(
    parameter int INPUT_WIDTH     = 8,
    parameter int PRODUCT_WIDTH   = 16,
    parameter int PAIR_WIDTH      = 17,
    parameter int PARTIAL_WIDTH   = 18,
    parameter int ACC_WIDTH       = 32,
    parameter int PARALLELISM     = 4,
    parameter int MAX_K           = 256,
    parameter int LENGTH_WIDTH    = $clog2(MAX_K + 1),
    parameter int PARTIAL_LATENCY = 2,
    parameter int ACC_LATENCY     = 3
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          cmd_valid,
    output logic                          cmd_ready,
    input  logic [LENGTH_WIDTH-1:0]       cmd_length,
    input  logic                          s_valid,
    output logic                          s_ready,
    input  logic signed [INPUT_WIDTH-1:0] s_a0,
    input  logic signed [INPUT_WIDTH-1:0] s_a1,
    input  logic signed [INPUT_WIDTH-1:0] s_a2,
    input  logic signed [INPUT_WIDTH-1:0] s_a3,
    input  logic signed [INPUT_WIDTH-1:0] s_b0,
    input  logic signed [INPUT_WIDTH-1:0] s_b1,
    input  logic signed [INPUT_WIDTH-1:0] s_b2,
    input  logic signed [INPUT_WIDTH-1:0] s_b3,
    input  logic [PARALLELISM-1:0]        s_keep,
    input  logic                          acc_clear,
    input  logic                          acc_enable,
    output logic                          m_valid,
    input  logic                          m_ready,
    output logic signed [ACC_WIDTH-1:0]   m_result,
    output logic                          partial_valid,
    output logic signed [PARTIAL_WIDTH-1:0] partial_sum,
    output logic                          acc_valid,
    output logic signed [ACC_WIDTH-1:0]   acc_value
);
    typedef enum logic [1:0] {IDLE, RUN, DRAIN, OUTPUT} state_t;

    state_t state;
    logic [LENGTH_WIDTH-1:0] length_reg;
    logic [LENGTH_WIDTH-1:0] beat_total_reg;
    logic [LENGTH_WIDTH-1:0] beat_count;
    logic [LENGTH_WIDTH-1:0] acc_count;
    logic signed [ACC_WIDTH-1:0] result_reg;

    logic signed [PRODUCT_WIDTH-1:0] product0_s1;
    logic signed [PRODUCT_WIDTH-1:0] product1_s1;
    logic signed [PRODUCT_WIDTH-1:0] product2_s1;
    logic signed [PRODUCT_WIDTH-1:0] product3_s1;
    logic                            valid_s1;

    logic signed [PAIR_WIDTH-1:0] pair_sum0_s2;
    logic signed [PAIR_WIDTH-1:0] pair_sum1_s2;
    logic                         valid_s2;

    logic signed [PARTIAL_WIDTH-1:0] partial_sum_s3;
    logic                            valid_s3;

    logic signed [ACC_WIDTH-1:0] acc_reg;
    logic                        valid_s4;

    logic cmd_accept;
    logic input_accept;
    logic debug_mode;
    logic pipeline_clear;
    logic pipeline_enable;

    logic signed [PRODUCT_WIDTH-1:0] product0_in;
    logic signed [PRODUCT_WIDTH-1:0] product1_in;
    logic signed [PRODUCT_WIDTH-1:0] product2_in;
    logic signed [PRODUCT_WIDTH-1:0] product3_in;
    logic signed [PAIR_WIDTH-1:0]    product0_ext17;
    logic signed [PAIR_WIDTH-1:0]    product1_ext17;
    logic signed [PAIR_WIDTH-1:0]    product2_ext17;
    logic signed [PAIR_WIDTH-1:0]    product3_ext17;
    logic signed [PAIR_WIDTH-1:0]    pair_sum0_comb;
    logic signed [PAIR_WIDTH-1:0]    pair_sum1_comb;
    logic signed [PARTIAL_WIDTH-1:0] pair_sum0_ext18;
    logic signed [PARTIAL_WIDTH-1:0] pair_sum1_ext18;
    logic signed [PARTIAL_WIDTH-1:0] partial_sum_comb;

    // --- 流控与握手 ---
    assign debug_mode      = (state == IDLE) && !cmd_valid;
    assign cmd_ready       = (state == IDLE) && (cmd_length != '0) && (cmd_length <= MAX_K);
    assign cmd_accept      = cmd_valid && cmd_ready;
    assign s_ready         = ((state == RUN) && (beat_count < beat_total_reg)) ||
                             (debug_mode && acc_enable);
    assign input_accept    = s_valid && s_ready;
    assign m_valid         = (state == OUTPUT);
    assign m_result        = result_reg;
    assign pipeline_clear  = cmd_accept || acc_clear;
    assign pipeline_enable = (state == RUN) || (state == DRAIN) ||
                             (debug_mode && acc_enable);

    // --- 4-lane 乘法（keep 掩码）---
    assign product0_in = s_keep[0] ? (s_a0 * s_b0) : PRODUCT_WIDTH'(0);
    assign product1_in = s_keep[1] ? (s_a1 * s_b1) : PRODUCT_WIDTH'(0);
    assign product2_in = s_keep[2] ? (s_a2 * s_b2) : PRODUCT_WIDTH'(0);
    assign product3_in = s_keep[3] ? (s_a3 * s_b3) : PRODUCT_WIDTH'(0);

    // --- 组合加：16→17→18→32 位宽扩展 ---
    assign product0_ext17 = {{1{product0_s1[PRODUCT_WIDTH-1]}}, product0_s1};
    assign product1_ext17 = {{1{product1_s1[PRODUCT_WIDTH-1]}}, product1_s1};
    assign product2_ext17 = {{1{product2_s1[PRODUCT_WIDTH-1]}}, product2_s1};
    assign product3_ext17 = {{1{product3_s1[PRODUCT_WIDTH-1]}}, product3_s1};
    assign pair_sum0_comb = product0_ext17 + product1_ext17;
    assign pair_sum1_comb = product2_ext17 + product3_ext17;

    assign pair_sum0_ext18 = {{1{pair_sum0_s2[PAIR_WIDTH-1]}}, pair_sum0_s2};
    assign pair_sum1_ext18 = {{1{pair_sum1_s2[PAIR_WIDTH-1]}}, pair_sum1_s2};
    assign partial_sum_comb = pair_sum0_ext18 + pair_sum1_ext18;

    assign partial_valid = valid_s3;
    assign partial_sum   = partial_sum_s3;
    assign acc_valid     = valid_s4;
    assign acc_value     = acc_reg;

    // --- 主 FSM：beat/acc 计数，RUN→DRAIN→OUTPUT ---
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state          <= IDLE;
            length_reg     <= '0;
            beat_total_reg <= '0;
            beat_count     <= '0;
            acc_count      <= '0;
            result_reg     <= '0;
        end else if (acc_clear) begin
            state      <= IDLE;
            beat_count <= '0;
            acc_count  <= '0;
        end else begin
            case (state)
                IDLE: begin
                    beat_count <= '0;
                    acc_count  <= '0;
                    if (cmd_accept) begin
                        length_reg     <= cmd_length;
                        beat_total_reg <= (cmd_length + PARALLELISM - 1) >> 2;
                        state          <= RUN;
                    end
                end
                RUN: begin
                    if (input_accept) begin
                        if (beat_count == beat_total_reg - 1'b1) begin
                            state <= DRAIN;
                        end else begin
                            beat_count <= beat_count + 1'b1;
                        end
                    end
                    if (acc_valid) begin
                        if (acc_count == beat_total_reg - 1'b1) begin
                            result_reg <= acc_reg;
                            state      <= OUTPUT;
                        end else begin
                            acc_count <= acc_count + 1'b1;
                        end
                    end
                end
                DRAIN: begin
                    if (acc_valid) begin
                        if (acc_count == beat_total_reg - 1'b1) begin
                            result_reg <= acc_reg;
                            state      <= OUTPUT;
                        end else begin
                            acc_count <= acc_count + 1'b1;
                        end
                    end
                end
                OUTPUT: begin
                    if (m_ready) begin
                        state <= IDLE;
                    end
                end
                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end

    // --- 流水线 S1：寄存 4 路乘积 ---
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            product0_s1 <= '0;
            product1_s1 <= '0;
            product2_s1 <= '0;
            product3_s1 <= '0;
            valid_s1    <= 1'b0;
        end else if (pipeline_clear) begin
            valid_s1 <= 1'b0;
        end else begin
            valid_s1 <= input_accept;
            if (input_accept) begin
                product0_s1 <= product0_in;
                product1_s1 <= product1_in;
                product2_s1 <= product2_in;
                product3_s1 <= product3_in;
            end
        end
    end

    // --- 流水线 S2：pair sum ---
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            pair_sum0_s2 <= '0;
            pair_sum1_s2 <= '0;
            valid_s2     <= 1'b0;
        end else if (pipeline_clear) begin
            valid_s2 <= 1'b0;
        end else begin
            valid_s2 <= valid_s1;
            if (valid_s1) begin
                pair_sum0_s2 <= pair_sum0_comb;
                pair_sum1_s2 <= pair_sum1_comb;
            end
        end
    end

    // --- 流水线 S3：partial sum ---
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            partial_sum_s3 <= '0;
            valid_s3       <= 1'b0;
        end else if (pipeline_clear) begin
            valid_s3 <= 1'b0;
        end else begin
            valid_s3 <= valid_s2;
            if (valid_s2) begin
                partial_sum_s3 <= partial_sum_comb;
            end
        end
    end

    // --- 流水线 S4：累加器 ---
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            acc_reg  <= '0;
            valid_s4 <= 1'b0;
        end else if (pipeline_clear) begin
            acc_reg  <= '0;
            valid_s4 <= 1'b0;
        end else begin
            valid_s4 <= pipeline_enable && valid_s3;
            if (pipeline_enable && valid_s3) begin
                acc_reg <= acc_reg + {{(ACC_WIDTH-PARTIAL_WIDTH){partial_sum_s3[PARTIAL_WIDTH-1]}}, partial_sum_s3};
            end
        end
    end

endmodule
