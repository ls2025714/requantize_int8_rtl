// ============================================================================
// 文件: int8_score_gemm.sv
// 阶段: D8 Attention Score Q @ K^T
// 作用: 单 head Score[seq,seq] = Q[seq,d] @ K[seq,d]^T，INT32 输出
//       K 仅按行寻址实现转置视角，不复制矩阵
// 依赖: int8_dot_product_parallel
// 验证: tb_int8_score_gemm.sv
// ============================================================================

module int8_score_gemm #(
    parameter int INPUT_WIDTH = 8,
    parameter int ACC_WIDTH   = 32,
    parameter int MAX_SEQ     = 4,
    parameter int HEAD_DIM    = 16,
    parameter int PARALLELISM = 4,
    parameter int SEQ_WIDTH   = $clog2(MAX_SEQ + 1),
    parameter int K_WIDTH     = $clog2(HEAD_DIM + 1)
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          cmd_valid,
    output logic                          cmd_ready,
    input  logic [SEQ_WIDTH-1:0]          cmd_seq,
    input  logic                          q_valid,
    output logic                          q_ready,
    input  logic signed [INPUT_WIDTH-1:0] q_data,
    input  logic                          k_valid,
    output logic                          k_ready,
    input  logic signed [INPUT_WIDTH-1:0] k_data,
    output logic                          s_valid,
    input  logic                          s_ready,
    output logic signed [ACC_WIDTH-1:0]   s_data,
    output logic [SEQ_WIDTH-1:0]          s_row,
    output logic [SEQ_WIDTH-1:0]          s_col
);
    localparam int MEM_DEPTH       = MAX_SEQ * HEAD_DIM;
    localparam int COUNT_WIDTH     = $clog2(MEM_DEPTH + 1);
    localparam int ADDR_WIDTH      = (MEM_DEPTH <= 1) ? 1 : $clog2(MEM_DEPTH);

    typedef enum logic [2:0] {
        IDLE, LOAD_Q, LOAD_K, DOT_CMD, DOT_FEED, DOT_WAIT, OUTPUT_S
    } state_t;

    state_t state;
    logic [SEQ_WIDTH-1:0] seq_reg;
    logic [COUNT_WIDTH-1:0] load_total;
    logic [COUNT_WIDTH-1:0] load_count;
    logic [SEQ_WIDTH-1:0] row_index;
    logic [SEQ_WIDTH-1:0] col_index;
    logic [K_WIDTH-1:0] beat_index;
    logic signed [INPUT_WIDTH-1:0] q_mem [0:MEM_DEPTH-1];
    logic signed [INPUT_WIDTH-1:0] k_mem [0:MEM_DEPTH-1];
    logic signed [ACC_WIDTH-1:0] s_data_reg;

    logic dot_cmd_valid;
    logic dot_cmd_ready;
    logic [K_WIDTH-1:0] dot_cmd_length;
    logic dot_s_valid;
    logic dot_s_ready;
    logic signed [INPUT_WIDTH-1:0] dot_s_a0, dot_s_a1, dot_s_a2, dot_s_a3;
    logic signed [INPUT_WIDTH-1:0] dot_s_b0, dot_s_b1, dot_s_b2, dot_s_b3;
    logic [PARALLELISM-1:0] dot_s_keep;
    logic dot_m_valid;
    logic dot_m_ready;
    logic signed [ACC_WIDTH-1:0] dot_m_result;

    logic [K_WIDTH-1:0] beat_total;
    logic [ADDR_WIDTH-1:0] q_row_base;
    logic [ADDR_WIDTH-1:0] k_row_base;
    logic [ADDR_WIDTH-1:0] q_read_addr;
    logic [ADDR_WIDTH-1:0] k_read_addr;
    logic cmd_accept;
    logic q_accept;
    logic k_accept;
    logic s_accept;

    assign cmd_ready = (state == IDLE) && (cmd_seq != '0) && (cmd_seq <= MAX_SEQ);
    assign cmd_accept = cmd_valid && cmd_ready;
    assign q_ready = (state == LOAD_Q);
    assign q_accept = q_valid && q_ready;
    assign k_ready = (state == LOAD_K);
    assign k_accept = k_valid && k_ready;
    assign s_valid = (state == OUTPUT_S);
    assign s_accept = s_valid && s_ready;
    assign s_data = s_data_reg;
    assign s_row = row_index;
    assign s_col = col_index;

    assign beat_total = K_WIDTH'(HEAD_DIM / PARALLELISM);
    assign q_row_base = ADDR_WIDTH'(row_index) * ADDR_WIDTH'(HEAD_DIM);
    assign k_row_base = ADDR_WIDTH'(col_index) * ADDR_WIDTH'(HEAD_DIM);
    assign q_read_addr = q_row_base + ADDR_WIDTH'(beat_index) * ADDR_WIDTH'(PARALLELISM);
    assign k_read_addr = k_row_base + ADDR_WIDTH'(beat_index) * ADDR_WIDTH'(PARALLELISM);

    assign dot_cmd_valid = (state == DOT_CMD);
    assign dot_cmd_length = K_WIDTH'(HEAD_DIM);
    assign dot_s_valid = (state == DOT_FEED);
    assign dot_m_ready = (state == DOT_WAIT);

    // Score[i][j] = sum_k Q[i][k]*K[j][k] —— K 按行 j 读，即转置视角
    assign dot_s_a0 = q_mem[q_read_addr + ADDR_WIDTH'(0)];
    assign dot_s_a1 = q_mem[q_read_addr + ADDR_WIDTH'(1)];
    assign dot_s_a2 = q_mem[q_read_addr + ADDR_WIDTH'(2)];
    assign dot_s_a3 = q_mem[q_read_addr + ADDR_WIDTH'(3)];
    assign dot_s_b0 = k_mem[k_read_addr + ADDR_WIDTH'(0)];
    assign dot_s_b1 = k_mem[k_read_addr + ADDR_WIDTH'(1)];
    assign dot_s_b2 = k_mem[k_read_addr + ADDR_WIDTH'(2)];
    assign dot_s_b3 = k_mem[k_read_addr + ADDR_WIDTH'(3)];
    assign dot_s_keep = {PARALLELISM{1'b1}};

    int8_dot_product_parallel #(
        .INPUT_WIDTH(INPUT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .MAX_K(HEAD_DIM),
        .LENGTH_WIDTH(K_WIDTH),
        .PARALLELISM(PARALLELISM)
    ) u_dot (
        .clk(clk),
        .rst_n(rst_n),
        .cmd_valid(dot_cmd_valid),
        .cmd_ready(dot_cmd_ready),
        .cmd_length(dot_cmd_length),
        .s_valid(dot_s_valid),
        .s_ready(dot_s_ready),
        .s_a0(dot_s_a0),
        .s_a1(dot_s_a1),
        .s_a2(dot_s_a2),
        .s_a3(dot_s_a3),
        .s_b0(dot_s_b0),
        .s_b1(dot_s_b1),
        .s_b2(dot_s_b2),
        .s_b3(dot_s_b3),
        .s_keep(dot_s_keep),
        .acc_clear(1'b0),
        .acc_enable(1'b0),
        .m_valid(dot_m_valid),
        .m_ready(dot_m_ready),
        .m_result(dot_m_result),
        .partial_valid(),
        .partial_sum(),
        .acc_valid(),
        .acc_value()
    );

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state      <= IDLE;
            seq_reg    <= '0;
            load_total <= '0;
            load_count <= '0;
            row_index  <= '0;
            col_index  <= '0;
            beat_index <= '0;
            s_data_reg <= '0;
        end else begin
            unique case (state)
                IDLE: begin
                    if (cmd_accept) begin
                        seq_reg    <= cmd_seq;
                        load_total <= COUNT_WIDTH'(cmd_seq) * COUNT_WIDTH'(HEAD_DIM);
                        load_count <= '0;
                        row_index  <= '0;
                        col_index  <= '0;
                        beat_index <= '0;
                        state      <= LOAD_Q;
                    end
                end

                LOAD_Q: begin
                    if (q_accept) begin
                        q_mem[load_count] <= q_data;
                        if (load_count == load_total - 1'b1) begin
                            load_count <= '0;
                            state      <= LOAD_K;
                        end else begin
                            load_count <= load_count + 1'b1;
                        end
                    end
                end

                LOAD_K: begin
                    if (k_accept) begin
                        k_mem[load_count] <= k_data;
                        if (load_count == load_total - 1'b1) begin
                            load_count <= '0;
                            row_index  <= '0;
                            col_index  <= '0;
                            beat_index <= '0;
                            state      <= DOT_CMD;
                        end else begin
                            load_count <= load_count + 1'b1;
                        end
                    end
                end

                DOT_CMD: begin
                    if (dot_cmd_ready) begin
                        beat_index <= '0;
                        state      <= DOT_FEED;
                    end
                end

                DOT_FEED: begin
                    if (dot_s_valid && dot_s_ready) begin
                        if (beat_index == beat_total - 1'b1) begin
                            state <= DOT_WAIT;
                        end else begin
                            beat_index <= beat_index + 1'b1;
                        end
                    end
                end

                DOT_WAIT: begin
                    if (dot_m_valid) begin
                        s_data_reg <= dot_m_result;
                        state      <= OUTPUT_S;
                    end
                end

                OUTPUT_S: begin
                    if (s_accept) begin
                        if (col_index == seq_reg - 1'b1) begin
                            if (row_index == seq_reg - 1'b1) begin
                                state <= IDLE;
                            end else begin
                                row_index  <= row_index + 1'b1;
                                col_index  <= '0;
                                beat_index <= '0;
                                state      <= DOT_CMD;
                            end
                        end else begin
                            col_index  <= col_index + 1'b1;
                            beat_index <= '0;
                            state      <= DOT_CMD;
                        end
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end
endmodule
