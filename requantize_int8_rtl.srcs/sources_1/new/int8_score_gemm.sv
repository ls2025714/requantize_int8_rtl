// ============================================================================
// 文件: int8_score_gemm.sv
// 学习阶段: D8 Attention Score（学习计划：Q @ K^T）
// ----------------------------------------------------------------------------
// 【本模块在整条链里的位置】
//   D6/D7 tiled Linear 产出某个 head 的 Q、K、V（各 [M,16] INT8）
//   D8 本模块：单 head 算 Score[seq,seq] = Q @ K^T，输出 INT32（先不量化）
//   之后 D9 才做 scale / causal mask / Softmax
//
// 【数学】
//   Score[i][j] = Σ_t Q[i][t] * K[j][t]     （t = 0 .. HEAD_DIM-1）
//   等价软件写法: Score = Q @ K^T
//   实现技巧: 不拷贝转置矩阵；算 (i,j) 时读 Q 第 i 行、K 第 j 行即可
//
// 【数据流人话】
//   1) 收 cmd_seq → 2) 流式装入 Q、K 到片上 RAM
//   3) 双重循环 for i, for j: 用 4 路点积算完一行×一行
//   4) 每个结果经 s_* 吐出（INT32 + 行列号）
//
// 【FSM】
//   IDLE → LOAD_Q → LOAD_K → (对每个 i,j) DOT_CMD → DOT_FEED → DOT_WAIT → OUTPUT_S
//   OUTPUT_S 后: 下一列 / 下一行 / 或回 IDLE
//
// 【依赖 / 验证】
//   工人: int8_dot_product_parallel（D2 已关账的点积）
//   TB: tb_int8_score_gemm.sv；seed=20260903，8/8 PASS
// ============================================================================

module int8_score_gemm #(
    parameter int INPUT_WIDTH = 8,
    parameter int ACC_WIDTH   = 32,
    parameter int MAX_SEQ     = 4,     // 仿真首版序列上限
    parameter int HEAD_DIM    = 16,    // 单 head 通道数（与 MiniGPT head_dim 一致）
    parameter int PARALLELISM = 4,     // 点积每拍 4 lane；HEAD_DIM 须能整除
    parameter int SEQ_WIDTH   = $clog2(MAX_SEQ + 1),
    parameter int K_WIDTH     = $clog2(HEAD_DIM + 1)
)(
    input  logic                          clk,
    input  logic                          rst_n,
    // --- 命令：只要序列长度；算完一张 Score 矩阵 ---
    input  logic                          cmd_valid,
    output logic                          cmd_ready,
    input  logic [SEQ_WIDTH-1:0]          cmd_seq,
    // --- 流式灌入 Q、K：各 seq×HEAD_DIM 个 INT8，row-major ---
    input  logic                          q_valid,
    output logic                          q_ready,
    input  logic signed [INPUT_WIDTH-1:0] q_data,
    input  logic                          k_valid,
    output logic                          k_ready,
    input  logic signed [INPUT_WIDTH-1:0] k_data,
    // --- Score 输出：INT32；s_row/s_col 标明是 Score[i][j] ---
    output logic                          s_valid,
    input  logic                          s_ready,
    output logic signed [ACC_WIDTH-1:0]   s_data,
    output logic [SEQ_WIDTH-1:0]          s_row,
    output logic [SEQ_WIDTH-1:0]          s_col
);
    localparam int MEM_DEPTH       = MAX_SEQ * HEAD_DIM;
    localparam int COUNT_WIDTH     = $clog2(MEM_DEPTH + 1);
    localparam int ADDR_WIDTH      = (MEM_DEPTH <= 1) ? 1 : $clog2(MEM_DEPTH);

    // -------------------------------------------------------------------------
    // 状态含义（学习对照）
    //   IDLE      等 cmd；可接受非 0 且 ≤MAX_SEQ 的 seq
    //   LOAD_Q    握手写入 q_mem[0 .. seq*16-1]
    //   LOAD_K    同理写入 k_mem；满后从 (row,col)=(0,0) 开始点积
    //   DOT_CMD   向 u_dot 发 length=16 的点积命令
    //   DOT_FEED  每拍喂 4 个 Q×4 个 K；共 beat_total=4 拍
    //   DOT_WAIT  等点积 DRAIN 完成（m_valid）
    //   OUTPUT_S  s_valid=1，等下游取走后再推进 (i,j)
    // -------------------------------------------------------------------------
    typedef enum logic [2:0] {
        IDLE, LOAD_Q, LOAD_K, DOT_CMD, DOT_FEED, DOT_WAIT, OUTPUT_S
    } state_t;

    state_t state;
    logic [SEQ_WIDTH-1:0] seq_reg;       // 锁存的序列长度
    logic [COUNT_WIDTH-1:0] load_total;  // 应收元素个数 = seq * HEAD_DIM
    logic [COUNT_WIDTH-1:0] load_count;  // 当前已写入个数（从 0 计）
    logic [SEQ_WIDTH-1:0] row_index;     // 当前算 Score 的行 i（Q 的行）
    logic [SEQ_WIDTH-1:0] col_index;     // 当前算 Score 的列 j（K 的行）
    logic [K_WIDTH-1:0] beat_index;      // 点积喂数拍号 0..3
    logic signed [INPUT_WIDTH-1:0] q_mem [0:MEM_DEPTH-1];
    logic signed [INPUT_WIDTH-1:0] k_mem [0:MEM_DEPTH-1];
    logic signed [ACC_WIDTH-1:0] s_data_reg;  // 锁存即将输出的 INT32 Score

    // 与点积工人的连线
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

    // --- 顶层握手：谁在什么状态开门 ---
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

    // --- 地址：实现 Q[i]·K[j]（即 K^T 视角）---
    // beat_total = 16/4 = 4：一行点积分 4 拍喂完
    assign beat_total = K_WIDTH'(HEAD_DIM / PARALLELISM);
    // Q 取第 i 行、K 取第 j 行（j = col_index）——转置只体现在“读 K 的哪一行”
    assign q_row_base = ADDR_WIDTH'(row_index) * ADDR_WIDTH'(HEAD_DIM);
    assign k_row_base = ADDR_WIDTH'(col_index) * ADDR_WIDTH'(HEAD_DIM);
    assign q_read_addr = q_row_base + ADDR_WIDTH'(beat_index) * ADDR_WIDTH'(PARALLELISM);
    assign k_read_addr = k_row_base + ADDR_WIDTH'(beat_index) * ADDR_WIDTH'(PARALLELISM);

    assign dot_cmd_valid = (state == DOT_CMD);
    assign dot_cmd_length = K_WIDTH'(HEAD_DIM);
    assign dot_s_valid = (state == DOT_FEED);
    assign dot_m_ready = (state == DOT_WAIT);

    // 当前拍送给点积的 4 个 A（Q）与 4 个 B（K）；HEAD_DIM 整除 4，keep 恒为全 1
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

    // --- 主 FSM：装 Q/K + 按 (i,j) 扫完 Score ---
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
                // 等命令；锁存 seq，准备收 seq*16 个 Q
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

                // 握手一次写入一个 Q；满后转去收 K（与 D6 LOAD_A 同套路）
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

                // 同理收满 K；从 Score[0][0] 开始发点积命令
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

                // 点积模块接住 cmd（length=16）后开始喂数
                DOT_CMD: begin
                    if (dot_cmd_ready) begin
                        beat_index <= '0;
                        state      <= DOT_FEED;
                    end
                end

                // 每拍推进 beat；最后一拍喂完 → 等结果（点积内部还需 DRAIN）
                DOT_FEED: begin
                    if (dot_s_valid && dot_s_ready) begin
                        if (beat_index == beat_total - 1'b1) begin
                            state <= DOT_WAIT;
                        end else begin
                            beat_index <= beat_index + 1'b1;
                        end
                    end
                end

                // 收到 INT32 点积结果，锁存后进入输出握手
                DOT_WAIT: begin
                    if (dot_m_valid) begin
                        s_data_reg <= dot_m_result;
                        state      <= OUTPUT_S;
                    end
                end

                // 下游取走 Score[i][j] 后推进列/行（行优先扫完矩阵）
                //   col 未完 → col++ 再 DOT_CMD
                //   col 完且 row 未完 → row++、col=0 再 DOT_CMD
                //   全部完 → IDLE
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
