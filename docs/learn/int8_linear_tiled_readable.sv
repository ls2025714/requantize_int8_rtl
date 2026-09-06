// ============================================================================
// 教学可读版｜D6 tiled Linear（对照用，不是工程 DUT）
// ----------------------------------------------------------------------------
// 正式关账 RTL：requantize_int8_rtl.srcs/sources_1/new/int8_linear_tiled.sv
// 本文件目的：把「一个大 always_ff」拆成规范结构，方便对照理解。
// 请勿加入 Vivado Design Sources；勿替换原模块跑回归。
//
// 人话算法（代码就是在实现这个双重循环）：
//   预载 WEIGHT + MULT；收满 A 到 a_buf
//   for n_tile = 0 .. 3:                    // 一组 4 个输出列
//       clear acc_mem
//       for k_tile = 0 .. 3:                // K 分 4 段
//           回放该 tile 权重；跑 GEMM(M,4,16)
//           acc_mem += partial               // INT32 累加
//       requantize 本组 4 列 → INT8 输出
// ============================================================================

module int8_linear_tiled_readable #(
    parameter int INPUT_WIDTH = 8,
    parameter int ACC_WIDTH   = 32,
    parameter int MAX_M       = 4,
    parameter int TILE_N      = 4,
    parameter int TILE_K      = 16,
    parameter int FULL_N      = 16,
    parameter int FULL_K      = 64,
    parameter int NUM_N_TILES = FULL_N / TILE_N,   // 4
    parameter int NUM_K_TILES = FULL_K / TILE_K,   // 4
    // D7 用：D6 学习时可当成 op=0,head=0 → region=0
    parameter int NUM_OPS     = 3,
    parameter int NUM_HEADS   = 4,
    parameter int HEAD_WEIGHT_ELEMS = FULL_N * FULL_K,          // 1024
    parameter int OP_WEIGHT_ELEMS   = NUM_HEADS * HEAD_WEIGHT_ELEMS,
    parameter int WEIGHT_DEPTH = NUM_OPS * OP_WEIGHT_ELEMS,
    parameter int MULT_DEPTH  = NUM_OPS * NUM_HEADS * FULL_N,
    parameter int M_WIDTH     = $clog2(MAX_M + 1),
    parameter int TILE_N_WIDTH = $clog2(TILE_N + 1),
    parameter int TILE_K_WIDTH = $clog2(TILE_K + 1),
    parameter int N_WIDTH     = $clog2(FULL_N + 1),
    parameter int K_WIDTH     = $clog2(FULL_K + 1),
    parameter int OP_WIDTH    = 2,
    parameter int HEAD_WIDTH  = 2,
    parameter int WEIGHT_ADDR_WIDTH = (WEIGHT_DEPTH <= 1) ? 1 : $clog2(WEIGHT_DEPTH),
    parameter int WEIGHT_COUNT_WIDTH = $clog2(WEIGHT_DEPTH + 1),
    parameter int MULT_ADDR_WIDTH = (MULT_DEPTH <= 1) ? 1 : $clog2(MULT_DEPTH),
    parameter int A_BUF_K_WIDTH = $clog2(FULL_K + 1),
    parameter int A_COUNT_WIDTH = $clog2(MAX_M * FULL_K + 1),
    parameter int TILE_C_COUNT_WIDTH = $clog2(MAX_M * TILE_N + 1),
    parameter int OUT_COUNT_WIDTH = $clog2(MAX_M * FULL_N + 1)
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          cmd_valid,
    output logic                          cmd_ready,
    input  logic [M_WIDTH-1:0]            cmd_m,
    input  logic [N_WIDTH-1:0]            cmd_n,
    input  logic [K_WIDTH-1:0]            cmd_k,
    input  logic [OP_WIDTH-1:0]           cmd_op_type,
    input  logic [HEAD_WIDTH-1:0]         cmd_head_idx,
    input  logic                          a_valid,
    output logic                          a_ready,
    input  logic signed [INPUT_WIDTH-1:0] a_data,
    input  logic                          w_valid,
    output logic                          w_ready,
    input  logic signed [INPUT_WIDTH-1:0] w_data,
    input  logic                          mult_valid,
    output logic                          mult_ready,
    input  logic [17:0]                   mult_data,
    output logic                          c_valid,
    input  logic                          c_ready,
    output logic signed [INPUT_WIDTH-1:0] c_data,
    output logic [M_WIDTH-1:0]            c_row,
    output logic [N_WIDTH-1:0]            c_col,
    output logic signed [ACC_WIDTH-1:0]   acc_debug
);

    // =========================================================================
    // 1) 状态定义
    // =========================================================================
    typedef enum logic [3:0] {
        ST_IDLE,        // 可收 cmd；可预载 weight / mult
        ST_LOAD_A,      // 流式收满 M×64 激活 → a_buf
        ST_TILE_START,  // 开一个 M×4×16 子 GEMM（清草稿 / 启回放 / 发 cmd）
        ST_TILE_RUN,    // 喂 A、收 partial、写入 acc_mem
        ST_RQ_HOLD,     // 锁存要量化的 acc + scale
        ST_RQ_FEED,     // 打一拍 requant in_valid
        ST_RQ_WAIT,     // 等 2 拍流水出 INT8
        ST_RQ_PRESENT,  // c_valid 等下游拿走
        ST_DONE         // 回到 IDLE
    } state_t;

    state_t state;

    // =========================================================================
    // 2) 寄存器 / 存储
    // =========================================================================
    logic [M_WIDTH-1:0]  m_reg;
    logic [OP_WIDTH-1:0] op_reg;
    logic [HEAD_WIDTH-1:0] head_reg;

    logic [$clog2(NUM_N_TILES+1)-1:0] n_tile;  // 外循环：第几组输出列
    logic [$clog2(NUM_K_TILES+1)-1:0] k_tile;  // 内循环：第几段 K

    logic signed [INPUT_WIDTH-1:0] a_buf   [0:MAX_M-1][0:FULL_K-1];
    logic signed [ACC_WIDTH-1:0]   acc_mem [0:MAX_M-1][0:TILE_N-1];  // 草稿纸
    logic [17:0]                   mult_mem [0:MULT_DEPTH-1];
    logic [MULT_ADDR_WIDTH-1:0]    mult_wr_ptr;

    logic [A_COUNT_WIDTH-1:0] a_load_count, a_total;
    logic [A_COUNT_WIDTH-1:0] a_feed_idx, a_feed_total;
    logic [TILE_C_COUNT_WIDTH-1:0] tile_c_count, tile_c_total;
    logic [OUT_COUNT_WIDTH-1:0] out_idx;

    logic replay_start;
    logic [WEIGHT_ADDR_WIDTH-1:0] replay_base_addr;
    logic [WEIGHT_COUNT_WIDTH-1:0] replay_length;
    logic gemm_cmd_pending;

    logic signed [ACC_WIDTH-1:0] held_acc;
    logic [M_WIDTH-1:0] held_row;
    logic [N_WIDTH-1:0] held_col;
    logic [17:0] held_mult;
    logic requant_in_valid;

    // =========================================================================
    // 3) 组合：当前 tile 的地址（纯公式，无状态）
    // =========================================================================
    logic [K_WIDTH-1:0] k_base;
    logic [N_WIDTH-1:0] n_base;
    logic [$clog2(NUM_N_TILES*NUM_K_TILES+1)-1:0] tile_idx;
    logic [WEIGHT_ADDR_WIDTH-1:0] region_base;
    logic [MULT_ADDR_WIDTH-1:0]   mult_region_base;

    localparam int TILE_IDX_W =
        (NUM_N_TILES * NUM_K_TILES <= 1) ? 1 : $clog2(NUM_N_TILES * NUM_K_TILES + 1);

    assign k_base  = K_WIDTH'(k_tile) * K_WIDTH'(TILE_K);     // 0,16,32,48
    assign n_base  = N_WIDTH'(n_tile) * N_WIDTH'(TILE_N);     // 0,4,8,12
    assign tile_idx = TILE_IDX_W'(n_tile) * TILE_IDX_W'(NUM_K_TILES)
                    + TILE_IDX_W'(k_tile);

    assign region_base =
        WEIGHT_ADDR_WIDTH'(op_reg)   * WEIGHT_ADDR_WIDTH'(OP_WEIGHT_ELEMS) +
        WEIGHT_ADDR_WIDTH'(head_reg) * WEIGHT_ADDR_WIDTH'(HEAD_WEIGHT_ELEMS);

    assign mult_region_base =
        MULT_ADDR_WIDTH'(op_reg)   * MULT_ADDR_WIDTH'(NUM_HEADS * FULL_N) +
        MULT_ADDR_WIDTH'(head_reg) * MULT_ADDR_WIDTH'(FULL_N);

    // 喂 GEMM 时：把 a_feed_idx 拆成 (row, k_local)
    logic [M_WIDTH-1:0]      a_feed_row;
    logic [TILE_K_WIDTH-1:0] a_feed_klocal;
    logic [N_WIDTH-1:0]      local_col;

    always_comb begin
        a_feed_row    = a_feed_idx / A_COUNT_WIDTH'(TILE_K);
        a_feed_klocal = TILE_K_WIDTH'(a_feed_idx % A_COUNT_WIDTH'(TILE_K));
        local_col     = N_WIDTH'(out_idx % OUT_COUNT_WIDTH'(TILE_N));
    end

    // =========================================================================
    // 4) 顶层握手（谁在什么状态可以收什么）
    // =========================================================================
    assign cmd_ready  = (state == ST_IDLE);
    assign w_ready    = (state == ST_IDLE) && loader_w_ready;
    assign mult_ready = (state == ST_IDLE);
    assign a_ready    = (state == ST_LOAD_A);
    assign acc_debug  = held_acc;

    assign tile_c_total = TILE_C_COUNT_WIDTH'(m_reg) * TILE_C_COUNT_WIDTH'(TILE_N);

    logic tile_complete;
    assign tile_complete =
        gemm_c_valid && gemm_c_ready &&
        (tile_c_count == (tile_c_total - TILE_C_COUNT_WIDTH'(1)));

    // =========================================================================
    // 5) 子模块连线（工人：GEMM / loader / requant）
    // =========================================================================
    logic gemm_cmd_valid, gemm_cmd_ready;
    logic gemm_a_valid, gemm_a_ready;
    logic signed [INPUT_WIDTH-1:0] gemm_a_data;
    logic gemm_b_valid, gemm_b_ready;
    logic signed [INPUT_WIDTH-1:0] gemm_b_data;
    logic gemm_c_valid, gemm_c_ready;
    logic signed [ACC_WIDTH-1:0] gemm_c_data;
    logic [M_WIDTH-1:0] gemm_c_row;
    logic [TILE_N_WIDTH-1:0] gemm_c_col;

    logic loader_w_ready, loader_r_valid, loader_r_ready;
    logic signed [INPUT_WIDTH-1:0] loader_r_data;
    logic requant_out_valid;
    logic signed [INPUT_WIDTH-1:0] requant_out_i;

    assign gemm_cmd_valid = gemm_cmd_pending;
    assign gemm_a_valid   = (state == ST_TILE_RUN) && (a_feed_idx < a_feed_total);
    assign gemm_a_data    = a_buf[a_feed_row][k_base + A_BUF_K_WIDTH'(a_feed_klocal)];
    assign gemm_b_valid   = loader_r_valid;
    assign gemm_b_data    = loader_r_data;
    assign loader_r_ready = gemm_b_ready;
    assign gemm_c_ready   = (state == ST_TILE_RUN);

    int8_gemm_parallel #(
        .INPUT_WIDTH(INPUT_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .MAX_M(MAX_M), .MAX_N(TILE_N), .MAX_K(TILE_K),
        .M_WIDTH(M_WIDTH), .N_WIDTH(TILE_N_WIDTH), .K_WIDTH(TILE_K_WIDTH)
    ) u_gemm (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(gemm_cmd_valid), .cmd_ready(gemm_cmd_ready),
        .cmd_m(m_reg),
        .cmd_n(TILE_N_WIDTH'(TILE_N)),
        .cmd_k(TILE_K_WIDTH'(TILE_K)),
        .a_valid(gemm_a_valid), .a_ready(gemm_a_ready), .a_data(gemm_a_data),
        .b_valid(gemm_b_valid), .b_ready(gemm_b_ready), .b_data(gemm_b_data),
        .c_valid(gemm_c_valid), .c_ready(gemm_c_ready), .c_data(gemm_c_data),
        .c_row(gemm_c_row), .c_col(gemm_c_col)
    );

    int8_weight_loader #(
        .INPUT_WIDTH(INPUT_WIDTH), .MAX_K(TILE_K), .MAX_N(TILE_N),
        .DEPTH(WEIGHT_DEPTH), .ADDR_WIDTH(WEIGHT_ADDR_WIDTH),
        .COUNT_WIDTH(WEIGHT_COUNT_WIDTH)
    ) u_weight_loader (
        .clk(clk), .rst_n(rst_n),
        .w_valid(w_valid), .w_ready(loader_w_ready), .w_data(w_data),
        .replay_start(replay_start),
        .replay_base_addr(replay_base_addr),
        .replay_length(replay_length),
        .r_valid(loader_r_valid), .r_ready(loader_r_ready), .r_data(loader_r_data)
    );

    requantize_int8_pipeline #(.SHIFT_BITS(24)) u_requant (
        .clk(clk), .rst_n(rst_n),
        .in_valid(requant_in_valid),
        .acc_i(held_acc),
        .multiplier_i(held_mult),
        .out_valid(requant_out_valid),
        .out_i(requant_out_i)
    );

    // =========================================================================
    // 6) 旁路存储：各管各的（规范：不要全塞进 FSM case）
    // =========================================================================

    // 6a) IDLE 时预载每列 scale
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            mult_wr_ptr <= '0;
        end else if (mult_valid && mult_ready) begin
            mult_mem[mult_wr_ptr] <= mult_data;
            mult_wr_ptr           <= mult_wr_ptr + 1'b1;
        end else if (state == ST_IDLE && cmd_valid && cmd_ready) begin
            mult_wr_ptr <= '0;  // 新命令：写指针归零（与原版行为对齐）
        end
    end

    // 6b) LOAD_A：把激活写入 a_buf（数据通路）
    always_ff @(posedge clk) begin
        if (state == ST_LOAD_A && a_valid && a_ready) begin
            a_buf[a_load_count / A_COUNT_WIDTH'(FULL_K)]
                 [a_load_count % A_COUNT_WIDTH'(FULL_K)] <= a_data;
        end
    end

    // 6c) TILE_RUN：GEMM 吐出的 INT32 累加进草稿纸
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            // 综合友好：可省略全清；START 里会按需清
        end else if (state == ST_TILE_START && k_tile == '0) begin
            for (int r = 0; r < MAX_M; r++)
                for (int c = 0; c < TILE_N; c++)
                    acc_mem[r][c] <= '0;
        end else if (state == ST_TILE_RUN && gemm_c_valid && gemm_c_ready) begin
            acc_mem[gemm_c_row][gemm_c_col] <=
                acc_mem[gemm_c_row][gemm_c_col] + gemm_c_data;
        end
    end

    // 6d) 喂 A 时推进计数
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            a_feed_idx <= '0;
        end else if (state == ST_TILE_START) begin
            a_feed_idx <= '0;
        end else if (gemm_a_valid && gemm_a_ready) begin
            a_feed_idx <= a_feed_idx + 1'b1;
        end
    end

    // =========================================================================
    // 7) 主 FSM：只写「控制」——状态、tile 下标、发脉冲、输出握手
    //    LOAD_A 的「收满没有」也放在这里（你问过的那种规范写法）
    // =========================================================================
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state            <= ST_IDLE;
            m_reg            <= '0;
            op_reg           <= '0;
            head_reg         <= '0;
            n_tile           <= '0;
            k_tile           <= '0;
            a_load_count     <= '0;
            a_total          <= '0;
            a_feed_total     <= '0;
            tile_c_count     <= '0;
            out_idx          <= '0;
            replay_start     <= 1'b0;
            replay_base_addr <= '0;
            replay_length    <= '0;
            gemm_cmd_pending <= 1'b0;
            requant_in_valid <= 1'b0;
            held_acc         <= '0;
            held_row         <= '0;
            held_col         <= '0;
            held_mult        <= '0;
            c_valid          <= 1'b0;
            c_data           <= '0;
            c_row            <= '0;
            c_col            <= '0;
        end else begin
            // 默认：脉冲信号只亮一拍
            replay_start     <= 1'b0;
            requant_in_valid <= 1'b0;

            unique case (state)

                // ---------------------------------------------------------
                // IDLE：等 cmd（权重/mult 由顶层握手在 IDLE 写入）
                // ---------------------------------------------------------
                ST_IDLE: begin
                    c_valid          <= 1'b0;
                    gemm_cmd_pending <= 1'b0;
                    if (cmd_valid && cmd_ready) begin
                        m_reg        <= cmd_m;
                        op_reg       <= cmd_op_type;
                        head_reg     <= cmd_head_idx;
                        a_total      <= A_COUNT_WIDTH'(cmd_m) * A_COUNT_WIDTH'(FULL_K);
                        a_load_count <= '0;
                        state        <= ST_LOAD_A;
                    end
                end

                // ---------------------------------------------------------
                // LOAD_A：收激活。写 a_buf 在上面 6b；这里管计数和跳转
                // ---------------------------------------------------------
                ST_LOAD_A: begin
                    gemm_cmd_pending <= 1'b0;
                    if (a_valid && a_ready) begin
                        if (a_load_count == a_total - 1'b1) begin
                            a_load_count <= '0;
                            n_tile       <= '0;   // 外循环从头
                            k_tile       <= '0;   // 内循环从头
                            state        <= ST_TILE_START;
                        end else begin
                            a_load_count <= a_load_count + 1'b1;
                        end
                    end
                end

                // ---------------------------------------------------------
                // TILE_START：开「一小盘」= 一个 M×4×16
                // ---------------------------------------------------------
                ST_TILE_START: begin
                    // ① 让 loader 从该 tile 的权重块开始吐
                    replay_start     <= 1'b1;
                    replay_base_addr <= region_base
                        + WEIGHT_ADDR_WIDTH'(tile_idx)
                          * WEIGHT_ADDR_WIDTH'(TILE_K * TILE_N);  // *64
                    replay_length    <= WEIGHT_COUNT_WIDTH'(TILE_K * TILE_N);

                    // ② 给 GEMM 发命令
                    gemm_cmd_pending <= 1'b1;
                    a_feed_total     <= A_COUNT_WIDTH'(m_reg) * A_COUNT_WIDTH'(TILE_K);
                    tile_c_count     <= '0;

                    // ③ 新一组输出列（k_tile==0）时清草稿纸 —— 实际清零在 6c
                    state <= ST_TILE_RUN;
                end

                // ---------------------------------------------------------
                // TILE_RUN：等这一盘算完，再决定「下一顿 k」还是「开始量化」
                // ---------------------------------------------------------
                ST_TILE_RUN: begin
                    if (gemm_cmd_pending && gemm_cmd_ready)
                        gemm_cmd_pending <= 1'b0;

                    if (gemm_c_valid && gemm_c_ready)
                        tile_c_count <= tile_c_count + 1'b1;

                    if (tile_complete) begin
                        if (k_tile != NUM_K_TILES - 1) begin
                            // 内循环：还有下一段 K
                            k_tile <= k_tile + 1'b1;
                            state  <= ST_TILE_START;
                        end else begin
                            // 内循环结束 → 量化本组 4 列
                            out_idx <= '0;
                            state   <= ST_RQ_HOLD;
                        end
                    end
                end

                // ---------------------------------------------------------
                // Requant 四拍（与 D4 相同）：HOLD → FEED → WAIT → PRESENT
                // ---------------------------------------------------------
                ST_RQ_HOLD: begin
                    held_row  <= out_idx / OUT_COUNT_WIDTH'(TILE_N);
                    held_col  <= n_base + local_col;  // 全局列号
                    held_acc  <= acc_mem[out_idx / OUT_COUNT_WIDTH'(TILE_N)][local_col];
                    held_mult <= mult_mem[mult_region_base
                                        + MULT_ADDR_WIDTH'(n_base + local_col)];
                    state <= ST_RQ_FEED;
                end

                ST_RQ_FEED: begin
                    requant_in_valid <= 1'b1;
                    state            <= ST_RQ_WAIT;
                end

                ST_RQ_WAIT: begin
                    if (requant_out_valid) begin
                        c_data  <= requant_out_i;
                        c_row   <= held_row;
                        c_col   <= held_col;
                        c_valid <= 1'b1;
                        state   <= ST_RQ_PRESENT;
                    end
                end

                ST_RQ_PRESENT: begin
                    if (c_valid && c_ready) begin
                        c_valid <= 1'b0;
                        if (out_idx != tile_c_total - 1'b1) begin
                            // 本组还有下一个元素
                            out_idx <= out_idx + 1'b1;
                            state   <= ST_RQ_HOLD;
                        end else if (n_tile != NUM_N_TILES - 1) begin
                            // 外循环：下一组输出列
                            n_tile <= n_tile + 1'b1;
                            k_tile <= '0;
                            state  <= ST_TILE_START;
                        end else begin
                            state <= ST_DONE;
                        end
                    end
                end

                ST_DONE: begin
                    state <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
