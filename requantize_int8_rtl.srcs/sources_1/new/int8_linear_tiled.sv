// ============================================================================
// 文件: int8_linear_tiled.sv
// 阶段: D6 Full Tile + D7 Q/K/V 多 Head 基地址
// 作用: 外层 FSM 调度 16 个 M×4×16 子 GEMM，K 维 INT32 累加后再 requantize
//       D7: cmd_op_type / cmd_head_idx → region_base / mult_base
// 依赖: int8_gemm_parallel, int8_weight_loader, requantize_int8_pipeline
// 验证: tb_linear_tiled_head0_q.sv（D6）、tb_linear_tiled_qkv.sv（D7）
// ============================================================================

module int8_linear_tiled #(
    parameter int INPUT_WIDTH = 8,
    parameter int ACC_WIDTH   = 32,
    parameter int MAX_M       = 4,
    parameter int TILE_N      = 4,
    parameter int TILE_K      = 16,
    parameter int FULL_N      = 16,
    parameter int FULL_K      = 64,
    parameter int NUM_N_TILES = FULL_N / TILE_N,
    parameter int NUM_K_TILES = FULL_K / TILE_K,
    parameter int NUM_OPS     = 3,
    parameter int NUM_HEADS   = 4,
    parameter int HEAD_WEIGHT_ELEMS = FULL_N * FULL_K,
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
    parameter int TILE_WEIGHT_COUNT_WIDTH = $clog2(TILE_K * TILE_N + 1),
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
    typedef enum logic [3:0] {
        TILE_IDLE,
        TILE_LOAD_A,
        TILE_START,
        TILE_RUN,
        TILE_REQ_HOLD,
        TILE_REQ_FEED,
        TILE_REQ_WAIT,
        TILE_REQ_PRESENT,
        TILE_DONE
    } tile_state_t;

    tile_state_t tile_state;
    logic [M_WIDTH-1:0] m_reg;
    logic [N_WIDTH-1:0] n_reg;
    logic [K_WIDTH-1:0] k_reg;
    logic [OP_WIDTH-1:0] op_reg;
    logic [HEAD_WIDTH-1:0] head_reg;
    logic [$clog2(NUM_N_TILES+1)-1:0] n_tile;
    logic [$clog2(NUM_K_TILES+1)-1:0] k_tile;
    logic [$clog2(NUM_N_TILES*NUM_K_TILES+1)-1:0] tile_idx;
    logic [K_WIDTH-1:0] k_base;
    logic [N_WIDTH-1:0] n_base;
    logic [A_COUNT_WIDTH-1:0] a_load_count;
    logic [A_COUNT_WIDTH-1:0] a_total;
    logic signed [INPUT_WIDTH-1:0] a_buf [0:MAX_M-1][0:FULL_K-1];
    logic signed [ACC_WIDTH-1:0] acc_mem [0:MAX_M-1][0:TILE_N-1];
    logic [17:0] mult_mem [0:MULT_DEPTH-1];
    logic [MULT_ADDR_WIDTH-1:0] mult_wr_ptr;
    logic [WEIGHT_ADDR_WIDTH-1:0] region_base;
    logic [MULT_ADDR_WIDTH-1:0] mult_region_base;
    logic [N_WIDTH-1:0] local_col;
    logic replay_start;
    logic [WEIGHT_ADDR_WIDTH-1:0] replay_base_addr;
    logic [WEIGHT_COUNT_WIDTH-1:0] replay_length;
    logic gemm_cmd_valid;
    logic gemm_cmd_ready;
    logic [M_WIDTH-1:0] gemm_cmd_m;
    logic [TILE_N_WIDTH-1:0] gemm_cmd_n;
    logic [TILE_K_WIDTH-1:0] gemm_cmd_k;
    logic gemm_a_valid;
    logic gemm_a_ready;
    logic signed [INPUT_WIDTH-1:0] gemm_a_data;
    logic gemm_b_valid;
    logic gemm_b_ready;
    logic signed [INPUT_WIDTH-1:0] gemm_b_data;
    logic gemm_c_valid;
    logic gemm_c_ready;
    logic signed [ACC_WIDTH-1:0] gemm_c_data;
    logic [M_WIDTH-1:0] gemm_c_row;
    logic [TILE_N_WIDTH-1:0] gemm_c_col;
    logic loader_r_valid;
    logic loader_r_ready;
    logic signed [INPUT_WIDTH-1:0] loader_r_data;
    logic requant_in_valid;
    logic requant_out_valid;
    logic signed [INPUT_WIDTH-1:0] requant_out_i;
    logic signed [ACC_WIDTH-1:0] held_acc;
    logic [M_WIDTH-1:0] held_row;
    logic [N_WIDTH-1:0] held_col;
    logic [17:0] held_mult;
    logic signed [INPUT_WIDTH-1:0] held_out;
    logic gemm_cmd_pending;
    logic [A_COUNT_WIDTH-1:0] a_feed_idx;
    logic [A_COUNT_WIDTH-1:0] a_feed_total;
    logic [M_WIDTH-1:0] a_feed_row;
    logic [TILE_K_WIDTH-1:0] a_feed_klocal;
    logic [TILE_C_COUNT_WIDTH-1:0] tile_c_count;
    logic [TILE_C_COUNT_WIDTH-1:0] tile_c_total;
    logic [OUT_COUNT_WIDTH-1:0] out_idx;
    logic [OUT_COUNT_WIDTH-1:0] out_total;
    logic loader_w_ready;
    logic cmd_accept;
    logic tile_complete;

    assign cmd_accept     = cmd_valid && cmd_ready;
    assign cmd_ready      = (tile_state == TILE_IDLE);
    assign w_ready        = (tile_state == TILE_IDLE) && loader_w_ready;
    assign mult_ready     = (tile_state == TILE_IDLE);
    assign a_ready        = (tile_state == TILE_LOAD_A);
    assign gemm_cmd_valid = gemm_cmd_pending;
    assign gemm_cmd_m     = m_reg;
    assign gemm_cmd_n     = TILE_N_WIDTH'(TILE_N);
    assign gemm_cmd_k     = TILE_K_WIDTH'(TILE_K);
    assign gemm_a_valid   = (tile_state == TILE_RUN) && (a_feed_idx < a_feed_total);
    assign gemm_a_data    = a_buf[a_feed_row][k_base + A_BUF_K_WIDTH'(a_feed_klocal)];
    assign gemm_b_valid   = loader_r_valid;
    assign gemm_b_data    = loader_r_data;
    assign loader_r_ready = gemm_b_ready;
    assign gemm_c_ready   = (tile_state == TILE_RUN);
    assign acc_debug      = held_acc;
    assign tile_c_total   = TILE_C_COUNT_WIDTH'(m_reg) * TILE_C_COUNT_WIDTH'(TILE_N);
    assign out_total      = OUT_COUNT_WIDTH'(m_reg) * OUT_COUNT_WIDTH'(FULL_N);
    assign tile_complete  = gemm_c_valid && gemm_c_ready &&
                            (tile_c_count == (tile_c_total - TILE_C_COUNT_WIDTH'(1)));
    localparam int TILE_IDX_W = (NUM_N_TILES * NUM_K_TILES <= 1) ? 1 : $clog2(NUM_N_TILES * NUM_K_TILES + 1);
    assign k_base         = K_WIDTH'(k_tile) * K_WIDTH'(TILE_K);
    assign n_base         = N_WIDTH'(n_tile) * N_WIDTH'(TILE_N);
    assign tile_idx       = TILE_IDX_W'(n_tile) * TILE_IDX_W'(NUM_K_TILES) + TILE_IDX_W'(k_tile);
    assign region_base    = WEIGHT_ADDR_WIDTH'(op_reg) * WEIGHT_ADDR_WIDTH'(OP_WEIGHT_ELEMS)
                          + WEIGHT_ADDR_WIDTH'(head_reg) * WEIGHT_ADDR_WIDTH'(HEAD_WEIGHT_ELEMS);
    assign mult_region_base = MULT_ADDR_WIDTH'(op_reg) * MULT_ADDR_WIDTH'(NUM_HEADS * FULL_N)
                            + MULT_ADDR_WIDTH'(head_reg) * MULT_ADDR_WIDTH'(FULL_N);

    always_comb begin
        a_feed_row    = a_feed_idx / A_COUNT_WIDTH'(TILE_K);
        a_feed_klocal = TILE_K_WIDTH'(a_feed_idx % A_COUNT_WIDTH'(TILE_K));
        local_col     = N_WIDTH'(out_idx % OUT_COUNT_WIDTH'(TILE_N));
    end

    int8_gemm_parallel #(
        .INPUT_WIDTH(INPUT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .MAX_M(MAX_M),
        .MAX_N(TILE_N),
        .MAX_K(TILE_K),
        .M_WIDTH(M_WIDTH),
        .N_WIDTH(TILE_N_WIDTH),
        .K_WIDTH(TILE_K_WIDTH)
    ) u_gemm (
        .clk(clk),
        .rst_n(rst_n),
        .cmd_valid(gemm_cmd_valid),
        .cmd_ready(gemm_cmd_ready),
        .cmd_m(gemm_cmd_m),
        .cmd_n(gemm_cmd_n),
        .cmd_k(gemm_cmd_k),
        .a_valid(gemm_a_valid),
        .a_ready(gemm_a_ready),
        .a_data(gemm_a_data),
        .b_valid(gemm_b_valid),
        .b_ready(gemm_b_ready),
        .b_data(gemm_b_data),
        .c_valid(gemm_c_valid),
        .c_ready(gemm_c_ready),
        .c_data(gemm_c_data),
        .c_row(gemm_c_row),
        .c_col(gemm_c_col)
    );

    requantize_int8_pipeline #(
        .SHIFT_BITS(24)
    ) u_requant (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(requant_in_valid),
        .acc_i(held_acc),
        .multiplier_i(held_mult),
        .out_valid(requant_out_valid),
        .out_i(requant_out_i)
    );

    int8_weight_loader #(
        .INPUT_WIDTH(INPUT_WIDTH),
        .MAX_K(TILE_K),
        .MAX_N(TILE_N),
        .DEPTH(WEIGHT_DEPTH),
        .ADDR_WIDTH(WEIGHT_ADDR_WIDTH),
        .COUNT_WIDTH(WEIGHT_COUNT_WIDTH)
    ) u_weight_loader (
        .clk(clk),
        .rst_n(rst_n),
        .w_valid(w_valid),
        .w_ready(loader_w_ready),
        .w_data(w_data),
        .replay_start(replay_start),
        .replay_base_addr(replay_base_addr),
        .replay_length(replay_length),
        .r_valid(loader_r_valid),
        .r_ready(loader_r_ready),
        .r_data(loader_r_data)
    );

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            tile_state        <= TILE_IDLE;
            m_reg             <= '0;
            n_reg             <= '0;
            k_reg             <= '0;
            op_reg            <= '0;
            head_reg          <= '0;
            n_tile            <= '0;
            k_tile            <= '0;
            a_load_count      <= '0;
            a_total           <= '0;
            mult_wr_ptr       <= '0;
            replay_start      <= 1'b0;
            replay_base_addr  <= '0;
            replay_length     <= '0;
            gemm_cmd_pending  <= 1'b0;
            a_feed_idx        <= '0;
            a_feed_total      <= '0;
            tile_c_count      <= '0;
            out_idx           <= '0;
            c_valid           <= 1'b0;
            c_data            <= '0;
            c_row             <= '0;
            c_col             <= '0;
            requant_in_valid  <= 1'b0;
            held_acc          <= '0;
            held_row          <= '0;
            held_col          <= '0;
            held_mult         <= '0;
            held_out          <= '0;
        end else begin
            replay_start     <= 1'b0;
            requant_in_valid <= 1'b0;

            if (mult_valid && mult_ready) begin
                mult_mem[mult_wr_ptr] <= mult_data;
                mult_wr_ptr           <= mult_wr_ptr + 1'b1;
            end

            if (tile_state == TILE_LOAD_A && a_valid && a_ready) begin
                a_buf[a_load_count / A_COUNT_WIDTH'(FULL_K)]
                     [a_load_count % A_COUNT_WIDTH'(FULL_K)] <= a_data;
                if (a_load_count == a_total - 1'b1) begin
                    a_load_count <= '0;
                    n_tile       <= '0;
                    k_tile       <= '0;
                    tile_state   <= TILE_START;
                end else begin
                    a_load_count <= a_load_count + 1'b1;
                end
            end

            if (gemm_a_valid && gemm_a_ready) begin
                a_feed_idx <= a_feed_idx + 1'b1;
            end

            if (gemm_c_valid && gemm_c_ready) begin
                acc_mem[gemm_c_row][gemm_c_col] <=
                    acc_mem[gemm_c_row][gemm_c_col] + gemm_c_data;
                tile_c_count <= tile_c_count + 1'b1;
            end

            unique case (tile_state)
                TILE_IDLE: begin
                    c_valid          <= 1'b0;
                    gemm_cmd_pending <= 1'b0;
                    if (cmd_accept) begin
                        m_reg        <= cmd_m;
                        n_reg        <= cmd_n;
                        k_reg        <= cmd_k;
                        op_reg       <= cmd_op_type;
                        head_reg     <= cmd_head_idx;
                        a_total      <= A_COUNT_WIDTH'(cmd_m) * A_COUNT_WIDTH'(FULL_K);
                        a_load_count <= '0;
                        mult_wr_ptr  <= '0;
                        tile_state   <= TILE_LOAD_A;
                    end
                end

                TILE_LOAD_A: begin
                    gemm_cmd_pending <= 1'b0;
                end

                TILE_START: begin
                    replay_start     <= 1'b1;
                    replay_base_addr <= region_base
                        + WEIGHT_ADDR_WIDTH'(tile_idx) * WEIGHT_ADDR_WIDTH'(TILE_K * TILE_N);
                    replay_length    <= WEIGHT_COUNT_WIDTH'(TILE_K * TILE_N);
                    gemm_cmd_pending <= 1'b1;
                    a_feed_idx       <= '0;
                    a_feed_total     <= A_COUNT_WIDTH'(m_reg) * A_COUNT_WIDTH'(TILE_K);
                    tile_c_count     <= '0;
                    if (k_tile == '0) begin
                        for (int row = 0; row < MAX_M; row++) begin
                            for (int col = 0; col < TILE_N; col++) begin
                                acc_mem[row][col] <= '0;
                            end
                        end
                    end
                    tile_state <= TILE_RUN;
                end

                TILE_RUN: begin
                    if (gemm_cmd_pending && gemm_cmd_ready) begin
                        gemm_cmd_pending <= 1'b0;
                    end
                    if (tile_complete) begin
                        a_feed_idx <= '0;
                        if (k_tile != NUM_K_TILES - 1) begin
                            k_tile     <= k_tile + 1'b1;
                            tile_state <= TILE_START;
                        end else begin
                            out_idx    <= '0;
                            tile_state <= TILE_REQ_HOLD;
                        end
                    end
                end

                // 先锁存 acc/row/col/mult，下一拍再打 in_valid（与 D4 linear_layer 一致）
                TILE_REQ_HOLD: begin
                    held_row  <= out_idx / OUT_COUNT_WIDTH'(TILE_N);
                    held_col  <= n_base + local_col;
                    held_acc  <= acc_mem[out_idx / OUT_COUNT_WIDTH'(TILE_N)][local_col];
                    held_mult <= mult_mem[mult_region_base + MULT_ADDR_WIDTH'(n_base + local_col)];
                    tile_state <= TILE_REQ_FEED;
                end

                TILE_REQ_FEED: begin
                    requant_in_valid <= 1'b1;
                    tile_state       <= TILE_REQ_WAIT;
                end

                TILE_REQ_WAIT: begin
                    if (requant_out_valid) begin
                        held_out   <= requant_out_i;
                        c_data     <= requant_out_i;
                        c_row      <= held_row;
                        c_col      <= held_col;
                        c_valid    <= 1'b1;
                        tile_state <= TILE_REQ_PRESENT;
                    end
                end

                TILE_REQ_PRESENT: begin
                    if (c_valid && c_ready) begin
                        c_valid <= 1'b0;
                        if (out_idx == tile_c_total - 1'b1) begin
                            if (n_tile != NUM_N_TILES - 1) begin
                                n_tile     <= n_tile + 1'b1;
                                k_tile     <= '0;
                                tile_state <= TILE_START;
                            end else begin
                                tile_state <= TILE_DONE;
                            end
                        end else begin
                            out_idx    <= out_idx + 1'b1;
                            tile_state <= TILE_REQ_HOLD;
                        end
                    end
                end

                TILE_DONE: begin
                    tile_state <= TILE_IDLE;
                end

                default: begin
                    tile_state <= TILE_IDLE;
                end
            endcase
        end
    end
endmodule
