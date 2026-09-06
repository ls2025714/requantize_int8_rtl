// ============================================================================
// 文件: int8_gemm_parallel.sv
// 学习阶段: D3 并行 GEMM 控制器
// ----------------------------------------------------------------------------
// 【定位】只做调度：装 A/B → 按 (row,col) 调 4-lane 点积 → 吐 INT32 C
//   不算 Linear/requant；那些是 D4
// 【相对串行】DOT_FEED 每拍 4 对；s_keep 掩 K 非 4 倍数的尾拍；beat_total=ceil(K/4)
// 【地址】A: a[i*K+k+l]；B: b[(k+l)*N+j]；仅在点积输入握手后推进 k_base
// 【FSM】与串行同骨架：IDLE→LOAD_A/B→DOT_CMD→DOT_FEED→DOT_WAIT→OUTPUT_C
// 【验证】tb_int8_gemm_parallel / tb_gemm_parallel_python_vectors（seed=20260827，24/24）
// ============================================================================
//
module int8_gemm_parallel #(
    parameter int INPUT_WIDTH = 8,
    parameter int ACC_WIDTH   = 32,
    parameter int MAX_M       = 4,
    parameter int MAX_N       = 4,
    parameter int MAX_K       = 16,
    parameter int PARALLELISM = 4,
    parameter int M_WIDTH     = $clog2(MAX_M + 1),
    parameter int N_WIDTH     = $clog2(MAX_N + 1),
    parameter int K_WIDTH     = $clog2(MAX_K + 1)
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          cmd_valid,
    output logic                          cmd_ready,
    input  logic [M_WIDTH-1:0]            cmd_m,
    input  logic [N_WIDTH-1:0]            cmd_n,
    input  logic [K_WIDTH-1:0]            cmd_k,
    input  logic                          a_valid,
    output logic                          a_ready,
    input  logic signed [INPUT_WIDTH-1:0] a_data,
    input  logic                          b_valid,
    output logic                          b_ready,
    input  logic signed [INPUT_WIDTH-1:0] b_data,
    output logic                          c_valid,
    input  logic                          c_ready,
    output logic signed [ACC_WIDTH-1:0]   c_data,
    output logic [M_WIDTH-1:0]            c_row,
    output logic [N_WIDTH-1:0]            c_col
);
    localparam int A_DEPTH       = MAX_M * MAX_K;
    localparam int B_DEPTH       = MAX_K * MAX_N;
    localparam int A_COUNT_WIDTH = $clog2(A_DEPTH + 1);
    localparam int B_COUNT_WIDTH = $clog2(B_DEPTH + 1);
    localparam int A_ADDR_WIDTH  = (A_DEPTH <= 1) ? 1 : $clog2(A_DEPTH);
    localparam int B_ADDR_WIDTH  = (B_DEPTH <= 1) ? 1 : $clog2(B_DEPTH);

    // FSM 状态: IDLE→LOAD_A→LOAD_B→DOT_CMD→DOT_FEED→DOT_WAIT→OUTPUT_C

    typedef enum logic [2:0] {IDLE, LOAD_A, LOAD_B, DOT_CMD, DOT_FEED, DOT_WAIT, OUTPUT_C} state_t;

    state_t state;
    logic [M_WIDTH-1:0] m_reg;
    logic [N_WIDTH-1:0] n_reg;
    logic [K_WIDTH-1:0] k_reg;
    logic [A_COUNT_WIDTH-1:0] a_total_reg;
    logic [B_COUNT_WIDTH-1:0] b_total_reg;
    logic [A_COUNT_WIDTH-1:0] a_load_count;
    logic [B_COUNT_WIDTH-1:0] b_load_count;
    logic [M_WIDTH-1:0] row_index;
    logic [N_WIDTH-1:0] col_index;
    logic [K_WIDTH-1:0] beat_index;
    logic signed [INPUT_WIDTH-1:0] a_mem [0:A_DEPTH-1];
    logic signed [INPUT_WIDTH-1:0] b_mem [0:B_DEPTH-1];
    logic signed [ACC_WIDTH-1:0] c_data_reg;
    logic [A_ADDR_WIDTH-1:0] a_row_base;
    logic [A_ADDR_WIDTH-1:0] a_read_addr;
    logic [B_ADDR_WIDTH-1:0] b_read_addr;

    logic dot_cmd_valid;
    logic dot_cmd_ready;
    logic [K_WIDTH-1:0] dot_cmd_length;
    logic dot_s_valid;
    logic dot_s_ready;
    logic signed [INPUT_WIDTH-1:0] dot_s_a0;
    logic signed [INPUT_WIDTH-1:0] dot_s_a1;
    logic signed [INPUT_WIDTH-1:0] dot_s_a2;
    logic signed [INPUT_WIDTH-1:0] dot_s_a3;
    logic signed [INPUT_WIDTH-1:0] dot_s_b0;
    logic signed [INPUT_WIDTH-1:0] dot_s_b1;
    logic signed [INPUT_WIDTH-1:0] dot_s_b2;
    logic signed [INPUT_WIDTH-1:0] dot_s_b3;
    logic [PARALLELISM-1:0] dot_s_keep;
    logic dot_m_valid;
    logic dot_m_ready;
    logic signed [ACC_WIDTH-1:0] dot_m_result;

    logic cmd_accept;
    logic a_accept;
    logic b_accept;
    logic c_accept;
    logic [K_WIDTH-1:0] beat_total;
    logic [B_ADDR_WIDTH-1:0] b_stride;

    // --- 顶层握手 ---
    assign cmd_ready = (state == IDLE) && (cmd_m != '0) && (cmd_n != '0) && (cmd_k != '0) &&
                       (cmd_m <= MAX_M) && (cmd_n <= MAX_N) && (cmd_k <= MAX_K);
    assign cmd_accept = cmd_valid && cmd_ready;
    assign a_ready    = (state == LOAD_A);
    assign a_accept   = a_valid && a_ready;
    assign b_ready    = (state == LOAD_B);
    assign b_accept   = b_valid && b_ready;
    assign c_valid    = (state == OUTPUT_C);
    assign c_accept   = c_valid && c_ready;
    assign c_data     = c_data_reg;
    assign c_row      = row_index;
    assign c_col      = col_index;

    assign beat_total    = (k_reg + K_WIDTH'(PARALLELISM - 1)) >> 2;
    assign b_stride      = B_ADDR_WIDTH'(n_reg);
    assign dot_cmd_valid = (state == DOT_CMD);
    assign dot_cmd_length = k_reg;
    assign dot_s_valid   = (state == DOT_FEED);
    assign dot_m_ready   = (state == DOT_WAIT);

    // --- 从 a_mem/b_mem 组合读 4 lane ---
    assign dot_s_a0 = a_mem[a_read_addr + A_ADDR_WIDTH'(0)];
    assign dot_s_a1 = a_mem[a_read_addr + A_ADDR_WIDTH'(1)];
    assign dot_s_a2 = a_mem[a_read_addr + A_ADDR_WIDTH'(2)];
    assign dot_s_a3 = a_mem[a_read_addr + A_ADDR_WIDTH'(3)];
    assign dot_s_b0 = b_mem[b_read_addr + B_ADDR_WIDTH'(0) * b_stride];
    assign dot_s_b1 = b_mem[b_read_addr + B_ADDR_WIDTH'(1) * b_stride];
    assign dot_s_b2 = b_mem[b_read_addr + B_ADDR_WIDTH'(2) * b_stride];
    assign dot_s_b3 = b_mem[b_read_addr + B_ADDR_WIDTH'(3) * b_stride];

    // 尾 beat: s_keep 掩掉超出 K 的 lane
    always_comb begin
        for (int lane = 0; lane < PARALLELISM; lane++) begin
            dot_s_keep[lane] = (beat_index * K_WIDTH'(PARALLELISM) + K_WIDTH'(lane)) < k_reg;
        end
    end

    // --- 点积核实例 ---
    int8_dot_product_parallel #(
        .INPUT_WIDTH(INPUT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .MAX_K(MAX_K),
        .LENGTH_WIDTH(K_WIDTH),
        .PARALLELISM(PARALLELISM)
    ) u_dot_product (
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
            state        <= IDLE;
            m_reg        <= '0;
            n_reg        <= '0;
            k_reg        <= '0;
            a_total_reg  <= '0;
            b_total_reg  <= '0;
            a_load_count <= '0;
            b_load_count <= '0;
            row_index    <= '0;
            col_index    <= '0;
            beat_index   <= '0;
            a_row_base   <= '0;
            a_read_addr  <= '0;
            b_read_addr  <= '0;
            c_data_reg   <= '0;
        end else begin
            // --- GEMM 主 FSM ---
            case (state)
                // 收 cmd，计算 a_total=M*K, b_total=K*N
                IDLE: begin
                    a_load_count <= '0;
                    b_load_count <= '0;
                    row_index    <= '0;
                    col_index    <= '0;
                    beat_index   <= '0;
                    a_row_base   <= '0;
                    a_read_addr  <= '0;
                    b_read_addr  <= '0;
                    if (cmd_accept) begin
                        m_reg       <= cmd_m;
                        n_reg       <= cmd_n;
                        k_reg       <= cmd_k;
                        a_total_reg <= A_COUNT_WIDTH'(cmd_m) * A_COUNT_WIDTH'(cmd_k);
                        b_total_reg <= B_COUNT_WIDTH'(cmd_k) * B_COUNT_WIDTH'(cmd_n);
                        state       <= LOAD_A;
                    end
                end
                // 流式写 a_mem
                LOAD_A: begin
                    if (a_accept) begin
                        a_mem[a_load_count] <= a_data;
                        if (a_load_count == a_total_reg - 1'b1) begin
                            b_load_count <= '0;
                            state        <= LOAD_B;
                        end else begin
                            a_load_count <= a_load_count + 1'b1;
                        end
                    end
                end
                // 流式写 b_mem
                LOAD_B: begin
                    if (b_accept) begin
                        b_mem[b_load_count] <= b_data;
                        if (b_load_count == b_total_reg - 1'b1) begin
                            row_index   <= '0;
                            col_index   <= '0;
                            beat_index  <= '0;
                            a_row_base  <= '0;
                            a_read_addr <= '0;
                            b_read_addr <= '0;
                            state       <= DOT_CMD;
                        end else begin
                            b_load_count <= b_load_count + 1'b1;
                        end
                    end
                end
                // 等 dot cmd_ready，复位 beat/读地址
                DOT_CMD: begin
                    if (dot_cmd_ready) begin
                        beat_index  <= '0;
                        a_read_addr <= a_row_base;
                        b_read_addr <= B_ADDR_WIDTH'(col_index);
                        state       <= DOT_FEED;
                    end
                end
                // 每 beat 握手 dot_s_ready，地址 +4 / +4N
                DOT_FEED: begin
                    if (dot_s_ready) begin
                        if (beat_index == beat_total - 1'b1) begin
                            state <= DOT_WAIT;
                        end else begin
                            beat_index  <= beat_index + 1'b1;
                            a_read_addr <= a_read_addr + A_ADDR_WIDTH'(PARALLELISM);
                            b_read_addr <= b_read_addr + (b_stride << 2);
                        end
                    end
                end
                // 等 dot_m_valid，锁存 INT32 结果
                DOT_WAIT: begin
                    if (dot_m_valid) begin
                        c_data_reg <= dot_m_result;
                        state      <= OUTPUT_C;
                    end
                end
                // 输出 C[i][j]，按 row/col 推进
                OUTPUT_C: begin
                    if (c_accept) begin
                        if ((row_index == m_reg - 1'b1) && (col_index == n_reg - 1'b1)) begin
                            state <= IDLE;
                        end else if (col_index == n_reg - 1'b1) begin
                            row_index  <= row_index + 1'b1;
                            col_index  <= '0;
                            a_row_base <= a_row_base + A_ADDR_WIDTH'(k_reg);
                            state      <= DOT_CMD;
                        end else begin
                            col_index <= col_index + 1'b1;
                            state     <= DOT_CMD;
                        end
                    end
                end
                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end
endmodule
