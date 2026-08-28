// ============================================================================
// 文件: int8_linear_layer.sv
// 阶段: D4 Linear 顶层
// 作用: GEMM(INT32) + weight_loader + requantize → INT8 输出
// 依赖: int8_gemm_parallel, int8_weight_loader, requantize_int8_pipeline
// 验证: tb_linear_int8_python_vectors.sv
// ============================================================================

// D4 顶层数据流:
//   TB → w_* 预存 WEIGHT → weight_loader
//   TB → mult_* 写入 mult_mem[0..N-1]（每输出列一个 18-bit scale）
//   TB → cmd + a_* → u_gemm；LOAD_B 时 loader 回放代替 TB 送 b
//   GEMM 每出一个 INT32 C 元素 → requantize → 对外 INT8 c_valid/c_data
// out_state: OUT_IDLE → OUT_FEED → OUT_REQUANT → OUT_PRESENT
//   OUT_IDLE 期间才接受新 cmd；gemm_c_ready 仅在 OUT_PRESENT 且下游握手
// acc_debug: 波形/debug 看 raw accumulator
//
// 端口:
//   w_*    — 预加载权重（B 矩阵，K×N 个 INT8）
//   mult_* — 预加载每列 requantize scale（N 个 18-bit，在 cmd 前送完）
//   a_*    — LOAD_A 态流式送激活
//   cmd_*  — 矩阵维度 M/N/K（仅 OUT_IDLE 可接受）
//   c_*    — 输出 INT8 结果 + row/col；acc_debug 为对应 INT32 累加值
//
module int8_linear_layer #(
    parameter int INPUT_WIDTH = 8,
    parameter int ACC_WIDTH   = 32,
    parameter int MAX_M       = 4,
    parameter int MAX_N       = 4,
    parameter int MAX_K       = 16,
    parameter int WEIGHT_DEPTH = 64,
    parameter int M_WIDTH     = $clog2(MAX_M + 1),
    parameter int N_WIDTH     = $clog2(MAX_N + 1),
    parameter int K_WIDTH     = $clog2(MAX_K + 1),
    parameter int WEIGHT_COUNT_WIDTH = $clog2(WEIGHT_DEPTH + 1)
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
    // 输出侧 FSM（与 GEMM 内部 FSM 独立）

    typedef enum logic [2:0] {OUT_IDLE, OUT_FEED, OUT_REQUANT, OUT_PRESENT} out_state_t;

    out_state_t out_state;
    logic replay_start;
    logic [WEIGHT_COUNT_WIDTH-1:0] replay_length;
    logic gemm_cmd_valid;
    logic gemm_cmd_ready;
    logic [M_WIDTH-1:0] gemm_cmd_m;
    logic [N_WIDTH-1:0] gemm_cmd_n;
    logic [K_WIDTH-1:0] gemm_cmd_k;
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
    logic [N_WIDTH-1:0] gemm_c_col;
    logic loader_r_valid;
    logic loader_r_ready;
    logic signed [INPUT_WIDTH-1:0] loader_r_data;
    logic requant_in_valid;
    logic requant_out_valid;
    logic signed [INPUT_WIDTH-1:0] requant_out_i;
    logic signed [ACC_WIDTH-1:0] held_acc;
    logic [M_WIDTH-1:0] held_row;
    logic [N_WIDTH-1:0] held_col;
    logic signed [INPUT_WIDTH-1:0] held_out;
    logic [17:0] mult_mem [0:MAX_N-1];
    logic [N_WIDTH-1:0] mult_wr_ptr;
    logic cmd_accept;
    logic requant_fed;

    // --- 顶层握手：仅 OUT_IDLE 可收 cmd ---
    assign cmd_ready      = gemm_cmd_ready && (out_state == OUT_IDLE);
    assign cmd_accept     = cmd_valid && cmd_ready;
    assign gemm_cmd_valid = cmd_valid;
    assign gemm_cmd_m     = cmd_m;
    assign gemm_cmd_n     = cmd_n;
    assign gemm_cmd_k     = cmd_k;
    assign a_ready        = gemm_a_ready;
    assign gemm_a_valid   = a_valid;
    assign gemm_a_data    = a_data;
    assign gemm_b_valid   = loader_r_valid;
    assign gemm_b_data    = loader_r_data;
    assign loader_r_ready = gemm_b_ready;
    assign mult_ready     = gemm_cmd_ready && (out_state == OUT_IDLE);
    assign acc_debug      = held_acc;
    assign gemm_c_ready   = (out_state == OUT_PRESENT) && c_valid && c_ready;

    // --- 子模块: GEMM ---
    int8_gemm_parallel #(
        .INPUT_WIDTH(INPUT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .MAX_M(MAX_M),
        .MAX_N(MAX_N),
        .MAX_K(MAX_K),
        .M_WIDTH(M_WIDTH),
        .N_WIDTH(N_WIDTH),
        .K_WIDTH(K_WIDTH)
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

    // --- 子模块: 重量化 ---
    requantize_int8_pipeline #(
        .SHIFT_BITS(24)
    ) u_requant (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(requant_in_valid),
        .acc_i(held_acc),
        .multiplier_i(mult_mem[held_col]),
        .out_valid(requant_out_valid),
        .out_i(requant_out_i)
    );

    // --- 子模块: 权重 RAM ---
    int8_weight_loader #(
        .INPUT_WIDTH(INPUT_WIDTH),
        .MAX_K(MAX_K),
        .MAX_N(MAX_N),
        .DEPTH(WEIGHT_DEPTH),
        .ADDR_WIDTH((WEIGHT_DEPTH <= 1) ? 1 : $clog2(WEIGHT_DEPTH)),
        .COUNT_WIDTH(WEIGHT_COUNT_WIDTH)
    ) u_weight_loader (
        .clk(clk),
        .rst_n(rst_n),
        .w_valid(w_valid),
        .w_ready(w_ready),
        .w_data(w_data),
        .replay_start(replay_start),
        .replay_base_addr('0),
        .replay_length(replay_length),
        .r_valid(loader_r_valid),
        .r_ready(loader_r_ready),
        .r_data(loader_r_data)
    );

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            replay_start  <= 1'b0;
            replay_length <= '0;
            mult_wr_ptr   <= '0;
            out_state     <= OUT_IDLE;
            c_valid       <= 1'b0;
            c_data        <= '0;
            c_row         <= '0;
            c_col         <= '0;
            requant_in_valid <= 1'b0;
            held_acc      <= '0;
            held_row      <= '0;
            held_col      <= '0;
            held_out      <= '0;
            requant_fed   <= 1'b0;
        end else begin
            replay_start     <= 1'b0;
            requant_in_valid <= 1'b0;

            // 预加载每列 requantize multiplier
            if (mult_valid && mult_ready) begin
                mult_mem[mult_wr_ptr] <= mult_data;
                mult_wr_ptr           <= mult_wr_ptr + 1'b1;
            end

            // cmd 接受时启动 weight replay，长度 K*N
            if (cmd_accept) begin
                replay_start  <= 1'b1;
                replay_length <= WEIGHT_COUNT_WIDTH'(cmd_k) * WEIGHT_COUNT_WIDTH'(cmd_n);
                mult_wr_ptr   <= '0;
                requant_fed   <= 1'b0;
            end

            // --- INT32→INT8 输出流水线 ---
            unique case (out_state)
                // 捕获 GEMM 的 INT32 输出
                OUT_IDLE: begin
                    c_valid <= 1'b0;
                    if (gemm_c_valid && !requant_fed) begin
                        held_acc    <= gemm_c_data;
                        held_row    <= gemm_c_row;
                        held_col    <= gemm_c_col;
                        requant_fed <= 1'b1;
                        out_state   <= OUT_FEED;
                    end
                end
                // 打 requant in_valid
                OUT_FEED: begin
                    requant_in_valid <= 1'b1;
                    out_state        <= OUT_REQUANT;
                end
                // 等 2 拍 pipeline 出 INT8
                OUT_REQUANT: begin
                    if (requant_out_valid) begin
                        held_out <= requant_out_i;
                        c_data   <= requant_out_i;
                        c_row    <= held_row;
                        c_col    <= held_col;
                        c_valid  <= 1'b1;
                        out_state <= OUT_PRESENT;
                    end
                end
                // c_valid=1，握手后释放 gemm_c_ready
                OUT_PRESENT: begin
                    if (c_valid && c_ready) begin
                        c_valid     <= 1'b0;
                        requant_fed <= 1'b0;
                        out_state   <= OUT_IDLE;
                    end
                end
                default: begin
                    out_state <= OUT_IDLE;
                end
            endcase
        end
    end
endmodule
