// ============================================================================
// 文件: int8_gemm_serial.sv
// 阶段: 串行 GEMM baseline
// 作用: M×N×K 矩阵乘控制器，内部串行点积，C 元素 INT32 输出
// 验证: tb_int8_gemm_serial.sv
// ============================================================================

// 计算 C[M×N] = A[M×K] × B[K×N]，行优先存 a_mem / b_mem
// FSM: IDLE → LOAD_A → LOAD_B → (对每个 C[i][j]) DOT_CMD → DOT_FEED → DOT_WAIT → OUTPUT_C
// DOT_FEED: 每拍读 a_mem[a_read_addr] 与 b_mem[b_read_addr]
//   A 地址沿行 +1；B 地址沿列 +N（b_read_addr += n_reg）
// 输出: 每个 C 元素 INT32，带 c_row/c_col
module int8_gemm_serial #(
    parameter int INPUT_WIDTH = 8,
    parameter int ACC_WIDTH = 32,
    parameter int MAX_M = 4,
    parameter int MAX_N = 4,
    parameter int MAX_K = 16,
    parameter int M_WIDTH = $clog2(MAX_M + 1),
    parameter int N_WIDTH = $clog2(MAX_N + 1),
    parameter int K_WIDTH = $clog2(MAX_K + 1)
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
localparam int A_DEPTH = MAX_M * MAX_K;
localparam int B_DEPTH = MAX_K * MAX_N;
localparam int A_COUNT_WIDTH = $clog2(A_DEPTH + 1);
localparam int B_COUNT_WIDTH = $clog2(B_DEPTH + 1);
localparam int A_ADDR_WIDTH = (A_DEPTH <= 1) ? 1 : $clog2(A_DEPTH);
localparam int B_ADDR_WIDTH = (B_DEPTH <= 1) ? 1 : $clog2(B_DEPTH);
// FSM 同并行版，DOT_FEED 为每拍 1 对 (a,b)

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
logic [K_WIDTH-1:0] k_index;
logic signed [INPUT_WIDTH-1:0] a_mem [0:A_DEPTH-1];
logic signed [INPUT_WIDTH-1:0] b_mem [0:B_DEPTH-1];
logic signed [ACC_WIDTH-1:0] c_data_reg;
// 地址优化：使用递增地址寄存器，移除DOT_FEED中的运行时乘法
logic [A_ADDR_WIDTH-1:0] a_row_base;
logic [A_ADDR_WIDTH-1:0] a_read_addr;
logic [B_ADDR_WIDTH-1:0] b_read_addr;
logic dot_cmd_valid;
logic dot_cmd_ready;
logic [K_WIDTH-1:0] dot_cmd_length;
logic dot_s_valid;
logic dot_s_ready;
logic signed [INPUT_WIDTH-1:0] dot_s_a;
logic signed [INPUT_WIDTH-1:0] dot_s_b;
logic dot_m_valid;
logic dot_m_ready;
logic signed [ACC_WIDTH-1:0] dot_m_result;
logic cmd_accept;
logic a_accept;
logic b_accept;
logic c_accept;
assign cmd_ready = (state == IDLE) && (cmd_m != '0) && (cmd_n != '0) && (cmd_k != '0) && (cmd_m <= MAX_M) && (cmd_n <= MAX_N) && (cmd_k <= MAX_K);
assign cmd_accept = cmd_valid && cmd_ready;
assign a_ready = (state == LOAD_A);
assign a_accept = a_valid && a_ready;
assign b_ready = (state == LOAD_B);
assign b_accept = b_valid && b_ready;
assign c_valid = (state == OUTPUT_C);
assign c_accept = c_valid && c_ready;
assign c_data = c_data_reg;
assign c_row = row_index;
assign c_col = col_index;
// --- 点积子模块连线 ---
assign dot_cmd_valid = (state == DOT_CMD);
assign dot_cmd_length = k_reg;
assign dot_s_valid = (state == DOT_FEED);
assign dot_s_a = a_mem[a_read_addr];
assign dot_s_b = b_mem[b_read_addr];
assign dot_m_ready = (state == DOT_WAIT);
dot_product_int8 #(
    .INPUT_WIDTH(INPUT_WIDTH),
    .ACC_WIDTH(ACC_WIDTH),
    .MAX_K(MAX_K),
    .LENGTH_WIDTH(K_WIDTH)
) u_dot_product (
    .clk(clk),
    .rst_n(rst_n),
    .cmd_valid(dot_cmd_valid),
    .cmd_ready(dot_cmd_ready),
    .cmd_length(dot_cmd_length),
    .s_valid(dot_s_valid),
    .s_ready(dot_s_ready),
    .s_a(dot_s_a),
    .s_b(dot_s_b),
    .m_valid(dot_m_valid),
    .m_ready(dot_m_ready),
    .m_result(dot_m_result)
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
        k_index      <= '0;
        a_row_base   <= '0;
        a_read_addr  <= '0;
        b_read_addr  <= '0;
        c_data_reg   <= '0;
    end else begin
        case (state)
            IDLE: begin
                a_load_count <= '0;
                b_load_count <= '0;
                row_index    <= '0;
                col_index    <= '0;
                k_index      <= '0;
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
            LOAD_B: begin
                if (b_accept) begin
                    b_mem[b_load_count] <= b_data;
                    if (b_load_count == b_total_reg - 1'b1) begin
                        row_index   <= '0;
                        col_index   <= '0;
                        k_index     <= '0;
                        a_row_base  <= '0;
                        a_read_addr <= '0;
                        b_read_addr <= '0;
                        state       <= DOT_CMD;
                    end else begin
                        b_load_count <= b_load_count + 1'b1;
                    end
                end
            end
            DOT_CMD: begin
                if (dot_cmd_ready) begin
                    k_index     <= '0;
                    // 当前C[i][j]开始：A从当前行首读取，B从当前列首读取
                    a_read_addr <= a_row_base;
                    b_read_addr <= B_ADDR_WIDTH'(col_index);
                    state       <= DOT_FEED;
                end
            end
                // k_index 递增；A+1，B+N
            DOT_FEED: begin
                if (dot_s_ready) begin
                    if (k_index == k_reg - 1'b1) begin
                        state <= DOT_WAIT;
                    end else begin
                        k_index     <= k_index + 1'b1;
                        // 下一项：A沿行连续+1，B沿列跨N步进
                        a_read_addr <= a_read_addr + 1'b1;
                        b_read_addr <= b_read_addr + B_ADDR_WIDTH'(n_reg);
                    end
                end
            end
            DOT_WAIT: begin
                if (dot_m_valid) begin
                    c_data_reg <= dot_m_result;
                    state      <= OUTPUT_C;
                end
            end
            OUTPUT_C: begin
                if (c_accept) begin
                    if ((row_index == m_reg - 1'b1) && (col_index == n_reg - 1'b1)) begin
                        state <= IDLE;
                    end else if (col_index == n_reg - 1'b1) begin
                        row_index  <= row_index + 1'b1;
                        col_index  <= '0;
                        // 换到A的下一行，行首地址在原基础上增加K
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
