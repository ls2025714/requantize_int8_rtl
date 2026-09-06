// ============================================================================
// 文件: int8_residual_add.sv
// 学习阶段: D11 残差连接（Res1 / Res2）
// ----------------------------------------------------------------------------
// 【在整条链的位置】
//   D12 里两处：Wo 输出 + 输入 X → Res1；FFN down 输出 + Res1 → Res2
//   本模块假设两路 INT8 已对齐同一量化域（由上层保证）
//
// 【算式】z = saturate_sym(x + y) ∈ [-127, 127]
//   中间用 9-bit 有符号和，再饱和回 INT8
//
// 【FSM】IDLE→(WAIT_X→WAIT_Y→EMIT)×cmd_len → IDLE
//   学习注意（D12 踩坑）：COLLECT 时 z_ready 应保持可接收，勿用 !z_valid 门控饿死
// 【验证】tb_int8_residual_add.sv
// ============================================================================
//
module int8_residual_add #(
    parameter int DATA_WIDTH = 8,
    parameter int IDX_WIDTH  = 10
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          cmd_valid,
    output logic                          cmd_ready,
    input  logic [IDX_WIDTH-1:0]          cmd_len,
    input  logic                          x_valid,
    output logic                          x_ready,
    input  logic signed [DATA_WIDTH-1:0]  x_data,
    input  logic                          y_valid,
    output logic                          y_ready,
    input  logic signed [DATA_WIDTH-1:0]  y_data,
    output logic                          z_valid,
    input  logic                          z_ready,
    output logic signed [DATA_WIDTH-1:0]  z_data,
    output logic [IDX_WIDTH-1:0]          z_idx
);
    typedef enum logic [1:0] {IDLE, WAIT_X, WAIT_Y, EMIT} state_t;
    state_t state;
    logic [IDX_WIDTH-1:0] len_reg, idx;
    logic signed [DATA_WIDTH-1:0] x_hold, y_hold, z_hold;
    logic signed [DATA_WIDTH:0]   sum_w;

    // --- 握手 ---
    assign cmd_ready = (state == IDLE);
    assign x_ready   = (state == WAIT_X);
    assign y_ready   = (state == WAIT_Y);
    assign z_data    = z_hold;
    assign z_idx     = idx;

    // --- 主 FSM：收 x/y → 饱和加法 → 发射 ---
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state  <= IDLE;
            len_reg <= '0;
            idx    <= '0;
            x_hold <= '0;
            y_hold <= '0;
            z_hold <= '0;
            z_valid <= 1'b0;
        end else begin
            z_valid <= 1'b0;
            case (state)
                IDLE: begin
                    idx <= '0;
                    if (cmd_valid && cmd_ready) begin
                        len_reg <= cmd_len;
                        state   <= WAIT_X;
                    end
                end
                WAIT_X: begin
                    if (x_valid && x_ready) begin
                        x_hold <= x_data;
                        state  <= WAIT_Y;
                    end
                end
                WAIT_Y: begin
                    if (y_valid && y_ready) begin
                        y_hold <= y_data;
                        sum_w  = $signed({x_hold[DATA_WIDTH-1], x_hold}) +
                                 $signed({y_data[DATA_WIDTH-1], y_data});
                        if (sum_w > DATA_WIDTH'(127))
                            z_hold <= 8'sd127;
                        else if (sum_w < -DATA_WIDTH'(127))
                            z_hold <= -8'sd127;
                        else
                            z_hold <= sum_w[DATA_WIDTH-1:0];
                        state <= EMIT;
                    end
                end
                EMIT: begin
                    z_valid <= 1'b1;
                    if (z_ready) begin
                        if (idx == len_reg - IDX_WIDTH'(1))
                            state <= IDLE;
                        else begin
                            idx   <= idx + 1'b1;
                            state <= WAIT_X;
                        end
                    end
                end
            endcase
        end
    end
endmodule
