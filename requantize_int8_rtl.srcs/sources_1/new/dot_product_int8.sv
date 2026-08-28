// ============================================================================
// 文件: dot_product_int8.sv
// 阶段: 串行点积
// 作用: 向量点积（每拍 1 对 INT8），输出 INT32 累加和
// 验证: tb_dot_product_int8.sv
// ============================================================================

// 协议:
//   1) cmd_valid/cmd_length — 告诉点积长度 K
//   2) RUN 态流式 s_valid + s_a/s_b，每拍一对
//   3) DRAIN 等 MAC 流水线尾部结果进入 acc
//   4) OUTPUT 态 m_valid=1 输出 m_result，等 m_ready 回 IDLE
// 内部实例化 int8_mac_pipeline；cmd_accept 时 pulse clear
module dot_product_int8 #(
    parameter int INPUT_WIDTH  = 8,
    parameter int ACC_WIDTH    = 32,
    parameter int MAX_K        = 256,
    parameter int LENGTH_WIDTH = $clog2(MAX_K + 1)
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          cmd_valid,
    output logic                          cmd_ready,
    input  logic [LENGTH_WIDTH-1:0]       cmd_length,
    input  logic                          s_valid,
    output logic                          s_ready,
    input  logic signed [INPUT_WIDTH-1:0] s_a,
    input  logic signed [INPUT_WIDTH-1:0] s_b,
    output logic                          m_valid,
    input  logic                          m_ready,
    output logic signed [ACC_WIDTH-1:0]   m_result
);
// FSM: IDLE→RUN→DRAIN→OUTPUT

typedef enum logic [1:0] {IDLE, RUN, DRAIN, OUTPUT} state_t;
state_t state;
logic [LENGTH_WIDTH-1:0] length_reg;
logic [LENGTH_WIDTH-1:0] input_count;
logic [LENGTH_WIDTH-1:0] acc_count;
logic signed [ACC_WIDTH-1:0] result_reg;
logic mac_clear;
logic mac_enable;
logic mac_in_valid;
logic mac_out_valid;
logic signed [ACC_WIDTH-1:0] mac_acc_out;
logic cmd_accept;
logic input_accept;
// --- 握手 ---
assign cmd_ready = (state == IDLE) && (cmd_length != '0) && (cmd_length <= MAX_K);
assign cmd_accept = cmd_valid && cmd_ready;
assign s_ready = (state == RUN) && (input_count < length_reg);
assign input_accept = s_valid && s_ready;
assign m_valid = (state == OUTPUT);
assign m_result = result_reg;
assign mac_clear = cmd_accept;
assign mac_enable = (state == RUN) || (state == DRAIN);
assign mac_in_valid = input_accept;
int8_mac_pipeline #(
    .INPUT_WIDTH(INPUT_WIDTH),
    .ACC_WIDTH(ACC_WIDTH)
) u_mac (
    .clk(clk),
    .rst_n(rst_n),
    .clear(mac_clear),
    .enable(mac_enable),
    .in_valid(mac_in_valid),
    .activation(s_a),
    .weight(s_b),
    .out_valid(mac_out_valid),
    .acc_out(mac_acc_out)
);
always_ff @(posedge clk) begin
    if (!rst_n) begin
        state       <= IDLE;
        length_reg  <= '0;
        input_count <= '0;
        acc_count   <= '0;
        result_reg  <= '0;
    end else begin
        // --- 主 FSM ---
        case (state)
            IDLE: begin
                input_count <= '0;
                acc_count   <= '0;
                if (cmd_accept) begin
                    length_reg <= cmd_length;
                    state      <= RUN;
                end
            end
            // 收 K 个输入；与 mac_out_valid 计数对齐
            RUN: begin
                if (input_accept) begin
                    if (input_count == length_reg - 1'b1) begin
                        state <= DRAIN;
                    end else begin
                        input_count <= input_count + 1'b1;
                    end
                end
                if (mac_out_valid) begin
                    if (acc_count == length_reg - 1'b1) begin
                        result_reg <= mac_acc_out;
                        state      <= OUTPUT;
                    end else begin
                        acc_count <= acc_count + 1'b1;
                    end
                end
            end
            // 不收新输入，等 pipeline 尾部
            DRAIN: begin
                if (mac_out_valid) begin
                    if (acc_count == length_reg - 1'b1) begin
                        result_reg <= mac_acc_out;
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
endmodule
