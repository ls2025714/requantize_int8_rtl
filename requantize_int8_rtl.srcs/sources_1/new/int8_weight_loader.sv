// ============================================================================
// 文件: int8_weight_loader.sv
// 阶段: D4 权重加载
// 作用: 小 RAM — w 口预存权重，replay 口按序回放给 GEMM LOAD_B
// 验证: tb_linear_int8_python_vectors.sv（间接）
// ============================================================================

// 写口 w_*: 预加载权重到 mem[]，replay 期间 w_ready=0
// 读口 r_*: replay_start 脉冲后，按 replay_length 顺序输出
//   仅在 r_valid && r_ready 时 r_addr++（与 GEMM LOAD_B 握手对齐）
// 回放结束: replay_active=0，w_wr_addr 复位以便下一 case 重写
// D6/D7: replay_base_addr — 从 mem[base+0..length-1] 顺序回放
//   D6 DEPTH=1024（单 head）；D7 DEPTH=12288（Q/K/V × 4 head）
module int8_weight_loader #(
    parameter int INPUT_WIDTH = 8,
    parameter int MAX_K       = 16,
    parameter int MAX_N       = 4,
    parameter int DEPTH       = 64,
    parameter int ADDR_WIDTH  = (DEPTH <= 1) ? 1 : $clog2(DEPTH),
    parameter int COUNT_WIDTH = $clog2(DEPTH + 1)
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          w_valid,
    output logic                          w_ready,
    input  logic signed [INPUT_WIDTH-1:0] w_data,
    input  logic                          replay_start,
    input  logic [ADDR_WIDTH-1:0]         replay_base_addr,
    input  logic [COUNT_WIDTH-1:0]        replay_length,
    output logic                          r_valid,
    input  logic                          r_ready,
    output logic signed [INPUT_WIDTH-1:0] r_data
);
    // mem: 权重存储；w_wr_addr 写指针；r_addr 读指针

    logic signed [INPUT_WIDTH-1:0] mem [0:DEPTH-1];
    logic [ADDR_WIDTH-1:0] w_wr_addr;
    logic [ADDR_WIDTH-1:0] r_addr;
    logic [ADDR_WIDTH-1:0] r_base;
    logic [COUNT_WIDTH-1:0] r_remaining;
    logic replay_active;

    // --- 握手 ---
    assign w_ready = !replay_active;
    assign r_valid = replay_active && (r_remaining != '0);
    assign r_data  = mem[r_base + r_addr];

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            w_wr_addr     <= '0;
            r_addr        <= '0;
            r_base        <= '0;
            r_remaining   <= '0;
            replay_active <= 1'b0;
        end else begin
            // 写口：预加载
            if (w_valid && w_ready) begin
                mem[w_wr_addr] <= w_data;
                if (w_wr_addr == ADDR_WIDTH'(DEPTH - 1)) begin
                    w_wr_addr <= '0;
                end else begin
                    w_wr_addr <= w_wr_addr + 1'b1;
                end
            end

            // 读口：回放 K*N 个权重
            if (replay_start) begin
                replay_active <= 1'b1;
                r_base        <= replay_base_addr;
                r_addr        <= '0;
                r_remaining   <= replay_length;
            end else if (replay_active && r_valid && r_ready) begin
                if (r_remaining == COUNT_WIDTH'(1)) begin
                    replay_active <= 1'b0;
                    r_remaining   <= '0;
                    w_wr_addr     <= '0;
                end else begin
                    r_addr      <= r_addr + 1'b1;
                    r_remaining <= r_remaining - 1'b1;
                end
            end
        end
    end
endmodule
