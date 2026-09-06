// ============================================================================
// 文件: int8_weight_loader.sv
// 学习阶段: D4 权重 RAM；D6/D7 扩展 DEPTH + replay_base_addr
// ----------------------------------------------------------------------------
// 【干什么】IDLE 时用 w_* 顺序写入 mem；计算时用 replay_* 按段吐给 GEMM 当 B
//   回放中 w_ready=0；仅 r_valid&&r_ready 推进读指针（与 LOAD_B 握手对齐）
// 【D6/D7】replay_base_addr 从任意基址起吐 length 个
//   D4 常 DEPTH=64、base=0；D6 head 块 1024；D7 QKV 全库 12288
// 【验证】经 linear / tiled TB 间接验证
// ============================================================================
//
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
