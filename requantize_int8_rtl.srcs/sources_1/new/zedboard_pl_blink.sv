// ============================================================================
// 文件: zedboard_pl_blink.sv
// 学习阶段: 板级冒烟（非 Transformer；与 INT8 数据通路无关）
// ----------------------------------------------------------------------------
// 【目的】确认 ZedBoard 电源 / JTAG / bitstream / LED 管脚可用
//   器件 xc7z020clg484-1；clk=Y9 100MHz；LED=LD0..LD3
// 【构建/烧录】scripts/build_zedboard_blink.bat · program_zedboard_blink.bat
// 【约束】constrs_1/new/zedboard_pl_blink.xdc
// 下一步工程: PS+AXI 封装 INT8（blink PASS 之后）
// ============================================================================
//
module zedboard_pl_blink (
    input  logic       clk,
    output logic [3:0] led
);
    // 100 MHz / 2^26 ≈ 1.49 Hz 翻转 → 肉眼可见闪烁
    logic [26:0] cnt;

    always_ff @(posedge clk) begin
        cnt <= cnt + 27'd1;
    end

    assign led[0] = cnt[26];
    assign led[1] = cnt[25];
    assign led[2] = cnt[24];
    assign led[3] = cnt[23];
endmodule
