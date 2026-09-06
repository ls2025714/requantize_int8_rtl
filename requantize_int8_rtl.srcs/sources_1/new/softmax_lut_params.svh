// ============================================================================
// 文件: softmax_lut_params.svh（自动生成，勿手改）
// 来源: scripts/softmax_fixed_ref.py
// 用途: Softmax 定点参数与 exp LUT；与 int8_softmax_causal.sv 内嵌常量 bit-exact 对照
// ============================================================================
localparam int SOFT_SCALE_MULT = 1;
localparam int SOFT_SCALE_SHIFT = 2;
localparam int SOFT_MASK_VAL = -1073741824;
localparam int SOFT_EXP_MIN = -32;
localparam int SOFT_EXP_MAX = 0;
localparam int SOFT_P_FRAC = 15;
localparam logic [14:0] SOFT_EXP_LUT [0:32] = '{
    15'd0, 15'd0, 15'd0, 15'd0, 15'd0, 15'd0, 15'd0, 15'd0, 15'd0, 15'd0, 15'd0, 15'd0, 15'd0, 15'd0, 15'd0, 15'd0, 15'd0, 15'd0, 15'd0, 15'd0, 15'd0, 15'd1, 15'd1, 15'd4, 15'd11, 15'd30, 15'd81, 15'd221, 15'd600, 15'd1631, 15'd4435, 15'd12054, 15'd32767
};
