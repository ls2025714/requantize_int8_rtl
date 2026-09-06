// ============================================================================
// 文件: int8_transformer_block.sv
// 学习阶段: D12 单层 Transformer Block E2E（学习计划终点）
// ----------------------------------------------------------------------------
// 【定位】把 D6～D11 已关账子模块串成一条流水：预载 → Attention → FFN → 输出
//   规格: n_embd=64, n_head=4, head_dim=16, d_ff=256；E2E 已验 seq=1
//   CLI: scripts/run_transformer_block_xsim.bat；seed=20260908，1/1 EXACT
//
// 【大状态 st 人话顺序】
//   PRE_*     依次预载 QKV / Wo / gate / up / down 的 WEIGHT 与 MULT（必须进 case，勿掉 default）
//   IDLE      等 cmd_seq（预载完成后）
//   LOAD_X    收输入激活 → x_mem
//   Q/K/V     复用 qkv tiled linear，按 head_r=0..3 各算 [seq,16]
//   SCORE     调 int8_score_gemm：Q@K^T → score_mem INT32
//   SOFT      调 int8_softmax_causal → prob_mem UQ1.15
//   ATTN      调 int8_attn_v：P@V → attn_mem
//   CONCAT    4 个 head 拼成 [seq,64] → concat_mem
//   WO        tiled linear 输出投影 → wo_mem
//   RES1      residual(x, wo) → res1_mem
//   GATE/UP   SwiGLU 两路宽线性（64→256）
//   SILU      SiLU(gate)；ELMUL: silu×up；DOWN: 256→64
//   RES2      residual(res1, ffn) → out_mem
//   OUTPUT    流式吐出
//
// 【子状态 ls】复用于 Linear / Score / Soft / Attn / Res / SiLU / Mul：
//   LS_IDLE → LS_CMD（发命令；l_cmd_v 仅在此）→ LS_FEED（灌数据）→ LS_COLLECT（收结果）
//
// 【学习踩坑摘要】（详见 NOTES.md D12）
//   - ST_PRE_* 必须出现在 unique case，否则 default→IDLE 死锁
//   - feed 侧 data+地址与 valid 同拍组合驱动
//   - SiLU 一进一出；Residual 的 z_ready 在 COLLECT 保持高
//   - Golden 用 frozen weight/mult bank，勿重算 scale
// ============================================================================
//
module int8_transformer_block #(
    parameter int INPUT_WIDTH = 8,
    parameter int ACC_WIDTH   = 32,
    parameter int MAX_SEQ     = 4,
    parameter int EMBD        = 64,
    parameter int HEAD_DIM    = 16,
    parameter int NUM_HEADS   = 4,
    parameter int D_FF        = 256,
    parameter int SEQ_WIDTH   = $clog2(MAX_SEQ + 1),
    parameter int IDX_WIDTH   = 10,
    parameter int QKV_FULL_N  = 16,
    parameter int QKV_FULL_K  = 64,
    parameter int QKV_NUM_OPS = 3,
    parameter int QKV_WEIGHT_DEPTH = QKV_NUM_OPS * NUM_HEADS * QKV_FULL_N * QKV_FULL_K,
    parameter int QKV_MULT_DEPTH   = QKV_NUM_OPS * NUM_HEADS * QKV_FULL_N,
    parameter int WO_FULL_N   = 64,
    parameter int WO_FULL_K   = 64,
    parameter int WO_WEIGHT_DEPTH = WO_FULL_N * WO_FULL_K,
    parameter int WO_MULT_DEPTH   = WO_FULL_N,
    parameter int WIDE_FULL_N = 256,
    parameter int WIDE_FULL_K = 64,
    parameter int WIDE_WEIGHT_DEPTH = WIDE_FULL_N * WIDE_FULL_K,
    parameter int WIDE_MULT_DEPTH   = WIDE_FULL_N,
    parameter int DOWN_FULL_N = 64,
    parameter int DOWN_FULL_K = 256,
    parameter int DOWN_WEIGHT_DEPTH = DOWN_FULL_N * DOWN_FULL_K,
    parameter int DOWN_MULT_DEPTH   = DOWN_FULL_N,
    parameter int M_WIDTH     = $clog2(MAX_SEQ + 1)
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          cmd_valid,
    output logic                          cmd_ready,
    input  logic [SEQ_WIDTH-1:0]          cmd_seq,
    input  logic                          w_valid,
    output logic                          w_ready,
    input  logic signed [INPUT_WIDTH-1:0] w_data,
    input  logic                          mult_valid,
    output logic                          mult_ready,
    input  logic [17:0]                   mult_data,
    input  logic                          x_valid,
    output logic                          x_ready,
    input  logic signed [INPUT_WIDTH-1:0] x_data,
    output logic                          out_valid,
    input  logic                          out_ready,
    output logic signed [INPUT_WIDTH-1:0] out_data,
    output logic [IDX_WIDTH-1:0]          out_idx
);
    // ---- 大 FSM 状态（学习对照表）----
    // PRE_*_W/M : 预载对应银行的权重 / scale（成对出现）
    // IDLE/LOAD_X : 等命令 / 收激活
    // Q,K,V : D7 风格按 head 跑 tiled linear
    // SCORE/SOFT/ATTN : D8/D9/D10
    // CONCAT/WO/RES1 : 拼 head → 输出投影 → 残差1
    // GATE/UP/SILU/ELMUL/DOWN/RES2 : SwiGLU + 残差2
    // OUTPUT : 吐最终 INT8
    typedef enum logic [5:0] {
        ST_PRE_QKV_W, ST_PRE_QKV_M,   // QKV 权重银行 + mult
        ST_PRE_WO_W,  ST_PRE_WO_M,    // Wo
        ST_PRE_FG_W,  ST_PRE_FG_M,    // FFN gate（宽）
        ST_PRE_FU_W,  ST_PRE_FU_M,    // FFN up
        ST_PRE_FD_W,  ST_PRE_FD_M,    // FFN down
        ST_IDLE, ST_LOAD_X,
        ST_Q, ST_K, ST_V,
        ST_SCORE, ST_SOFT, ST_ATTN,
        ST_CONCAT, ST_WO,
        ST_RES1, ST_GATE, ST_UP, ST_SILU, ST_ELMUL, ST_DOWN, ST_RES2, ST_OUTPUT
    } blk_state_t;

    // 子阶段：发命令 → 喂数 → 收结果（Linear/Score/Soft/Attn/Res/SiLU/Mul 共用骨架）
    typedef enum logic [1:0] {LS_IDLE, LS_CMD, LS_FEED, LS_COLLECT} ls_t;

    blk_state_t st;
    ls_t ls;
    logic [SEQ_WIDTH-1:0] seq_r;
    logic [1:0] head_r;                 // 当前 head 0..3
    logic [IDX_WIDTH-1:0] idx_r;        // 通用流计数
    logic [IDX_WIDTH-1:0] score_total;
    logic [17:0] pre_cnt;               // 预载还剩多少个
    logic preload_done;

    // ---- 片上中间张量（Buffer 生命周期：写入阶段后只读）----
    logic signed [7:0] x_mem [0:MAX_SEQ-1][0:EMBD-1];
    logic signed [7:0] q_mem [0:NUM_HEADS-1][0:MAX_SEQ-1][0:HEAD_DIM-1];
    logic signed [7:0] k_mem [0:NUM_HEADS-1][0:MAX_SEQ-1][0:HEAD_DIM-1];
    logic signed [7:0] v_mem [0:NUM_HEADS-1][0:MAX_SEQ-1][0:HEAD_DIM-1];
    logic signed [31:0] score_mem [0:NUM_HEADS-1][0:MAX_SEQ-1][0:MAX_SEQ-1];
    logic [15:0] prob_mem [0:NUM_HEADS-1][0:MAX_SEQ-1][0:MAX_SEQ-1];
    logic signed [7:0] attn_mem [0:NUM_HEADS-1][0:MAX_SEQ-1][0:HEAD_DIM-1];
    logic signed [7:0] concat_mem [0:MAX_SEQ-1][0:EMBD-1];
    logic signed [7:0] wo_mem [0:MAX_SEQ-1][0:EMBD-1];
    logic signed [7:0] res1_mem [0:MAX_SEQ-1][0:EMBD-1];
    logic signed [7:0] gate_mem [0:MAX_SEQ-1][0:D_FF-1];
    logic signed [7:0] up_mem [0:MAX_SEQ-1][0:D_FF-1];
    logic signed [7:0] hidden_mem [0:MAX_SEQ-1][0:D_FF-1];
    logic signed [7:0] ffn_mem [0:MAX_SEQ-1][0:EMBD-1];
    logic signed [7:0] out_mem [0:MAX_SEQ-1][0:EMBD-1];

    logic act_qkv, act_wo, act_gate, act_up, act_down;
    logic act_score, act_soft, act_attn, act_res, act_silu, act_emul;

    logic l_cmd_v, l_cmd_r, l_a_v, l_a_r, l_c_v, l_c_r;
    logic [M_WIDTH-1:0] l_cmd_m;
    logic [8:0] l_cmd_n;
    logic [8:0] l_cmd_k;
    logic [1:0] l_cmd_op, l_cmd_head;
    logic signed [7:0] l_a_d, l_c_d;
    logic [M_WIDTH-1:0] l_c_row;
    logic [8:0] l_c_col;

    logic qkv_cmd_r, qkv_a_r, qkv_w_r, qkv_m_r, qkv_c_v;
    logic signed [7:0] qkv_c_d;
    logic [M_WIDTH-1:0] qkv_c_row;
    logic [4:0] qkv_c_col;
    logic wo_cmd_r, wo_a_r, wo_w_r, wo_m_r, wo_c_v;
    logic signed [7:0] wo_c_d;
    logic [M_WIDTH-1:0] wo_c_row;
    logic [6:0] wo_c_col;
    logic gt_cmd_r, gt_a_r, gt_w_r, gt_m_r, gt_c_v;
    logic signed [7:0] gt_c_d;
    logic [M_WIDTH-1:0] gt_c_row;
    logic [8:0] gt_c_col;
    logic gu_cmd_r, gu_a_r, gu_w_r, gu_m_r, gu_c_v;
    logic signed [7:0] gu_c_d;
    logic [M_WIDTH-1:0] gu_c_row;
    logic [8:0] gu_c_col;
    logic dn_cmd_r, dn_a_r, dn_w_r, dn_m_r, dn_c_v;
    logic signed [7:0] dn_c_d;
    logic [M_WIDTH-1:0] dn_c_row;
    logic [6:0] dn_c_col;

    logic sc_cmd_v, sc_cmd_r;
    logic [SEQ_WIDTH-1:0] sc_seq;
    logic sc_q_v, sc_q_r, sc_k_v, sc_k_r, sc_s_v, sc_s_r;
    logic signed [7:0] sc_q_d, sc_k_d;
    logic signed [31:0] sc_s_d;
    logic [SEQ_WIDTH-1:0] sc_s_row, sc_s_col;

    logic sm_cmd_v, sm_cmd_r, sm_in_v, sm_in_r, sm_out_v, sm_out_r;
    logic [SEQ_WIDTH-1:0] sm_seq;
    logic signed [31:0] sm_in_d;
    logic [SEQ_WIDTH-1:0] sm_in_row, sm_in_col;
    logic [15:0] sm_out_d;
    logic [SEQ_WIDTH-1:0] sm_out_row, sm_out_col;

    logic av_cmd_v, av_cmd_r, av_p_v, av_p_r, av_v_v, av_v_r, av_y_v, av_y_r;
    logic [SEQ_WIDTH-1:0] av_seq;
    logic [15:0] av_p_d;
    logic [SEQ_WIDTH-1:0] av_p_row, av_p_col;
    logic signed [7:0] av_v_d, av_y_d;
    logic [SEQ_WIDTH-1:0] av_y_row;
    logic [4:0] av_y_col;

    logic ra_cmd_v, ra_cmd_r;
    logic [IDX_WIDTH-1:0] ra_len;
    logic ra_x_v, ra_x_r, ra_y_v, ra_y_r, ra_z_v, ra_z_r;
    logic signed [7:0] ra_x_d, ra_y_d, ra_z_d;
    logic [IDX_WIDTH-1:0] ra_z_i;

    logic si_in_v, si_in_r, si_out_v, si_out_r;
    logic signed [7:0] si_in_d, si_out_d;
    logic [IDX_WIDTH-1:0] si_in_i, si_out_i;

    logic em_cmd_v, em_cmd_r;
    logic [IDX_WIDTH-1:0] em_len;
    logic em_a_v, em_a_r, em_b_v, em_b_r, em_z_v, em_z_r;
    logic signed [7:0] em_a_d, em_b_d, em_z_d;
    logic [IDX_WIDTH-1:0] em_z_i;

    // --- 预载完成 / 写口 mux ---
    assign preload_done = (st == ST_IDLE);
    assign cmd_ready = preload_done && (ls == LS_IDLE);
    assign w_ready = (st == ST_PRE_QKV_W) ? qkv_w_r :
                     (st == ST_PRE_WO_W)  ? wo_w_r :
                     (st == ST_PRE_FG_W)   ? gt_w_r :
                     (st == ST_PRE_FU_W)   ? gu_w_r :
                     (st == ST_PRE_FD_W)   ? dn_w_r : 1'b0;
    assign mult_ready = (st == ST_PRE_QKV_M) ? qkv_m_r :
                        (st == ST_PRE_WO_M)  ? wo_m_r :
                        (st == ST_PRE_FG_M)  ? gt_m_r :
                        (st == ST_PRE_FU_M)  ? gu_m_r :
                        (st == ST_PRE_FD_M)  ? dn_m_r : 1'b0;

    assign score_total = IDX_WIDTH'(seq_r) * IDX_WIDTH'(HEAD_DIM);

    // --- 各阶段 linear cmd 维度 (N/K/op) ---
    always_comb begin
        l_cmd_m    = seq_r;
        l_cmd_n    = 9'd16;
        l_cmd_k    = 9'd64;
        l_cmd_op   = 2'd0;
        l_cmd_head = head_r;
        unique case (st)
            ST_Q: begin l_cmd_n = 9'd16; l_cmd_k = 9'd64; l_cmd_op = 2'd0; end
            ST_K: begin l_cmd_n = 9'd16; l_cmd_k = 9'd64; l_cmd_op = 2'd1; end
            ST_V: begin l_cmd_n = 9'd16; l_cmd_k = 9'd64; l_cmd_op = 2'd2; end
            ST_WO: begin l_cmd_n = 9'd64; l_cmd_k = 9'd64; end
            ST_GATE, ST_UP: begin l_cmd_n = 9'd256; l_cmd_k = 9'd64; end
            ST_DOWN: begin l_cmd_n = 9'd64; l_cmd_k = 9'd256; end
            default: ;
        endcase
    end

    // --- 阶段使能 / 子模块握手 mux ---
    assign act_qkv   = st inside {ST_Q, ST_K, ST_V};
    assign act_wo    = (st == ST_WO);
    assign act_gate  = (st == ST_GATE);
    assign act_up    = (st == ST_UP);
    assign act_down  = (st == ST_DOWN);
    assign act_score = (st == ST_SCORE);
    assign act_soft  = (st == ST_SOFT);
    assign act_attn  = (st == ST_ATTN);
    assign act_res   = st inside {ST_RES1, ST_RES2};
    assign act_silu  = (st == ST_SILU);
    assign act_emul  = (st == ST_ELMUL);

    assign l_cmd_v = (act_qkv || act_wo || act_gate || act_up || act_down) && (ls == LS_CMD);
    assign l_a_v   = (ls == LS_FEED) && (act_qkv || act_wo || act_gate || act_up || act_down);
    assign l_c_r   = (ls == LS_COLLECT) && (act_qkv || act_wo || act_gate || act_up || act_down);
    assign l_cmd_r = act_qkv ? qkv_cmd_r : act_wo ? wo_cmd_r : act_gate ? gt_cmd_r :
                     act_up ? gu_cmd_r : act_down ? dn_cmd_r : 1'b1;
    assign l_a_r   = act_qkv ? qkv_a_r : act_wo ? wo_a_r : act_gate ? gt_a_r :
                     act_up ? gu_a_r : act_down ? dn_a_r : 1'b0;
    assign l_c_v   = act_qkv ? qkv_c_v : act_wo ? wo_c_v : act_gate ? gt_c_v :
                     act_up ? gu_c_v : act_down ? dn_c_v : 1'b0;
    assign l_c_d   = act_qkv ? qkv_c_d : act_wo ? wo_c_d : act_gate ? gt_c_d :
                     act_up ? gu_c_d : dn_c_d;
    assign l_c_row = act_qkv ? qkv_c_row : act_wo ? wo_c_row : act_gate ? gt_c_row :
                     act_up ? gu_c_row : dn_c_row;
    assign l_c_col = act_qkv ? {4'b0, qkv_c_col} : act_wo ? {2'b0, wo_c_col} :
                     act_gate ? gt_c_col : act_up ? gu_c_col : {2'b0, dn_c_col};

    assign sc_cmd_v = act_score && (ls == LS_CMD);
    assign sc_seq   = seq_r;
    assign sc_q_v   = act_score && (ls == LS_FEED) && (idx_r < score_total);
    assign sc_k_v   = act_score && (ls == LS_FEED) && (idx_r >= score_total);
    assign sc_s_r   = act_score && (ls == LS_COLLECT);

    assign sm_cmd_v = act_soft && (ls == LS_CMD);
    assign sm_seq   = seq_r;
    assign sm_in_v  = act_soft && (ls == LS_FEED);
    assign sm_out_r = act_soft && (ls == LS_COLLECT);

    assign av_cmd_v = act_attn && (ls == LS_CMD);
    assign av_seq   = seq_r;
    assign av_p_v   = act_attn && (ls == LS_FEED) && (idx_r < (IDX_WIDTH'(seq_r) * IDX_WIDTH'(seq_r)));
    assign av_v_v   = act_attn && (ls == LS_FEED) && (idx_r >= (IDX_WIDTH'(seq_r) * IDX_WIDTH'(seq_r)));
    assign av_y_r   = act_attn && (ls == LS_COLLECT);

    assign ra_cmd_v = act_res && (ls == LS_CMD);
    assign ra_len   = IDX_WIDTH'(seq_r) * IDX_WIDTH'(EMBD);
    assign ra_x_v   = act_res && (ls == LS_FEED);
    // Present y until residual emits z; keep z_ready high for the whole COLLECT beat.
    assign ra_y_v   = act_res && (ls == LS_COLLECT) && !ra_z_v;
    assign ra_z_r   = act_res && (ls == LS_COLLECT);

    assign si_in_v  = act_silu && (ls == LS_FEED);
    assign si_out_r = act_silu && (ls == LS_COLLECT);

    assign em_cmd_v = act_emul && (ls == LS_CMD);
    assign em_len   = IDX_WIDTH'(seq_r) * IDX_WIDTH'(D_FF);
    assign em_a_v   = act_emul && (ls == LS_FEED);
    assign em_b_v   = act_emul && (ls == LS_COLLECT) && !em_z_v;
    assign em_z_r   = act_emul && (ls == LS_COLLECT);

    // --- FEED 侧组合数据（须与 *_valid 同拍，禁止 NBA 滞后）---
    always_comb begin
        l_a_d = '0;
        if (ls == LS_FEED) begin
            if (st inside {ST_Q, ST_K, ST_V})
                l_a_d = x_mem[idx_r / EMBD][idx_r % EMBD];
            else if (st == ST_WO)
                l_a_d = concat_mem[idx_r / EMBD][idx_r % EMBD];
            else if (st inside {ST_GATE, ST_UP})
                l_a_d = res1_mem[idx_r / EMBD][idx_r % EMBD];
            else if (st == ST_DOWN)
                l_a_d = hidden_mem[idx_r / D_FF][idx_r % D_FF];
        end
    end

    always_comb begin
        sc_q_d = '0;
        sc_k_d = '0;
        if (act_score && ls == LS_FEED) begin
            if (idx_r < score_total)
                sc_q_d = q_mem[head_r][idx_r/HEAD_DIM][idx_r%HEAD_DIM];
            else if (idx_r < 2*score_total)
                sc_k_d = k_mem[head_r][(idx_r-score_total)/HEAD_DIM][(idx_r-score_total)%HEAD_DIM];
        end
    end

    always_comb begin
        sm_in_d   = '0;
        sm_in_row = '0;
        sm_in_col = '0;
        if (act_soft && ls == LS_FEED && seq_r != '0) begin
            sm_in_row = SEQ_WIDTH'(idx_r / seq_r);
            sm_in_col = SEQ_WIDTH'(idx_r % seq_r);
            sm_in_d   = score_mem[head_r][sm_in_row][sm_in_col];
        end
    end

    always_comb begin
        av_p_d   = '0;
        av_p_row = '0;
        av_p_col = '0;
        av_v_d   = '0;
        if (act_attn && ls == LS_FEED && seq_r != '0) begin
            if (idx_r < IDX_WIDTH'(seq_r)*IDX_WIDTH'(seq_r)) begin
                av_p_row = SEQ_WIDTH'(idx_r / seq_r);
                av_p_col = SEQ_WIDTH'(idx_r % seq_r);
                av_p_d   = prob_mem[head_r][av_p_row][av_p_col];
            end else begin
                av_v_d = v_mem[head_r]
                    [(idx_r-IDX_WIDTH'(seq_r)*IDX_WIDTH'(seq_r))/HEAD_DIM]
                    [(idx_r-IDX_WIDTH'(seq_r)*IDX_WIDTH'(seq_r))%HEAD_DIM];
            end
        end
    end

    always_comb begin
        ra_x_d = '0;
        ra_y_d = '0;
        if (act_res) begin
            if (ls == LS_FEED) begin
                if (st == ST_RES1)
                    ra_x_d = x_mem[idx_r/EMBD][idx_r%EMBD];
                else
                    ra_x_d = res1_mem[idx_r/EMBD][idx_r%EMBD];
            end
            if (ls == LS_COLLECT) begin
                if (st == ST_RES1)
                    ra_y_d = wo_mem[idx_r/EMBD][idx_r%EMBD];
                else
                    ra_y_d = ffn_mem[idx_r/EMBD][idx_r%EMBD];
            end
        end
    end

    always_comb begin
        si_in_d = '0;
        si_in_i = '0;
        if (act_silu && ls == LS_FEED) begin
            si_in_d = gate_mem[idx_r/D_FF][idx_r%D_FF];
            si_in_i = idx_r;
        end
    end

    always_comb begin
        em_a_d = '0;
        em_b_d = '0;
        if (act_emul) begin
            if (ls == LS_FEED)
                em_a_d = gate_mem[idx_r/D_FF][idx_r%D_FF];
            if (ls == LS_COLLECT)
                em_b_d = up_mem[idx_r/D_FF][idx_r%D_FF];
        end
    end

    // --- 子模块实例：QKV / Wo / FFN / Attn / Residual / SiLU / ElemMul ---
    int8_linear_tiled #(.MAX_M(MAX_SEQ), .FULL_N(16), .FULL_K(64), .NUM_OPS(3), .NUM_HEADS(4),
        .WEIGHT_DEPTH(QKV_WEIGHT_DEPTH), .MULT_DEPTH(QKV_MULT_DEPTH)) u_qkv (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(act_qkv && l_cmd_v), .cmd_ready(qkv_cmd_r),
        .cmd_m(l_cmd_m), .cmd_n(l_cmd_n[4:0]), .cmd_k(l_cmd_k[6:0]),
        .cmd_op_type(l_cmd_op), .cmd_head_idx(l_cmd_head),
        .a_valid(l_a_v), .a_ready(qkv_a_r), .a_data(l_a_d),
        .w_valid(st == ST_PRE_QKV_W && w_valid), .w_ready(qkv_w_r), .w_data(w_data),
        .mult_valid(st == ST_PRE_QKV_M && mult_valid), .mult_ready(qkv_m_r), .mult_data(mult_data),
        .c_valid(qkv_c_v), .c_ready(l_c_r), .c_data(qkv_c_d),
        .c_row(qkv_c_row), .c_col(qkv_c_col), .acc_debug());

    int8_linear_tiled #(.MAX_M(MAX_SEQ), .FULL_N(64), .FULL_K(64), .NUM_OPS(1), .NUM_HEADS(1),
        .WEIGHT_DEPTH(WO_WEIGHT_DEPTH), .MULT_DEPTH(WO_MULT_DEPTH)) u_wo (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(act_wo && l_cmd_v), .cmd_ready(wo_cmd_r),
        .cmd_m(l_cmd_m), .cmd_n(l_cmd_n[6:0]), .cmd_k(l_cmd_k[6:0]),
        .cmd_op_type(2'd0), .cmd_head_idx(2'd0),
        .a_valid(l_a_v), .a_ready(wo_a_r), .a_data(l_a_d),
        .w_valid(st == ST_PRE_WO_W && w_valid), .w_ready(wo_w_r), .w_data(w_data),
        .mult_valid(st == ST_PRE_WO_M && mult_valid), .mult_ready(wo_m_r), .mult_data(mult_data),
        .c_valid(wo_c_v), .c_ready(l_c_r), .c_data(wo_c_d),
        .c_row(wo_c_row), .c_col(wo_c_col), .acc_debug());

    int8_linear_tiled #(.MAX_M(MAX_SEQ), .FULL_N(256), .FULL_K(64), .NUM_OPS(1), .NUM_HEADS(1),
        .WEIGHT_DEPTH(WIDE_WEIGHT_DEPTH), .MULT_DEPTH(WIDE_MULT_DEPTH)) u_gate (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(act_gate && l_cmd_v), .cmd_ready(gt_cmd_r),
        .cmd_m(l_cmd_m), .cmd_n(l_cmd_n[8:0]), .cmd_k(l_cmd_k[6:0]),
        .cmd_op_type(2'd0), .cmd_head_idx(2'd0),
        .a_valid(l_a_v), .a_ready(gt_a_r), .a_data(l_a_d),
        .w_valid(st == ST_PRE_FG_W && w_valid), .w_ready(gt_w_r), .w_data(w_data),
        .mult_valid(st == ST_PRE_FG_M && mult_valid), .mult_ready(gt_m_r), .mult_data(mult_data),
        .c_valid(gt_c_v), .c_ready(l_c_r), .c_data(gt_c_d),
        .c_row(gt_c_row), .c_col(gt_c_col), .acc_debug());

    int8_linear_tiled #(.MAX_M(MAX_SEQ), .FULL_N(256), .FULL_K(64), .NUM_OPS(1), .NUM_HEADS(1),
        .WEIGHT_DEPTH(WIDE_WEIGHT_DEPTH), .MULT_DEPTH(WIDE_MULT_DEPTH)) u_up (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(act_up && l_cmd_v), .cmd_ready(gu_cmd_r),
        .cmd_m(l_cmd_m), .cmd_n(l_cmd_n[8:0]), .cmd_k(l_cmd_k[6:0]),
        .cmd_op_type(2'd0), .cmd_head_idx(2'd0),
        .a_valid(l_a_v), .a_ready(gu_a_r), .a_data(l_a_d),
        .w_valid(st == ST_PRE_FU_W && w_valid), .w_ready(gu_w_r), .w_data(w_data),
        .mult_valid(st == ST_PRE_FU_M && mult_valid), .mult_ready(gu_m_r), .mult_data(mult_data),
        .c_valid(gu_c_v), .c_ready(l_c_r), .c_data(gu_c_d),
        .c_row(gu_c_row), .c_col(gu_c_col), .acc_debug());

    int8_linear_tiled #(.MAX_M(MAX_SEQ), .FULL_N(64), .FULL_K(256), .NUM_OPS(1), .NUM_HEADS(1),
        .WEIGHT_DEPTH(DOWN_WEIGHT_DEPTH), .MULT_DEPTH(DOWN_MULT_DEPTH)) u_down (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(act_down && l_cmd_v), .cmd_ready(dn_cmd_r),
        .cmd_m(l_cmd_m), .cmd_n(l_cmd_n[6:0]), .cmd_k(l_cmd_k[8:0]),
        .cmd_op_type(2'd0), .cmd_head_idx(2'd0),
        .a_valid(l_a_v), .a_ready(dn_a_r), .a_data(l_a_d),
        .w_valid(st == ST_PRE_FD_W && w_valid), .w_ready(dn_w_r), .w_data(w_data),
        .mult_valid(st == ST_PRE_FD_M && mult_valid), .mult_ready(dn_m_r), .mult_data(mult_data),
        .c_valid(dn_c_v), .c_ready(l_c_r), .c_data(dn_c_d),
        .c_row(dn_c_row), .c_col(dn_c_col), .acc_debug());

    int8_score_gemm #(.MAX_SEQ(MAX_SEQ), .HEAD_DIM(HEAD_DIM)) u_score (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(sc_cmd_v), .cmd_ready(sc_cmd_r), .cmd_seq(sc_seq),
        .q_valid(sc_q_v), .q_ready(sc_q_r), .q_data(sc_q_d),
        .k_valid(sc_k_v), .k_ready(sc_k_r), .k_data(sc_k_d),
        .s_valid(sc_s_v), .s_ready(sc_s_r), .s_data(sc_s_d),
        .s_row(sc_s_row), .s_col(sc_s_col));

    int8_softmax_causal #(.MAX_SEQ(MAX_SEQ)) u_soft (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(sm_cmd_v), .cmd_ready(sm_cmd_r), .cmd_seq(sm_seq),
        .in_valid(sm_in_v), .in_ready(sm_in_r), .in_data(sm_in_d),
        .in_row(sm_in_row), .in_col(sm_in_col),
        .out_valid(sm_out_v), .out_ready(sm_out_r), .out_data(sm_out_d),
        .out_row(sm_out_row), .out_col(sm_out_col));

    int8_attn_v #(.MAX_SEQ(MAX_SEQ), .HEAD_DIM(HEAD_DIM)) u_attn (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(av_cmd_v), .cmd_ready(av_cmd_r), .cmd_seq(av_seq),
        .p_valid(av_p_v), .p_ready(av_p_r), .p_data(av_p_d),
        .p_row(av_p_row), .p_col(av_p_col),
        .v_valid(av_v_v), .v_ready(av_v_r), .v_data(av_v_d),
        .y_valid(av_y_v), .y_ready(av_y_r), .y_data(av_y_d),
        .y_row(av_y_row), .y_col(av_y_col));

    int8_residual_add #(.IDX_WIDTH(IDX_WIDTH)) u_res (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(ra_cmd_v), .cmd_ready(ra_cmd_r), .cmd_len(ra_len),
        .x_valid(ra_x_v), .x_ready(ra_x_r), .x_data(ra_x_d),
        .y_valid(ra_y_v), .y_ready(ra_y_r), .y_data(ra_y_d),
        .z_valid(ra_z_v), .z_ready(ra_z_r), .z_data(ra_z_d), .z_idx(ra_z_i));

    int8_silu_lut #(.IDX_WIDTH(IDX_WIDTH)) u_silu (
        .clk(clk), .rst_n(rst_n),
        .in_valid(si_in_v), .in_ready(si_in_r), .in_data(si_in_d), .in_idx(si_in_i),
        .out_valid(si_out_v), .out_ready(si_out_r), .out_data(si_out_d), .out_idx(si_out_i));

    int8_elem_mul #(.IDX_WIDTH(IDX_WIDTH)) u_emul (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(em_cmd_v), .cmd_ready(em_cmd_r), .cmd_len(em_len),
        .a_valid(em_a_v), .a_ready(em_a_r), .a_data(em_a_d),
        .b_valid(em_b_v), .b_ready(em_b_r), .b_data(em_b_d),
        .z_valid(em_z_v), .z_ready(em_z_r), .z_data(em_z_d), .z_idx(em_z_i));

    function automatic logic [IDX_WIDTH-1:0] f_embd_len(input logic [SEQ_WIDTH-1:0] s);
        return IDX_WIDTH'(s) * IDX_WIDTH'(EMBD);
    endfunction
    function automatic logic [IDX_WIDTH-1:0] f_dff_len(input logic [SEQ_WIDTH-1:0] s);
        return IDX_WIDTH'(s) * IDX_WIDTH'(D_FF);
    endfunction

    // --- 主 FSM：预载计数 + 各计算阶段 ---
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= ST_PRE_QKV_W; ls <= LS_IDLE; seq_r <= '0; head_r <= '0;
            idx_r <= '0; pre_cnt <= QKV_WEIGHT_DEPTH[17:0];
            x_ready <= 1'b0; out_valid <= 1'b0; out_data <= '0; out_idx <= '0;
        end else begin
            x_ready <= 1'b0; out_valid <= 1'b0;

            // --- 预载：依次灌 QKV/Wo/gate/up/down 的 weight 与 mult ---
            if (st == ST_PRE_QKV_W && w_valid && w_ready) begin
                if (pre_cnt <= 18'd1) begin st <= ST_PRE_QKV_M; pre_cnt <= QKV_MULT_DEPTH[17:0]; end
                else pre_cnt <= pre_cnt - 18'd1;
            end else if (st == ST_PRE_QKV_M && mult_valid && mult_ready) begin
                if (pre_cnt <= 18'd1) begin st <= ST_PRE_WO_W; pre_cnt <= WO_WEIGHT_DEPTH[17:0]; end
                else pre_cnt <= pre_cnt - 18'd1;
            end else if (st == ST_PRE_WO_W && w_valid && w_ready) begin
                if (pre_cnt <= 18'd1) begin st <= ST_PRE_WO_M; pre_cnt <= WO_MULT_DEPTH[17:0]; end
                else pre_cnt <= pre_cnt - 18'd1;
            end else if (st == ST_PRE_WO_M && mult_valid && mult_ready) begin
                if (pre_cnt <= 18'd1) begin st <= ST_PRE_FG_W; pre_cnt <= WIDE_WEIGHT_DEPTH[17:0]; end
                else pre_cnt <= pre_cnt - 18'd1;
            end else if (st == ST_PRE_FG_W && w_valid && w_ready) begin
                if (pre_cnt <= 18'd1) begin st <= ST_PRE_FG_M; pre_cnt <= WIDE_MULT_DEPTH[17:0]; end
                else pre_cnt <= pre_cnt - 18'd1;
            end else if (st == ST_PRE_FG_M && mult_valid && mult_ready) begin
                if (pre_cnt <= 18'd1) begin st <= ST_PRE_FU_W; pre_cnt <= WIDE_WEIGHT_DEPTH[17:0]; end
                else pre_cnt <= pre_cnt - 18'd1;
            end else if (st == ST_PRE_FU_W && w_valid && w_ready) begin
                if (pre_cnt <= 18'd1) begin st <= ST_PRE_FU_M; pre_cnt <= WIDE_MULT_DEPTH[17:0]; end
                else pre_cnt <= pre_cnt - 18'd1;
            end else if (st == ST_PRE_FU_M && mult_valid && mult_ready) begin
                if (pre_cnt <= 18'd1) begin st <= ST_PRE_FD_W; pre_cnt <= DOWN_WEIGHT_DEPTH[17:0]; end
                else pre_cnt <= pre_cnt - 18'd1;
            end else if (st == ST_PRE_FD_W && w_valid && w_ready) begin
                if (pre_cnt <= 18'd1) begin st <= ST_PRE_FD_M; pre_cnt <= DOWN_MULT_DEPTH[17:0]; end
                else pre_cnt <= pre_cnt - 18'd1;
            end else if (st == ST_PRE_FD_M && mult_valid && mult_ready) begin
                if (pre_cnt <= 18'd1) begin st <= ST_IDLE; pre_cnt <= 18'd0; end
                else pre_cnt <= pre_cnt - 18'd1;
            end

            unique case (st)
                // 预载完成后：收 cmd，开始装 X
                ST_IDLE: if (cmd_valid && cmd_ready) begin
                    seq_r <= cmd_seq; head_r <= '0; idx_r <= '0;
                    ls <= LS_IDLE; st <= ST_LOAD_X;
                end

                // 流式写入 x_mem；满后进入 ST_Q（head0）
                ST_LOAD_X: begin
                    x_ready <= 1'b1;
                    if (x_valid && x_ready) begin
                        x_mem[idx_r / EMBD][idx_r % EMBD] <= x_data;
                        if (idx_r == f_embd_len(seq_r) - 1) begin idx_r <= '0; ls <= LS_CMD; st <= ST_Q; end
                        else idx_r <= idx_r + 1'b1;
                    end
                end

                // Linear 族共用：CMD→FEED→COLLECT；Q/K/V 还要扫 head_r
                ST_Q, ST_K, ST_V, ST_WO, ST_GATE, ST_UP, ST_DOWN: begin
                    unique case (ls)
                        LS_IDLE, LS_CMD: if (l_cmd_v && l_cmd_r) begin ls <= LS_FEED; idx_r <= '0; end
                        LS_FEED: begin
                            if (l_a_v && l_a_r) begin
                                if (idx_r == ((st == ST_DOWN) ? f_dff_len(seq_r) : f_embd_len(seq_r)) - 1) begin
                                    idx_r <= '0; ls <= LS_COLLECT;
                                end else idx_r <= idx_r + 1'b1;
                            end
                        end
                        LS_COLLECT: if (l_c_v && l_c_r) begin
                            if (st == ST_Q) q_mem[head_r][l_c_row][l_c_col[3:0]] <= l_c_d;
                            else if (st == ST_K) k_mem[head_r][l_c_row][l_c_col[3:0]] <= l_c_d;
                            else if (st == ST_V) v_mem[head_r][l_c_row][l_c_col[3:0]] <= l_c_d;
                            else if (st == ST_WO) wo_mem[l_c_row][l_c_col[5:0]] <= l_c_d;
                            else if (st == ST_GATE) gate_mem[l_c_row][l_c_col[7:0]] <= l_c_d;
                            else if (st == ST_UP) up_mem[l_c_row][l_c_col[7:0]] <= l_c_d;
                            else ffn_mem[l_c_row][l_c_col[5:0]] <= l_c_d;
                            if (idx_r == ((st inside {ST_Q,ST_K,ST_V}) ? (IDX_WIDTH'(seq_r)*16-1) :
                                          (st inside {ST_GATE,ST_UP}) ? f_dff_len(seq_r)-1 : f_embd_len(seq_r)-1)) begin
                                ls <= LS_CMD;
                                if (st == ST_Q && head_r != 3) head_r <= head_r + 1'b1;
                                else if (st == ST_Q) begin head_r <= '0; st <= ST_K; end
                                else if (st == ST_K && head_r != 3) head_r <= head_r + 1'b1;
                                else if (st == ST_K) begin head_r <= '0; st <= ST_V; end
                                else if (st == ST_V && head_r != 3) head_r <= head_r + 1'b1;
                                else if (st == ST_V) begin head_r <= '0; st <= ST_SCORE; end
                                else if (st == ST_WO) begin st <= ST_RES1; ls <= LS_IDLE; end
                                else if (st == ST_GATE) st <= ST_UP;
                                else if (st == ST_UP) st <= ST_SILU;
                                else begin st <= ST_RES2; idx_r <= '0; ls <= LS_IDLE; end
                            end else idx_r <= idx_r + 1'b1;
                        end
                        default: ls <= LS_IDLE;
                    endcase
                end

                // D8: 灌 Q 再灌 K，收集 Score INT32
                ST_SCORE: begin
                    unique case (ls)
                        LS_IDLE, LS_CMD: if (sc_cmd_v && sc_cmd_r) begin ls <= LS_FEED; idx_r <= '0; end
                        LS_FEED: begin
                            if (idx_r < score_total) begin
                                if (sc_q_v && sc_q_r) idx_r <= idx_r + 1'b1;
                            end else if (idx_r < 2*score_total) begin
                                if (sc_k_v && sc_k_r) idx_r <= idx_r + 1'b1;
                            end else begin
                                idx_r <= '0;
                                ls    <= LS_COLLECT;
                            end
                        end
                        LS_COLLECT: if (sc_s_v && sc_s_r) begin
                            score_mem[head_r][sc_s_row][sc_s_col] <= sc_s_d;
                            if (idx_r == IDX_WIDTH'(seq_r)*IDX_WIDTH'(seq_r)-1) begin
                                idx_r <= '0; ls <= LS_CMD; st <= ST_SOFT;
                            end else idx_r <= idx_r + 1'b1;
                        end
                        default: ls <= LS_IDLE;
                    endcase
                end

                // D9 Softmax：灌 Score，收集概率 → ST_ATTN
                ST_SOFT: unique case (ls)
                    LS_IDLE, LS_CMD: if (sm_cmd_v && sm_cmd_r) begin ls <= LS_FEED; idx_r <= '0; end
                    LS_FEED: begin
                        if (sm_in_v && sm_in_r) begin
                            if (idx_r == IDX_WIDTH'(seq_r)*IDX_WIDTH'(seq_r)-1) begin idx_r <= '0; ls <= LS_COLLECT; end
                            else idx_r <= idx_r + 1'b1;
                        end
                    end
                    LS_COLLECT: if (sm_out_v && sm_out_r) begin
                        prob_mem[head_r][sm_out_row][sm_out_col] <= sm_out_d;
                        if (idx_r == IDX_WIDTH'(seq_r)*IDX_WIDTH'(seq_r)-1) begin idx_r <= '0; ls <= LS_CMD; st <= ST_ATTN; end
                        else idx_r <= idx_r + 1'b1;
                    end
                    default: ls <= LS_IDLE;
                endcase

                // D10 Attn@V：先灌 P 再灌 V；本 head 完后若还有 head → 回 ST_SCORE
                ST_ATTN: unique case (ls)
                    LS_IDLE, LS_CMD: if (av_cmd_v && av_cmd_r) begin ls <= LS_FEED; idx_r <= '0; end
                    LS_FEED: begin
                        if (idx_r < IDX_WIDTH'(seq_r)*IDX_WIDTH'(seq_r)) begin
                            if (av_p_v && av_p_r) idx_r <= idx_r + 1'b1;
                        end else begin
                            if (av_v_v && av_v_r) begin
                                if (idx_r == IDX_WIDTH'(seq_r)*IDX_WIDTH'(seq_r) + IDX_WIDTH'(seq_r)*HEAD_DIM - 1) begin
                                    idx_r <= '0;
                                    ls    <= LS_COLLECT;
                                end else idx_r <= idx_r + 1'b1;
                            end
                        end
                    end
                    LS_COLLECT: if (av_y_v && av_y_r) begin
                        attn_mem[head_r][av_y_row][av_y_col] <= av_y_d;
                        if (idx_r == IDX_WIDTH'(seq_r)*HEAD_DIM-1) begin
                            if (head_r == 3) begin head_r <= '0; st <= ST_CONCAT; ls <= LS_IDLE; end
                            else begin head_r <= head_r + 1'b1; st <= ST_SCORE; ls <= LS_CMD; idx_r <= '0; end
                        end else idx_r <= idx_r + 1'b1;
                    end
                    default: ls <= LS_IDLE;
                endcase

                // 4 head × 16 → 拼成 [seq,64]，然后 ST_WO
                ST_CONCAT: begin
                    for (int t = 0; t < MAX_SEQ; t++)
                        for (int h = 0; h < NUM_HEADS; h++)
                            for (int d = 0; d < HEAD_DIM; d++)
                                concat_mem[t][h*HEAD_DIM+d] <= attn_mem[h][t][d];
                    ls <= LS_CMD; st <= ST_WO;
                end

                // D11 残差：RES1 后进 GATE；RES2 后进 OUTPUT
                ST_RES1, ST_RES2: unique case (ls)
                    LS_IDLE: begin
                        ls    <= LS_CMD;
                        idx_r <= '0;
                    end
                    LS_CMD: begin
                        if (ra_cmd_v && ra_cmd_r) begin
                            ls    <= LS_FEED;
                            idx_r <= '0;
                        end
                    end
                    LS_FEED: begin
                        if (ra_x_v && ra_x_r) ls <= LS_COLLECT;
                    end
                    LS_COLLECT: begin
                        if (ra_z_v && ra_z_r) begin
                            if (st == ST_RES1)
                                res1_mem[idx_r/EMBD][idx_r%EMBD] <= ra_z_d;
                            else
                                out_mem[idx_r/EMBD][idx_r%EMBD] <= ra_z_d;
                            if (idx_r == f_embd_len(seq_r)-1) begin
                                idx_r <= '0;
                                if (st == ST_RES1) begin ls <= LS_CMD; st <= ST_GATE; end
                                else begin ls <= LS_IDLE; st <= ST_OUTPUT; end
                            end else begin
                                idx_r <= idx_r + 1'b1;
                                ls    <= LS_FEED;
                            end
                        end
                    end
                    default: ls <= LS_IDLE;
                endcase

                // SiLU(gate)，结果写回 gate_mem → ST_ELMUL
                ST_SILU: unique case (ls)
                    LS_IDLE, LS_CMD: begin ls <= LS_FEED; idx_r <= '0; end
                    LS_FEED: begin
                        if (si_in_v && si_in_r) ls <= LS_COLLECT;
                    end
                    LS_COLLECT: if (si_out_v && si_out_r) begin
                        gate_mem[si_out_i/D_FF][si_out_i%D_FF] <= si_out_d;
                        if (idx_r == f_dff_len(seq_r)-1) begin
                            ls <= LS_CMD;
                            st <= ST_ELMUL;
                        end else begin
                            idx_r <= idx_r + 1'b1;
                            ls    <= LS_FEED;
                        end
                    end
                    default: ls <= LS_IDLE;
                endcase

                // silu(gate)×up → hidden_mem → ST_DOWN
                ST_ELMUL: unique case (ls)
                    LS_IDLE: begin ls <= LS_CMD; idx_r <= '0; end
                    LS_CMD: if (em_cmd_v && em_cmd_r) begin ls <= LS_FEED; idx_r <= '0; end
                    LS_FEED: begin
                        if (em_a_v && em_a_r) ls <= LS_COLLECT;
                    end
                    LS_COLLECT: begin
                        if (em_z_v && em_z_r) begin
                            hidden_mem[idx_r/D_FF][idx_r%D_FF] <= em_z_d;
                            if (idx_r == f_dff_len(seq_r)-1) begin ls <= LS_CMD; st <= ST_DOWN; end
                            else begin idx_r <= idx_r + 1'b1; ls <= LS_FEED; end
                        end
                    end
                    default: ls <= LS_IDLE;
                endcase

                // 流式吐出 out_mem；握手接受时立刻更新下一拍 data/idx（防旧数据）
                ST_OUTPUT: begin
                    if (out_valid && out_ready) begin
                        if (idx_r == f_embd_len(seq_r)-1) begin
                            out_valid <= 1'b0;
                            st        <= ST_IDLE;
                            idx_r     <= '0;
                        end else begin
                            idx_r    <= idx_r + 1'b1;
                            out_idx  <= idx_r + 1'b1;
                            out_data <= out_mem[(idx_r + 1'b1)/EMBD][(idx_r + 1'b1)%EMBD];
                            out_valid <= 1'b1;
                        end
                    end else begin
                        out_valid <= 1'b1;
                        out_idx   <= idx_r;
                        out_data  <= out_mem[idx_r/EMBD][idx_r%EMBD];
                    end
                end

                // PRE_* 进度在上方 if 链推进；此处占位避免掉进 default→IDLE
                ST_PRE_QKV_W, ST_PRE_QKV_M,
                ST_PRE_WO_W,  ST_PRE_WO_M,
                ST_PRE_FG_W,  ST_PRE_FG_M,
                ST_PRE_FU_W,  ST_PRE_FU_M,
                ST_PRE_FD_W,  ST_PRE_FD_M: ;

                default: st <= ST_IDLE;
            endcase
        end
    end
endmodule
