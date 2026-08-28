# D4–D12 Linear / Attention / Block 详细笔记（2026-08-28）

> **用途：** 以后不想翻聊天、不想翻 PROGRESS 长文时，读本文件即可恢复 D4–D12 全貌。  
> 配套文档：[`PROGRESS.md`](PROGRESS.md)（阶段进度）、[`FILES.md`](FILES.md)（文件地图）。

---

## 0. 关账结论（先记住这几行）

### D12（当前主线，已关账）

| 项 | 值 |
|----|-----|
| 功能 | 单层 MiniGPT Transformer Block INT8 E2E（seq=1） |
| RTL | **`int8_transformer_block.sv`** |
| 路径 | QKV→Score→Softmax→AttnV→Concat→Wo→Res1→gate/up/SiLU/mul/down→Res2→Out |
| 规格 | `n_embd=64`，`n_head=4`，`head_dim=16`，`d_ff=256` |
| 仿真 TB | `tb_int8_transformer_block.sv` |
| 向量 | `transformer_block_vectors.txt`，seed=**20260908**，**1 case** |
| 仿真结果 | **1/1 EXACT**，~**2716160 ns** |
| 生成 | `python scripts/generate_transformer_block_vectors.py` |
| CLI | `scripts/run_transformer_block_xsim.bat` |
| Golden 要点 | EXPECT 必须用 **frozen weight/mult bank** 推演，与 preload 一致 |
| 踩坑 | feed 侧 data/地址须组合驱动；`l_cmd_v` 仅 `LS_CMD`；OUTPUT 握手同拍更新 data；SiLU 一进一出 |

### D7（仍有效）

| 项 | 值 |
|----|-----|
| 功能 | 同一 `int8_linear_tiled`：`op_type∈{WQ,WK,WV}` + `head_idx` 切换基地址 |
| 映射 | 见 **§0.2**；`WEIGHT_DEPTH=12288`，`MULT_DEPTH=192` |
| 仿真 TB | `tb_linear_tiled_qkv.sv` |
| 向量 | `tiled_qkv_vectors.txt`，seed=**20260831**，**5 case** |
| 仿真结果 | **5/5 PASS**，`mismatch_count=0`，~**524300 ns** |
| Case | head0 WQ/WK/WV M=4；WQ head1 M=4；WV head3 M=2 |
| 生成 | `python scripts/generate_linear_tiled_qkv_vectors.py` |
| CLI | `scripts/run_tiled_qkv_xsim.bat` |
| VCD Tcl | `scripts/xsim/launch_tiled_qkv_with_vcd.tcl` |

### D6（仍有效）

| 项 | 值 |
|----|-----|
| 功能 | head0 整层 Q：`[M,64]×Wq[16,64]^T→[M,16]` INT8 |
| RTL | **`int8_linear_tiled.sv`**（新建；`int8_linear_layer` **未改**） |
| Tile | 16 次 M×4×16；K 维 4 tile INT32 累加后再 quantize |
| 仿真 TB | `tb_linear_tiled_head0_q.sv` |
| 向量 | `tiled_head0_q_vectors.txt`，seed=**20260830**，**4 case** |
| 仿真结果 | **4/4 PASS**，`mismatch_count=0` |
| 仿真时长 | ~**261080 ns** |
| 生成 | `python scripts/generate_linear_tiled_head0_q_vectors.py` |
| CLI 回归 | `scripts/run_tiled_head0_q_xsim.bat` |
| VCD Tcl | `scripts/xsim/launch_tiled_head0_q_with_vcd.tcl` |

### D5（仍有效）

| 项 | 值 |
|----|-----|
| 功能 | 同 D4，权重来自 **MiniGPT 真实 checkpoint** |
| 权重 | `blocks.0.attention.qkv_proj` **Q 行子块**（Wq slice，M/N/K≤4/4/16） |
| Checkpoint | `F:/Users/22563/Desktop/learn/mini_gpt_ready/mini_gpt_checkpoint.pt` |
| 仿真 TB | `tb_linear_int8_gpt_vectors.sv` |
| 向量 | `gpt_linear_vectors.txt`，seed=**20260829**，**12 case** |
| 仿真结果 | **12/12 PASS**，`mismatch_count=0` |
| 仿真时长 | ~**63880 ns** |
| VCD | `gpt_linear_int8_full.vcd`（launch Tcl 生成） |
| 生成 | `python scripts/generate_linear_int8_gpt_vectors.py` |
| CLI 回归 | `scripts/run_linear_int8_gpt_xsim.bat` |

### D4（仍有效）

| 项 | 值 |
|----|-----|
| 功能 | INT8 Linear = GEMM(INT32) + 权重预加载 + requantize |
| RTL 顶层 | `int8_linear_layer.sv` |
| 仿真 TB | `tb_linear_int8_python_vectors.sv` |
| 向量 | `linear_int8_vectors.txt`，seed=**20260828**，24 case |
| 仿真结果 | **24/24 PASS**，`mismatch_count=0` |
| 仿真时长 | ~**59390 ns** |
| VCD | `linear_int8_full.vcd`，**274494 bytes**，**41 路信号** |
| 时钟 | TB `#5` 翻转 → **10 ns 周期 = 100 MHz**（与 xdc 一致） |

---

## 0.1 D6 架构要点

```text
cmd(M, N=16, K=64)
  → preload 1024 WEIGHT（tile-major：16 blocks × 64）+ 16 MULT
  → stream M×64 A
  → for n_base in {0,4,8,12}:
       clear acc[M][4]
       for k_base in {0,16,32,48}:
           replay_base = tile_idx*64
           gemm M×4×16 → INT32 partial
           acc += partial
       requantize 4 cols → INT8（全局 col = n_base+local）
```

**为何不能黑盒复用 `int8_linear_layer`：** 它每个子 GEMM 后立刻 requantize；K-tile 必须先累加 INT32。

**weight_loader D6 扩展：** `DEPTH=1024`，端口 `replay_base_addr`；D4/D5 绑 `replay_base_addr='0`。

**实现坑（已修）：** `TILE_IDLE` 里勿再写 `mult_wr_ptr <= mult_wr_ptr`，会盖掉 MULT 预加载的自增，导致 16 个 scale 全落到 `mult_mem[0]`。

---

## 0.2 D7 权重 / MULT 内存映射（冻结）

MiniGPT `qkv_proj.weight` = `[192,64]`：Q=`[0:64,:]`，K=`[64:128,:]`，V=`[128:192,:]`。  
`n_embd=64`，`n_head=4`，`head_dim=16`。每个 head 取连续 16 行。

```text
region(op, head) = op_base(op) + head_idx * 1024
  op_type: 0=WQ, 1=WK, 2=WV
  op_base: WQ=0, WK=4096, WV=8192
  每区 4 heads × 1024 = 4096；WEIGHT_DEPTH = 12288
  每 head 块内部：tile-major 16×64（与 D6 weight_tile_major_flat 相同）

tile 回放地址:
  replay_base = region(op,head) + tile_idx * 64
  tile_idx = n_tile * 4 + k_tile   （n_tile,k_tile ∈ 0..3）

MULT 表（独立基址，避免串读）:
  mult_base(op, head) = op * 64 + head * 16
  MULT_DEPTH = 3 * 4 * 16 = 192
  读取: mult_mem[mult_base + col]，col ∈ 0..15

单次 cmd:
  输入  [M,64] INT8（M≤4）
  输出  [M,16] INT8，布局 [token][head_dim]
  Q→K→V 由上层/TB 依次发 cmd（本阶段不接 Score）
```

| 字段 | 规则 |
|------|------|
| `cmd_op_type[1:0]` | 0=WQ, 1=WK, 2=WV |
| `cmd_head_idx[1:0]` | 0..3 |
| `base_addr` | `op_base + head_idx*1024` |
| 输出 shape | `[M,16]`，`c_row/c_col` 为 head 内坐标 |

D6 兼容：`cmd_op_type=0`、`cmd_head_idx=0` → 行为与原 head0 Q 一致（region=0）。

---

## 1. 工程里 D4 到底新增/改了什么

### 1.1 新增 RTL（2 个）

| 文件 | 作用 |
|------|------|
| `int8_weight_loader.sv` | 小 RAM：TB 经 `w_*` 预写 B 矩阵；GEMM 进入 LOAD_B 时经 `r_*` 按序回放 |
| `int8_linear_layer.sv` | D4 顶层：实例化 `int8_gemm_parallel` + `int8_weight_loader` + `requantize_int8_pipeline` |

### 1.2 复用 RTL（未改源码）

| 文件 | 在 D4 中的角色 |
|------|----------------|
| `int8_gemm_parallel.sv` | 并行 GEMM 控制器 + 4-lane 点积，输出 **INT32** `c_data` |
| `int8_dot_product_parallel.sv` | 被 GEMM 实例化，算单行×列点积 |
| `requantize_int8_pipeline.sv` | INT32 acc × 18-bit scale → INT8，**2 拍流水** |

### 1.3 新增 TB / 脚本 / 向量（D4）

| 文件 | 作用 |
|------|------|
| `tb_linear_int8_python_vectors.sv` | 读向量、驱动 DUT、逐元素比对 INT8 |
| `scripts/generate_linear_int8_vectors.py` | Python golden 生成器 |
| `linear_int8_vectors.txt` | 24 组 CASE（4 定向 + 20 随机） |
| `scripts/xsim/launch_linear_int8_with_vcd.tcl` | GUI 加波 + 录 VCD + `run all` |
| `scripts/xsim/linear_int8_wave_common.tcl` | 41 路信号列表 + TB 自动检测 |

### 1.3b D5 新增（不改 RTL）

| 文件 | 作用 |
|------|------|
| `scripts/export_mini_gpt_wq_slice.py` | 从 MiniGPT checkpoint 取 Q 权重 + norm1 激活，v2 量化 |
| `scripts/generate_linear_int8_gpt_vectors.py` | 写 `gpt_linear_vectors.txt` |
| `gpt_linear_vectors.txt` | 12 case 真实 Wq slice |
| `gpt_wq_export_meta.json` | slice 坐标、scales、checkpoint 路径 |
| `tb_linear_int8_gpt_vectors.sv` | D5 独立 TB |
| `scripts/xsim/launch_linear_int8_gpt_with_vcd.tcl` | D5 波形 + VCD |
| `scripts/run_linear_int8_gpt_xsim.bat` | CLI 一键回归 |

**Wq 说明：** MiniGPT 无独立 Wq 模块；Q/K/V 融合在 `qkv_proj`。D5 取 `weight[0:64,:]` 的行子块。

### 1.4 模块层次（Design Sources 展开后）

```text
int8_linear_layer                    ← 综合 Top（D4）
├── u_gemm : int8_gemm_parallel
│   └── u_dot_product : int8_dot_product_parallel
├── u_weight_loader : int8_weight_loader
└── u_requant : requantize_int8_pipeline
```

Design Sources 顶层只显示 **(4)** 个根节点是正常的；展开 `int8_linear_layer (3)` 后能看到完整子树，**10 个 RTL 文件都在 `.xpr` 里**。

---

## 2. 数据流总览（从向量到 INT8 输出）

```text
linear_int8_vectors.txt
        │
        ▼
tb_linear_int8_python_vectors
        │
        ├─ WEIGHT 段 ──w_valid/w_data──► int8_weight_loader.mem[]
        ├─ MULT 段  ──mult_valid/mult_data──► int8_linear_layer.mult_mem[0..N-1]
        ├─ cmd M/N/K ───────────────────────► int8_gemm_parallel（经顶层 cmd_ready 门控）
        ├─ A 段     ──a_valid/a_data──────► int8_gemm_parallel LOAD_A → a_mem[]
        │                                      LOAD_B 时 b 来自 loader.r_*，不是 TB
        │                                      DOT_* 态调用 4-lane 点积
        │                                      OUTPUT_C 吐 INT32 gemm_c_*
        │
        └─ EXPECT 段 ◄──c_valid/c_data/c_row/c_col── 顶层 out_state FSM + requantize
```

**和 D3 的关键区别：**

- D3：TB 在 LOAD_B 阶段**实时送** B 矩阵元素  
- D4：TB 事先用 `w_*` **预加载**进 loader；LOAD_B 时 loader **回放**；同时每列有独立 **requantize multiplier**

---

## 3. RTL 代码详解

### 3.1 `int8_linear_layer.sv` — 顶层

#### 端口一览

| 端口组 | 方向 | 含义 |
|--------|------|------|
| `cmd_*` | in/out | 矩阵维度 M/N/K；仅 `OUT_IDLE` 时 `cmd_ready=1` |
| `w_*` | in/out | 预加载权重（signed INT8），写进 loader RAM |
| `mult_*` | in/out | 预加载每列 18-bit unsigned multiplier；仅 `OUT_IDLE` 时 `mult_ready=1` |
| `a_*` | in/out | 流式激活 A，直通 GEMM LOAD_A |
| `c_*` | out/in | **INT8** 输出 + row/col；带 ready/valid 握手 |
| `acc_debug` | out | 当前输出元素对应的 **INT32 累加值**（波形调试） |

#### 关键连线（理解波形必看）

```systemverilog
// cmd 只在顶层 OUT_IDLE 且 GEMM 也在 IDLE 时可接受
assign cmd_ready    = gemm_cmd_ready && (out_state == OUT_IDLE);
assign mult_ready   = gemm_cmd_ready && (out_state == OUT_IDLE);

// B 矩阵来自 weight_loader，不是 TB
assign gemm_b_valid = loader_r_valid;
assign gemm_b_data  = loader_r_data;
assign loader_r_ready = gemm_b_ready;

// GEMM 的 INT32 输出只有在 OUT_PRESENT 且下游握手后才 advance
assign gemm_c_ready = (out_state == OUT_PRESENT) && c_valid && c_ready;
```

#### 顶层输出 FSM（`out_state`）

| 状态 | 编码 | 做什么 |
|------|------|--------|
| `OUT_IDLE` | 0 | 等 `gemm_c_valid`；捕获 INT32 到 `held_acc/held_row/held_col` |
| `OUT_FEED` | 1 | 打一拍 `requant_in_valid=1` |
| `OUT_REQUANT` | 2 | 等 `requant_out_valid`（pipeline 第 2 拍出 INT8） |
| `OUT_PRESENT` | 3 | `c_valid=1`，present INT8；等 `c_ready` 握手后回 IDLE |

**时序关系（每个 GEMM 输出元素）：**

```text
gemm_c_valid ──┐
               ▼
OUT_IDLE (capture acc) → OUT_FEED (in_valid) → OUT_REQUANT (等2拍) → OUT_PRESENT (c_valid)
                                                                              │
                                         gemm_c_ready 仅在 OUT_PRESENT∧c_ready 时为 1
```

#### `cmd_accept` 时发生的副作用

```systemverilog
if (cmd_accept) begin
    replay_start  <= 1'b1;           // 脉冲：启动 weight loader 回放
    replay_length <= cmd_k * cmd_n;  // 回放 K×N 个权重
    mult_wr_ptr   <= '0;             // mult 指针复位（mult 已在 cmd 前预加载完）
end
```

#### `mult_mem` 写入

```systemverilog
if (mult_valid && mult_ready) begin
    mult_mem[mult_wr_ptr] <= mult_data;
    mult_wr_ptr           <= mult_wr_ptr + 1'b1;
end
```

requantize 时用 **`mult_mem[held_col]`** — 按**输出列**选 multiplier，不是按 flat index。

---

### 3.2 `int8_weight_loader.sv` — 权重 RAM

#### 行为摘要

| 阶段 | `w_ready` | `replay_active` | `r_valid` |
|------|-----------|-----------------|-----------|
| 预加载（TB 写权重） | 1 | 0 | 0 |
| 回放中（GEMM LOAD_B） | **0**（禁止写） | 1 | 1（若 `r_remaining≠0`） |
| 回放结束 | 1 | 0 | 0 |

#### 存储布局

权重按 **列优先** 顺序写入，与向量文件 WEIGHT 段一致：

```text
index = kk * N + j    （kk=0..K-1, j=0..N-1）
mem[0]=B[0][0], mem[1]=B[1][0], ...  （先填满第 0 列，再第 1 列…）
```

这与 `int8_gemm_parallel` 的 `b_mem` 加载顺序一致，LOAD_B 时 GEMM 从 `b_valid/b_data` 收的数据就是回放顺序。

#### 回放结束条件

每次 `r_valid && r_ready` 握手，`r_remaining--`；减到 1 后再握手一次则 `replay_active=0`，`w_wr_addr` 复位，可开始下一 case 预加载。

---

### 3.3 `int8_gemm_parallel.sv` — GEMM 控制器（子模块）

#### FSM 状态

```text
IDLE → LOAD_A → LOAD_B → DOT_CMD → DOT_FEED → DOT_WAIT → OUTPUT_C
  ↑                                                              │
  └──────────────── 下一 (row,col) 或全部完成回 IDLE ─────────────┘
```

| 状态 | `a_ready` | `b_ready` | 说明 |
|------|-----------|-----------|------|
| `IDLE` | 0 | 0 | 等 cmd |
| `LOAD_A` | **1** | 0 | 收 M×K 个 A 进 `a_mem` |
| `LOAD_B` | 0 | **1** | 收 K×N 个 B 进 `b_mem`（D4 来自 loader） |
| `DOT_CMD` | 0 | 0 | 给点积核发 K |
| `DOT_FEED` | 0 | 0 | 按 beat 送 4-lane a/b |
| `DOT_WAIT` | 0 | 0 | 等点积 `m_valid` |
| `OUTPUT_C` | 0 | 0 | `c_valid=1`，输出 INT32 |

#### 规格上限

`MAX_M=4, MAX_N=4, MAX_K=16`；`PARALLELISM=4`（每 beat 最多 4 个乘加 lane）。

---

### 3.4 `requantize_int8_pipeline.sv` — 量化流水线

#### 数学（与 Python golden 一致）

```text
product = acc_s32 × multiplier_u18        // mult 零扩展后参与有符号乘
rounded = product + half_lsb              // half_lsb = 1 << (SHIFT_BITS-1)
          或 product + (half_lsb - 1)     // 负数时
shifted = rounded >>> SHIFT_BITS          // 算术右移，SHIFT_BITS=24
out     = saturate(shifted, -127, 127)    // 对称饱和
```

#### 流水延迟

| 拍 | 寄存器 | 内容 |
|----|--------|------|
| in_valid 当拍 | → S1 | `product_s1 = acc × mult` |
| +1 拍 | → S2 | round + shift 组合结果打入 S2 |
| +2 拍 | out_valid | 饱和后 `out_i` |

顶层 `OUT_FEED` 打 `requant_in_valid`，`OUT_REQUANT` 等 `requant_out_valid`，与这 2 拍对齐。

---

## 4. Python Golden 与向量文件

### 4.1 生成命令

```bat
cd /d F:\Users\22563\Desktop\FPGAlearn\requantize_int8_rtl
python scripts\generate_linear_int8_vectors.py
```

默认：`--seed 20260828`，`--random-count 20` → 4 定向 + 20 随机 = **24 case**。

### 4.2 向量文件格式

```text
# seed=20260828 num_cases=24
CASE <id> <M> <N> <K> <a_gap> <c_hold>
A
<M*K 行，每行一个 signed INT8>
WEIGHT
<K*N 行，每行一个 signed INT8，列优先 B[kk][j]>
MULT
<N 行，每行一个 unsigned 18-bit multiplier，对应列 j=0..N-1>
EXPECT
<M*N 行，每行 requantize 后的 signed INT8>
END
```

| 字段 | 含义 |
|------|------|
| `a_gap` | 从第 2 个 A 元素起，每次发送前插几个空拍（测 backpressure） |
| `c_hold` | `c_valid` 上升后，延迟几个周期再拉 `c_ready`（测输出保持） |

### 4.3 Case 1 完整示例（波形对照用）

**头部：** `CASE 1 2 2 3 0 2` → M=2, N=2, K=3, a_gap=0, c_hold=2

**A（2×3，行优先）：**

```text
行0: 1, 2, 3
行1: 4, 5, 6
```

**WEIGHT（3×2，列优先存储）：**

```text
列0: 7, 8, 9     → B[0][0]=7, B[1][0]=8, B[2][0]=9
列1: 10, 11, 12  → B[0][1]=10, B[1][1]=11, B[2][1]=12
```

**MULT：** 列0=26979，列1=88887

**手算 INT32 GEMM：**

| 输出 | 计算 | acc |
|------|------|-----|
| C[0][0] | 1×7 + 2×8 + 3×9 | **58** |
| C[0][1] | 1×10 + 2×11 + 3×12 | **64** |
| C[1][0] | 4×7 + 5×8 + 6×9 | **139** |
| C[1][1] | 4×10 + 5×11 + 6×12 | **154** |

**requantize 后 INT8（mult 按列）：**

| 输出 | acc | mult | INT8 |
|------|-----|------|------|
| C[0][0] | 58 | 26979 | **0** |
| C[0][1] | 64 | 88887 | **0** |
| C[1][0] | 139 | 26979 | **0** |
| C[1][1] | 154 | 88887 | **1** |

终端应看到：

```text
PASS case=1 C[0][0]=0
PASS case=1 C[0][1]=0
PASS case=1 C[1][0]=0
PASS case=1 C[1][1]=1
```

---

## 5. Testbench 代码详解

**文件：** `requantize_int8_rtl.srcs/sim_1/new/tb_linear_int8_python_vectors.sv`

### 5.1 时钟与复位

```systemverilog
forever #5 clk = ~clk;     // 5ns 半周期 → 100MHz
repeat (4) @(posedge clk);
rst_n = 1'b1;              // 约 40ns 后释放复位
```

### 5.2 单个 CASE 的执行顺序（`run_vector_case`）

```text
1. read_int_block   → 读 A 段（M×K 个数）
2. read_int_block   → 读 WEIGHT 段（K×N 个数）
3. read_mult_block  → 读 MULT 段（N 个数）
4. read_int_block   → 读 EXPECT 段（M×N 个数）
5. for K×N: send_weight(w_vals[idx])
6. for N:   send_mult(mult_vals[idx])
7. send_command(m, n, k)
8. for M×K: send_a(a_vals[idx], gap)
9. for i,j: receive_c(i, j, expected, case_id, c_hold)
10. test_count++
```

**注意：** 驱动顺序是 **先 WEIGHT、再 MULT、再 cmd、再 A**。mult 必须在 cmd 前送完，因为 cmd 接受后 `mult_wr_ptr` 会复位（mult 已进 `mult_mem` 不受影响）。

### 5.3 握手 task 时序模式（所有 send_* 共用）

```text
@(negedge clk)  → 驱动 valid=1, data=...
while (!ready) @(negedge clk)
@(posedge clk)
@(negedge clk)  → valid=0, data=0
```

### 5.4 `receive_c` — 比对 + 反压测试

```text
c_ready=0
while (!c_valid) wait
采样 c_data, c_row, c_col → 与 expected 比（不等则 mismatch_count++）
空等 c_hold 个周期，检查 c_valid 和数据是否保持稳定
c_ready=1 一个周期 → 握手
c_ready=0
```

FAIL 时会打印 `acc_debug`，方便对照 INT32 是否算错还是 requantize 算错。

### 5.5 最终 PASS 条件

```systemverilog
header_seed == 20260828
header_num_cases == 24
test_count == 24
mismatch_count == 0
→ "PYTHON AND RTL ARE 24/24 EXACT."
```

---

## 6. 仿真操作（你刚才跑的那套）

### 6.1 Top 设置

| 用途 | Top |
|------|-----|
| 综合 / 实现 | `int8_linear_layer` |
| 行为仿真 | `tb_linear_int8_python_vectors` |

### 6.2 一键流程

1. Vivado → Simulation Sources → `tb_linear_int8_python_vectors` → **Set as Top**
2. **Run Behavioral Simulation**
3. xsim Tcl Console：

```tcl
restart
source F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/scripts/xsim/launch_linear_int8_with_vcd.tcl
```

### 6.3 `launch_linear_int8_with_vcd.tcl` 逐步做了什么

| 步骤 | Tcl / 动作 | 说明 |
|------|------------|------|
| 1 | `source linear_int8_wave_common.tcl` | 加载信号列表 proc |
| 2 | 检查 `get_objects` 非空 | 若 `$finish` 后再 source 会失败 |
| 3 | `resolve_linear_tb_top` | 自动找 `/tb_linear_int8_python_vectors/clk` |
| 4 | `close_wave_config -force` 全部 | 清旧波形配置 |
| 5 | `create_wave_config linear_int8_d4` | 新建 wave 窗口 |
| 6 | `add_linear_int8_waves` | GUI 加 **41** 路 |
| 7 | `open_vcd linear_int8_full.vcd` | 开始录 VCD |
| 8 | `log_linear_int8_vcd` | 注册 41 路 log_vcd |
| 9 | `run all` | 跑完 24 case |
| 10 | `close_vcd` | 关闭，检查 size ≥ 4096 |

**关键：** VCD 必须在 `run all` **之前** `open_vcd`；WDB 不能事后转 VCD。

---

## 7. 波形与 VCD 详解（重点）

### 7.1 VCD 文件信息

| 项 | 值 |
|----|-----|
| 路径 | `F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/linear_int8_full.vcd` |
| 大小 | 274494 bytes |
| timescale | **1 ps**（GTKWave 里 `#5000` = 5 ns） |
| 层次根 | `tb_linear_int8_python_vectors` → `dut` → 三个子模块 |

### 7.2 全部 41 路信号（GUI 与 VCD 相同）

#### A. TB ↔ DUT 顶层接口（23 路）

| 信号路径 | 位宽 | 看什么 |
|----------|------|--------|
| `clk` | 1 | 100 MHz 时钟 |
| `rst_n` | 1 | 低有效复位，~40ns 拉高 |
| `cmd_valid` / `cmd_ready` | 1 | cmd 握手 |
| `cmd_m` / `cmd_n` / `cmd_k` | 3/3/5 | 当前 case 维度 |
| `a_valid` / `a_ready` / `a_data` | 1/1/8 | LOAD_A 阶段流式激活 |
| `w_valid` / `w_ready` / `w_data` | 1/1/8 | 预加载权重 |
| `mult_valid` / `mult_ready` / `mult_data` | 1/1/18 | 预加载 requantize scale |
| `c_valid` / `c_ready` / `c_data` | 1/1/8 | **最终 INT8 输出** |
| `c_row` / `c_col` | 3/3 | 输出坐标 |
| `acc_debug` | 32 | 对应 INT32 累加值 |

#### B. `dut` 顶层控制（3 路）

| 信号 | 含义 |
|------|------|
| `dut/out_state` | 0=IDLE, 1=FEED, 2=REQUANT, 3=PRESENT |
| `dut/replay_start` | cmd 接受时单周期脉冲，启动 loader |
| `dut/replay_length` | = K×N，回放权重个数 |

#### C. `dut/u_weight_loader`（5 路）

| 信号 | 含义 |
|------|------|
| `replay_active` | 1=正在回放，此时 w_ready=0 |
| `r_valid` / `r_ready` | 读口握手，接 GEMM b_valid/b_ready |
| `r_addr` | 当前读地址 |
| `r_remaining` | 剩余待回放个数 |

#### D. `dut/u_gemm`（6 路）

| 信号 | 含义 |
|------|------|
| `state` | 0=IDLE, 1=LOAD_A, 2=LOAD_B, 3=DOT_CMD, 4=DOT_FEED, 5=DOT_WAIT, 6=OUTPUT_C |
| `c_valid` / `c_ready` / `c_data` | **INT32** GEMM 输出（进 requantize 前） |
| `c_row` / `c_col` | GEMM 输出坐标 |

#### E. `dut/u_requant`（3 路）

| 信号 | 含义 |
|------|------|
| `in_valid` | 顶层 OUT_FEED 打入 |
| `out_valid` | 2 拍后有效 |
| `out_i` | 饱和后 INT8 |

#### F. 统计（2 路）

| 信号 | 期望终值 |
|------|----------|
| `test_count` | 24 |
| `mismatch_count` | 0 |

### 7.3 VCD 符号对照（Case 1 段 GTKWave 解码用）

VCD 里信号用短符号，对照 `$var` 定义：

| 符号 | 信号 | 符号 | 信号 |
|------|------|------|------|
| `!` | clk | `@` | u_gemm/c_valid |
| `"` | rst_n | `A` | u_gemm/c_ready |
| `#` | cmd_valid | `B` | u_gemm/c_data (INT32) |
| `$` | cmd_ready | `C/D` | u_gemm/c_row/c_col |
| `%` | cmd_m | `?` | u_gemm/state |
| `&` | cmd_n | `7` | dut/out_state |
| `'` | cmd_k | `8/9` | replay_start/length |
| `(` / `)` | a_valid/a_ready | `E/F/G` | requant in/out/out_i |
| `*` | a_data | `:` / `;` | replay_active/r_valid |
| `+` / `,` / `-` | w_valid/w_ready/w_data | `H/I` | test/mismatch count |
| `.` / `/` / `0` | mult_valid/ready/data | | |
| `1` / `2` / `3` | c_valid/c_ready/c_data | | |
| `4` / `5` / `6` | c_row/c_col/acc_debug | | |

### 7.4 Case 1 波形时间轴（约 40 ns – 1100 ns）

以下时间基于 VCD `#timestamp`（单位 ps），÷1000 得 ns。

| 时间 (ns) | 主要信号变化 | 含义 |
|-----------|--------------|------|
| 0–40 | `rst_n=0` | 复位 |
| 40 | `rst_n=1` | 释放复位，Case 1 开始 |
| ~50–180 | `w_valid` 脉冲 ×6，`w_data`=7,8,9,10,11,12 | 预加载 6 个权重 |
| ~180–250 | `mult_valid` ×2，`mult_data`=26979, 88887 | 预加载 2 个 multiplier |
| ~280–320 | `cmd_valid=1`，`cmd_m/n/k`=2,2,3 | 发送 cmd |
| ~320 | `replay_start` 脉冲，`replay_length`=6 | 启动 loader 回放 K×N |
| ~330–420 | `a_valid`/`a_data` 依次 1,2,3,4,5,6 | LOAD_A（6 个激活） |
| ~400–500 | `u_gemm/state`=LOAD_B(2)，`r_valid`/`r_ready` 握手 | loader 向 GEMM 回放 B |
| ~500–650 | `u_gemm/state` 在 DOT_* 间切换 | 点积计算 |
| ~465 | `u_gemm/c_valid`，`c_data`=58 (0x3A) | 第一个 INT32：C[0][0] |
| ~475 | `out_state`: 0→1→2 | 进入 requantize |
| ~485–515 | `u_requant/in_valid` → `out_valid`，`out_i`→0 | INT8 结果 0 |
| ~525–575 | `c_valid=1`，`c_data=0`，`c_row/col=0/0` | 顶层输出 C[0][0] |
| ~550–870 | 同样模式重复 3 次 | C[0][1], C[1][0] |
| ~1005 | `c_data=1`，`c_row=1`，`c_col=1` | **C[1][1]=1**，Case 1 完成 |

**GTKWave 操作建议：**

1. 打开 `linear_int8_full.vcd`
2. 展开 `tb_linear_int8_python_vectors` → `dut`
3. **Time → Goto → 40000**（40 ns）看复位结束
4. 把 `cmd_*`, `w_*`, `a_*`, `c_*`, `dut/out_state`, `dut/u_gemm/state` 拖入波形
5. **Time → Goto → 1000000**（1000 ns）看 Case 1 最后一个输出
6. 用 `test_count` 上升沿数 case 进度

### 7.5 两个 FSM 如何配合（看波形最容易晕的地方）

同一时刻可能在跑：

```text
u_gemm/state        : LOAD_A / LOAD_B / DOT_* / OUTPUT_C
dut/out_state       : IDLE / FEED / REQUANT / PRESENT
u_weight_loader     : replay_active
```

**典型顺序（每个输出元素）：**

```text
1. GEMM 在 OUTPUT_C 态：u_gemm/c_valid=1，INT32 出现在 u_gemm/c_data
2. 顶层 OUT_IDLE 捕获 → OUT_FEED 打 requant_in_valid
3. 2 拍后 requant_out_valid，顶层 OUT_PRESENT，c_valid=1
4. TB receive_c 看到 c_valid，比对 INT8
5. c_ready=1 握手 → gemm_c_ready=1 → GEMM 可输出下一个 INT32
```

所以在波形上：**u_gemm/c_valid 比顶层 c_valid 早出现**，中间隔 requantize 流水 + 顶层 FSM。

### 7.6 饱和 case 在波形里长什么样

Case 4、11、13 等 EXPECT 里大量 **±127**：

- 看 `u_gemm/c_data`：INT32 可能很大  
- 看 `u_requant/out_i` 或顶层 `c_data`：钳在 **0x7F (127)** 或 **0x81 (-127)**  
- `acc_debug` 仍显示 raw INT32，便于确认是「算对了但饱和」而非 GEMM 错

### 7.7 `c_hold` 反压在波形里

向量里 `c_hold=2` 或 `3` 的 case：

- `c_valid=1` 后，`c_ready` 仍保持 0 多个周期  
- `c_data/c_row/c_col` 必须**不变**；TB 若检测到变化会 `FAIL backpressure`  
- 握手发生在 `c_ready` 拉高那一拍

---

## 8. 综合 Top 与后续

### 8.1 综合

- **Design Top：** `int8_linear_layer`（`.xpr` 已设）  
- **约束：** `constrs_1/new/int8_gemm_serial_100mhz.xdc`（100 MHz）  
- **器件：** xc7z020clg484-1  
- 操作：Run Synthesis → 看 Utilization / Timing Summary  

### 8.2 参考：之前 standalone 点积综合（对比用）

| 资源 | int8_dot_product_parallel standalone |
|------|--------------------------------------|
| LUT | 388 (0.73%) |
| FF | 212 (0.20%) |
| DSP | 0 |
| WNS | +4.619 ns (post-synth) |

Linear 顶层综合资源会更大（含 GEMM 控制器 + loader + requantize），待你 Run Synthesis 后填入。

### 8.3 下一步

1. **综合** `int8_linear_tiled` / `int8_transformer_block` — 硬件侧关账  
2. **（可选）E2E seq=4** — 现仅关账 seq=1  
3. **（可选）Run Implementation** — post-route 时序  

---

## 9. 踩坑清单

| 现象 | 原因 | 处理 |
|------|------|------|
| Tcl 报 `tb_xxx/clk` 找不到 | Sim Top 不对 | D4→`tb_linear_int8_python_vectors`；D12→`tb_int8_transformer_block` |
| `$finish` 后 source launch | 仿真对象已销毁 | 先 `restart` |
| VCD 只有 ~138 bytes | 没在 run 前 open_vcd | 用 launch 一体脚本 |
| `mult_ready=0` 送 mult | 顶层不在 OUT_IDLE | 等上一 case 的 c 握手完成 |
| `w_ready=0` 写权重 | loader 在 replay | 等 GEMM LOAD_B 结束 |
| Design Sources 只有 (4) | 层次折叠 | 展开 `int8_linear_layer` |
| bat 只跑 [1/3] | 缺 `call xvlog` | 见项目 workflow 规则 |
| D12 preload 后挂死 | `ST_PRE_*` 掉进 `default→IDLE` | unique case 显式留空 PRE 态 |
| D12 Softmax 卡 COLLECT | row/col 寄存器晚一拍 | feed 侧组合驱动 `sm_in_*` |
| D12 SiLU 喂完卡住 | 1-deep 管，FEED 时 `out_ready=0` | 一进一出：FEED→COLLECT 交替 |
| D12 输出整体错位 | OUTPUT 握手同拍用旧 idx 出 data | 接受时立即换下一拍 data/idx |
| D12 ±1 大量 FAIL | golden 重算 scale/mult | EXPECT 用 frozen bank + `golden_requantize` |

---

## 10. 关键文件路径速查

```text
RTL:
  int8_transformer_block.sv          ← D12 Top
  int8_linear_tiled.sv               ← D6/D7/D10/D11 Linear
  int8_score_gemm.sv / softmax / attn_v / residual / silu / elem_mul

TB:
  tb_int8_transformer_block.sv       ← D12 E2E
  tb_linear_tiled_qkv.sv             ← D7

向量 / CLI:
  transformer_block_vectors.txt
  scripts/generate_transformer_block_vectors.py
  scripts/run_transformer_block_xsim.bat
  scripts/run_transformer_block_debug.bat
```

---

## 11. 新开聊天复制给 AI

```
请先读 NOTES.md 与 PROGRESS.md。
D12 已关账：int8_transformer_block E2E seq=1，seed=20260908，1/1 EXACT。
CLI：scripts/run_transformer_block_xsim.bat；向量=transformer_block_vectors.txt。
D4–D11 分层回归仍 PASS。下一步：综合 / 可选 seq=4。
```
