# requantize_int8_rtl 进度记录

> 最后更新：2026-09-03  
> 用途：新开 Cursor 聊天时先读此文件，快速恢复上下文。  
> **不想细看 → [`NOTES.md`](NOTES.md)（D4–D6 速查）**  
> **文件地图见 [`FILES.md`](FILES.md)**（每个文件干什么、哪些目录可忽略）。

---

## 项目路径

`F:\Users\22563\Desktop\FPGAlearn\requantize_int8_rtl`

---

## 当前主线：D12 Transformer Block Top（已关账）

**RTL：** `int8_transformer_block.sv`（QKV→Score→Softmax→AttnV→Concat→Wo→Res1→FFN→Res2）  
**规格：** `n_embd=64`，`n_head=4`，`head_dim=16`，`d_ff=256`，E2E **seq=1**  
**仿真 TB：** `tb_int8_transformer_block.sv`  
**向量：** `transformer_block_vectors.txt`，seed=**20260908**，**1/1 EXACT**  
**CLI：** `scripts/run_transformer_block_xsim.bat`  
**仿真时长：** ~**2716160 ns**

**D8–D11（分层仍有效）：** Score / Softmax / Attn@V+Wo / Residual+SiLU+FFN 均已对拍关账  
**D7（仍有效）：** `tb_linear_tiled_qkv.sv` — seed=20260831，5/5 PASS  
**D6（仍有效）：** `tb_linear_tiled_head0_q.sv` — seed=20260830，4/4 PASS  
**D5（仍有效）：** `tb_linear_int8_gpt_vectors.sv` — seed=20260829，12/12 PASS  
**D4（仍有效）：** `tb_linear_int8_python_vectors.sv` — seed=20260828，24/24 PASS

---

## §5 综合结果（2026-08-27）

**器件：** `xc7z020clg484-1`  
**约束：** `int8_gemm_serial_100mhz.xdc`（`create_clock -period 10.000`，100 MHz）

### 并行 standalone（`int8_dot_product_parallel` 作 Top）

| 资源 | 用量 | 备注 |
|------|------|------|
| LUT | **388** | 0.73% |
| FF | **212** | 0.20% |
| DSP | **0** | 乘法映射为 LUT，非 DSP48 |
| BRAM | 0 | — |

| 时序（post-synthesis） | 值 |
|------------------------|-----|
| WNS | **+4.619 ns** |
| WHS | **+0.137 ns** |
| Setup/Hold 失败端点 | **0** |

### 与串行对比（口径说明）

| 指标 | 串行（Notion：GEMM 内层次） | 4 路并行（standalone） | 变化 |
|------|------------------------------|------------------------|------|
| LUT | 158 | 388 | 约 2.5× |
| FF | 99 | 212 | 约 2.1× |
| DSP | — | 0 | 均未用 DSP48 |
| WNS | GEMM 整体 +2.918 ns（synth） | +4.619 ns（synth） | 均满足 100 MHz |

> IOB 171 为顶层端口 IBUF/OBUF，片内对比时忽略。  
> 公平对比待补：`dot_product_int8` 单独 Top 综合（同 Part、同 xdc）。

---

## 验证进度

| 阶段 | 状态 | TB | 说明 |
|------|------|-----|------|
| Task B | ✅ PASS | `tb_int8_dot_product_parallel.sv` | 单拍 partial/acc 时序，14 case |
| Task C | ✅ PASS | 同上 | FSM 定向 + 10 随机，共 36 case |
| Task D（点积） | ✅ PASS | `tb_dot_product_parallel_python_vectors.sv` | Python 向量 120 case，0 mismatch |
| D3 并行 GEMM | ✅ PASS | `tb_int8_gemm_parallel.sv` | 内联 24 case，0 mismatch |
| D3 Python 向量 | ✅ PASS | `tb_gemm_parallel_python_vectors.sv` | seed=20260827，24/24 exact |
| D4 Linear + Requant | ✅ PASS | `tb_linear_int8_python_vectors.sv` | seed=20260828，24/24 exact，INT8 输出 |
| D5 GPT Wq 真实权重 | ✅ PASS | `tb_linear_int8_gpt_vectors.sv` | seed=20260829，12/12 exact，MiniGPT qkv Q slice |
| D6 Full Tile head0 Q | ✅ PASS | `tb_linear_tiled_head0_q.sv` | seed=20260830，4/4 exact，N=16 K=64，16-tile |
| D7 Q/K/V 多 Head 地址 | ✅ PASS | `tb_linear_tiled_qkv.sv` | seed=20260831，5/5 exact；head0 Q/K/V + head1 WQ + head3 WV |
| D8 Score GEMM | ✅ PASS | `tb_int8_score_gemm.sv` | seed=20260903，8/8；Q@K^T INT32 |
| D9 Softmax causal | ✅ PASS | `tb_int8_softmax_causal.sv` | seed=20260902，6/6；LUT 定点 |
| D10 Attn@V + Wo | ✅ PASS | `tb_int8_attn_v` / `tb_linear_tiled_wo` | Attn 5/5；Wo 5/5 |
| D11 Residual + FFN | ✅ PASS | residual / silu / ffn_layer | Res 5/5；SiLU 3/3；FFN 3/3 |
| D12 Block E2E | ✅ PASS | `tb_int8_transformer_block.sv` | seed=20260908，**1/1 EXACT**，seq=1 |
| §5 综合 | 🔄 待做 | — | D4 Top=`int8_linear_layer` 综合；点积 standalone 已完成 |

### D12 Transformer Block E2E（seed=20260908）

- 规格：MiniGPT block0 INT8，`n_embd=64` `n_head=4` `d_ff=256`，E2E **seq=1**
- RTL：`int8_transformer_block.sv` 编排已验证子模块（QKV tiled / score / softmax / attn_v / wo / residual / silu / elem_mul / FFN tiled）
- 向量：`transformer_block_vectors.txt`（PRELOAD ~66k weight/mult + 1 CASE）
- Golden：与 RTL **同一套 frozen mult bank** 推演（勿用重新算 scale 的 `run_linear_int8` 链）
- 结果：`TEST RESULT: PASS 1/1 EXACT`，~**2716160 ns**
- CLI：`scripts/run_transformer_block_xsim.bat`
- Debug：`scripts/run_transformer_block_debug.bat`（`TRANSFORMER_BLOCK_DBG` + 短 run VCD）
- 回归：D4 24/24、D6 4/4、D7 5/5、D8–D11 分层均 PASS

### D7 Q/K/V 多 Head 结果（seed=20260831）

- Checkpoint：同 D5/D6 MiniGPT；`qkv_proj.weight[192,64]`
- 映射：`region=op_base+head*1024`；`op_base={0,4096,8192}`；`MULT_DEPTH=192`
- RTL：`int8_linear_tiled` 增加 `cmd_op_type` / `cmd_head_idx`；默认 `WEIGHT_DEPTH=12288`
- 向量：`tiled_qkv_vectors.txt`（PRELOAD 12288 WEIGHT + 192 MULT，再 5 CASE）
- Case：WQ/WK/WV head0 M=4；WQ head1 M=4；WV head3 M=2
- RTL：`test_count=5`，`mismatch_count=0`，`PYTHON AND RTL ARE 5/5 EXACT`，约 **524300 ns**
- 生成：`python scripts/generate_linear_tiled_qkv_vectors.py`
- CLI：`scripts/run_tiled_qkv_xsim.bat`
- Tcl：`scripts/xsim/launch_tiled_qkv_with_vcd.tcl`
- 回归：D6 4/4、D5 12/12、D4 24/24 PASS

### D6 Full Tile head0 Q 结果（seed=20260830）

- Checkpoint：同 D5 MiniGPT；Wq_head0 = `qkv_proj.weight[0:16,:]`（16×64）
- RTL：`int8_linear_tiled.sv`（新建）；`int8_weight_loader` 扩展 `DEPTH=1024` + `replay_base_addr`
- 向量：`tiled_head0_q_vectors.txt`（4 case：M=1/4/2/4，含 m_row_start=4）
- 元数据：`tiled_head0_q_export_meta.json`
- RTL：`test_count=4`，`mismatch_count=0`，`PYTHON AND RTL ARE 4/4 EXACT`，约 **261080 ns**
- 生成：`python scripts/generate_linear_tiled_head0_q_vectors.py`（需 `KMP_DUPLICATE_LIB_OK=TRUE` 若遇 OpenMP）
- CLI 回归：`scripts/run_tiled_head0_q_xsim.bat`
- 仿真 Tcl：`scripts/xsim/launch_tiled_head0_q_with_vcd.tcl` → `tiled_head0_q_full.vcd`
- D4/D5 回归（loader 改口后）：D4 24/24 PASS，D5 12/12 PASS

### D5 MiniGPT Wq 结果（seed=20260829）

- Checkpoint：`learn/mini_gpt_ready/mini_gpt_checkpoint.pt`（train 200 steps）
- 量化 golden：`learn/int8_gemm_reference/int8_gemm_reference_v2.py`（shift_bits=24）
- 向量：`gpt_linear_vectors.txt`（12 case，全部真实 Wq 子块）
- 元数据：`gpt_wq_export_meta.json`
- RTL：`test_count=12`，`mismatch_count=0`，`PYTHON AND RTL ARE 12/12 EXACT`，约 **63880 ns**
- 生成：`python scripts/generate_linear_int8_gpt_vectors.py`
- CLI 回归：`scripts/run_linear_int8_gpt_xsim.bat`
- 仿真 Tcl：`scripts/xsim/launch_linear_int8_gpt_with_vcd.tcl` → `gpt_linear_int8_full.vcd`

### D4 Linear Layer 结果（seed=20260828）

- Python 向量 TB：`test_count=24`，`mismatch_count=0`，`PYTHON AND RTL ARE 24/24 EXACT`，约 **59390 ns**
- 向量：`linear_int8_vectors.txt`（A / WEIGHT / MULT / INT8 EXPECT）
- 仿真 Tcl：`scripts/xsim/launch_linear_int8_with_vcd.tcl`（加波 + VCD + run all，无 bat）

### D3 并行 GEMM 结果（seed=20260827）

- 内联 TB：`test_count=24`，`mismatch_count=0`，仿真约 **46950 ns**
- Python 向量 TB：`test_count=24`，`mismatch_count=0`，`PYTHON AND RTL ARE 24/24 EXACT`，约 **48320 ns**
- 定向 case 1：`2×2×3`，`C=[[58,64],[139,154]]` 与串行 GEMM 一致

### Task D 结果（点积，seed=42）

- `test_count=120`，`mismatch_count=0`
- `PYTHON AND RTL ARE 120/120 EXACT`
- 仿真结束约 **43230 ns**

---

## 关键文件

| 文件 | 作用 |
|------|------|
| `scripts/generate_dot_product_parallel_vectors.py` | 生成向量 |
| `dot_product_parallel_vectors.txt` | 120 组 CASE/BEAT |
| `scripts/xsim/tb_dot_product_parallel_python_vectors.tcl` | Task D Launch：加波 + run all |
| `scripts/xsim/apply_waveform_task_d.tcl` | 手动恢复 32 信号波形 |
| `scripts/xsim/export_full_wave_task_d.tcl` | 导完整 VCD |
| `scripts/run_task_d_export_vcd.bat` | 一键编译 + 导 VCD |
| `scripts/run_task_d_xsim.bat` | 一键编译 + 跑 Task D（不导 VCD） |
| `scripts/generate_gemm_parallel_vectors.py` | 生成 D3 GEMM 向量 |
| `gemm_parallel_vectors.txt` | 24 组 CASE/A/B/EXPECT |
| `scripts/run_gemm_parallel_xsim.bat` | 一键生成向量 + 跑 D3 Python 向量 TB |
| `scripts/setup_gemm_parallel_sim.tcl` | 把 xpr sim top 切到 D3 GEMM + 波形/VCD 一体 Launch Tcl |
| `scripts/setup_gemm_parallel_sim_gui.bat` | 一键跑 setup（从 cmd，勿在 xsim 里跑） |
| `scripts/xsim/launch_gemm_parallel_with_vcd.tcl` | **一体脚本**：加波 + 录 VCD + run all |
| `scripts/xsim/tb_gemm_parallel_python_vectors.tcl` | 同上（wrapper） |
| `scripts/xsim/apply_waveform_gemm_parallel.tcl` | 仅加波（不导 VCD） |
| `scripts/xsim/export_full_wave_gemm_parallel.tcl` | 仅导 VCD（batch 用） |
| `scripts/run_gemm_parallel_gui_vcd.bat` | **cmd 推荐**：GUI 波形 + VCD，不改 xpr；`inline` 参数切内联 TB |
| `scripts/run_gemm_parallel_export_vcd.bat` | cmd 无 GUI：仅 VCD + 回归 |
| `scripts/run_gemm_parallel_xsim.bat` | cmd 最快：仅 PASS/FAIL，无波形 |
| `scripts/setup_gemm_parallel_sim_gui.bat` | **可选**：把 Launch 按钮绑到 D3（会覆盖 xpr sim 设置） |
| `scripts/setup_task_d_sim.tcl` | 把 xpr sim top 切到 Task D |
| `scripts/setup_vivado_sim.tcl` | 把 xpr sim top 切回 Task B/C |
| `task_d_full.vcd` | 完整波形（约 384KB，GTKWave 打开） |
| `scripts/verify_task_d_vcd.py` | 解析 VCD，120 case 对拍 golden |
| `scripts/generate_linear_int8_vectors.py` | 生成 D4 Linear 向量（GEMM + requantize golden） |
| `linear_int8_vectors.txt` | 24 组 CASE/A/WEIGHT/MULT/INT8 EXPECT |
| `int8_linear_layer.sv` | D4 顶层：GEMM + weight_loader + requantize |
| `int8_weight_loader.sv` | 权重预加载 + LOAD_B 回放 |
| `scripts/xsim/launch_linear_int8_with_vcd.tcl` | **D4 一体脚本**：加波 + 录 VCD + run all |
| `scripts/xsim/linear_int8_wave_common.tcl` | D4 波形/VCD 信号列表 |
| `scripts/export_mini_gpt_wq_slice.py` | D5：从 MiniGPT checkpoint 导出 Wq slice |
| `scripts/generate_linear_int8_gpt_vectors.py` | 生成 D5 GPT 向量 |
| `gpt_linear_vectors.txt` | 12 组真实 Wq CASE |
| `gpt_wq_export_meta.json` | D5 导出元数据（slice 坐标、scales） |
| `scripts/run_linear_int8_gpt_xsim.bat` | D5 CLI 一键回归 |
| `scripts/xsim/launch_linear_int8_gpt_with_vcd.tcl` | **D5 一体脚本**：加波 + VCD + run all |
| `scripts/xsim/gpt_linear_int8_wave_common.tcl` | D5 波形/VCD 信号列表 |
| `int8_linear_tiled.sv` | **D6/D7** 顶层：16-tile FSM + QKV 多 Head 基地址 |
| `tb_linear_tiled_head0_q.sv` | D6 TB，4 case head0 整层 Q |
| `tiled_head0_q_vectors.txt` | D6 向量（tile-major 1024 WEIGHT + 16 MULT） |
| `scripts/export_mini_gpt_head0_q_full.py` | D6：导出 head0 Wq[16,64] + golden |
| `scripts/generate_linear_tiled_head0_q_vectors.py` | 生成 D6 向量 |
| `scripts/run_tiled_head0_q_xsim.bat` | D6 CLI 一键回归 |
| `scripts/xsim/launch_tiled_head0_q_with_vcd.tcl` | **D6** 一体脚本：加波 + VCD + run all |
| `scripts/xsim/gpt_tiled_head0_q_wave_common.tcl` | D6 波形/VCD 信号列表 |
| `tb_linear_tiled_qkv.sv` | **D7** TB，5 case QKV 多 Head |
| `tiled_qkv_vectors.txt` | D7 向量（PRELOAD 12288+192 + CASE） |
| `scripts/export_mini_gpt_qkv_heads.py` | D7：导出 Q/K/V × 4 head 银行 |
| `scripts/generate_linear_tiled_qkv_vectors.py` | 生成 D7 向量 |
| `scripts/run_tiled_qkv_xsim.bat` | D7 CLI 一键回归 |
| `scripts/xsim/launch_tiled_qkv_with_vcd.tcl` | **D7** 一体脚本 |
| `scripts/xsim/gpt_tiled_qkv_wave_common.tcl` | D7 波形/VCD 信号列表 |

**Cursor 规则：**

- `.cursor/rules/fpga-int8-project-workflow.mdc` — 工程/波形/bat/Python/RTL 经验
- `.cursor/rules/vivado-xsim-waveforms.mdc` — 波形 Tcl 细则

---

## 常用命令

```bat
:: 项目根目录
cd /d F:\Users\22563\Desktop\FPGAlearn\requantize_int8_rtl

:: 重新生成向量
python scripts\generate_dot_product_parallel_vectors.py

:: Task D 仿真（命令行）
scripts\run_task_d_xsim.bat

:: D3 命令行：GUI 波形 + VCD（推荐，不改 xpr）
scripts\kill_xsim_locks.bat
scripts\run_gemm_parallel_gui_vcd.bat
scripts\run_gemm_parallel_gui_vcd.bat inline

:: D3 命令行：仅 VCD / 仅 PASS（无 GUI）
scripts\run_gemm_parallel_export_vcd.bat
scripts\run_gemm_parallel_xsim.bat

:: D3 可选：绑定 Vivado Launch 按钮（会改 xpr，不用也行）
scripts\setup_gemm_parallel_sim_gui.bat

:: 导完整 VCD（先关 Vivado）
scripts\kill_xsim_locks.bat
scripts\run_task_d_export_vcd.bat
```

**Vivado GUI（两种方式，二选一）：**

**A. 不绑定 xpr（推荐）** — cmd 启动，自动开波形窗口 + 写 VCD：

```bat
scripts\run_gemm_parallel_gui_vcd.bat
```

**B. xsim Tcl 手动（不绑 custom_tcl，仿真已 Launch 时）：**

```tcl
restart
source F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/scripts/xsim/launch_gemm_parallel_with_vcd.tcl
```

脚本内含 `run all` + `close_vcd`。成功标准：`INFO: size=224705 bytes` 量级（不是 138）。

**D6 xsim Tcl（sim Top = `tb_linear_tiled_head0_q`）：**

```tcl
restart
source F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/scripts/xsim/launch_tiled_head0_q_with_vcd.tcl
```

向量：`python scripts/generate_linear_tiled_head0_q_vectors.py`  
CLI：`scripts/run_tiled_head0_q_xsim.bat`

**D5 xsim Tcl（sim Top = `tb_linear_int8_gpt_vectors`）：**

```tcl
restart
source F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/scripts/xsim/launch_linear_int8_gpt_with_vcd.tcl
```

向量：`python scripts/generate_linear_int8_gpt_vectors.py`  
CLI：`scripts/run_linear_int8_gpt_xsim.bat`

**D4 xsim Tcl（sim Top = `tb_linear_int8_python_vectors`）：**

```tcl
restart
source F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/scripts/xsim/launch_linear_int8_with_vcd.tcl
```

向量：`python scripts/generate_linear_int8_vectors.py`

**C. 绑定 Launch 按钮（可选）** — 跑 `setup_gemm_parallel_sim_gui.bat` 后，在 GUI 点 Launch。

不要在 `$finish` 后再 source 导 VCD；不要在 xsim Tcl 里跑 `vivado -mode batch`。

**GUI 内导 VCD（仿真已 Launch）：**

```tcl
source F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/scripts/xsim/export_full_wave_task_d.tcl
```

---

## 已知坑（已踩过）

1. **Top 与 Tcl 不一致** → 报 `tb_int8_dot_product_parallel/clk` 找不到；检查 `simulate.bat` 和 `xpr` 里 `custom_tcl`
2. **bat 缺 `call xvlog`** → 只跑到 `[1/3]` 就停
3. **Vivado 开着时 CLI xelab** → `xsim.type for writing` 锁文件
4. **站在 `scripts\` 里跑 `scripts\xxx.bat`** → 路径双 `scripts`
5. **WDB 不能事后转 VCD**；须仿真中 `open_vcd`/`log_vcd`
6. **VCD 验收**：`python scripts\verify_task_d_vcd.py`；功能看 TB PASS + VCD 120/120

---

## 下一步

- [x] 综合 `int8_dot_product_parallel`（standalone Top，post-synthesis）
- [x] 100 MHz 约束（沿用现有 xdc）
- [x] 进入并行 GEMM Controller（D3）
- [x] D4：Requantization 接入 + 权重加载器
- [x] D5：mini GPT Wq 真实权重 slice 对拍（12/12 PASS）
- [x] D6：Full Tile head0 Q（4/4 PASS）
- [x] D7：Q/K/V 多 Head 基地址（5/5 PASS）
- [x] D8：Attention Score `Q @ K^T`（8/8 PASS）
- [x] D9：Causal Softmax LUT（6/6 PASS）
- [x] D10：Attn@V + Wo tiled（PASS）
- [x] D11：Residual + SwiGLU FFN 分层（PASS）
- [x] D12：Transformer Block E2E seq=1（1/1 EXACT）
- [x] ZedBoard：JTAG 烧录成功 + **LD0–3 闪烁已确认**（2026-09-03，`xc7z020_1`，Digilent/210248493052）
- [ ] 学习：吃透 D6 tile 切分 → D7 基地址
- [ ] 综合 `int8_linear_layer` / `int8_linear_tiled` / block
- [ ] （稍后）PS + AXI 封装 INT8；完整 Block **尚未上板**
- [ ] E2E 扩到 seq=4（可选）

---

## 聊天恢复提示（复制给新窗口 AI）

```
请先读项目根目录 PROGRESS.md 与 NOTES.md。
D12 已关账：int8_transformer_block E2E seq=1，seed=20260908，1/1 EXACT。
CLI：scripts/run_transformer_block_xsim.bat；向量=transformer_block_vectors.txt。
D4–D11 分层回归仍 PASS。
板级：ZedBoard blink 冒烟 **PASS**（烧录 + LD0–3 闪，2026-09-03）。
学习：D2–D4 已懂，D6 tiling 进行中。下一步：学完 D6 切分再 D7；工程稍后 AXI/综合。
```
