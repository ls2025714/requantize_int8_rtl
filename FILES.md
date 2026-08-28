# 项目文件索引（requantize_int8_rtl）

> 打开项目先看本文 + `PROGRESS.md`。  
> **你感觉「文件很多」，多半是因为 Vivado 自动生成了大量目录**（`.sim/`、`.runs/`、`.cache/`），那些不用管，也不要手改。

---

## 1. 目录分层（什么该看、什么可忽略）

| 目录 | 谁生成的 | 你要不要管 |
|------|----------|------------|
| `requantize_int8_rtl.srcs/sources_1/new/` | **你写的 RTL** | ✅ 综合用 |
| `requantize_int8_rtl.srcs/sim_1/new/` | **你写的 TB** | ✅ 仿真用 |
| `scripts/` | **你写的脚本** | ✅ Python / Tcl / bat |
| 仓库根目录 `*_vectors.txt` | Python 生成 | ✅ 对拍数据 |
| `requantize_int8_rtl.sim/` | Vivado 仿真产物 | ❌ 勿当源码改 |
| `requantize_int8_rtl.runs/` | 综合/实现产物 | ❌ 勿提交 git |
| `requantize_int8_rtl.cache/` | Vivado 缓存 | ❌ 勿提交 git |
| `*.vcd` / `*.wdb` | 波形录制 | 可选保留，可 .gitignore |

**整理建议（不必一次做完）：**

1. **`.gitignore` 加强**：忽略 `.sim/`、`.runs/`、`.cache/`、`*.wdb`、`__pycache__/`
2. **向量文件归拢**（可选）：建 `vectors/` 目录，把 `*_vectors.txt` 移进去，改 TB 里 `VECTOR_FILE` 路径
3. **错放的 TB 归位**：`sources_1/new/tb_requantize_*.sv` 应在 `sim_1/new/`（历史遗留，不影响仿真）
4. **脚本只记 3 个入口**：见下文「日常只记这些」

---

## 2. RTL 模块（`sources_1/new/`）

按数据流从底到顶：

| 文件 | 阶段 | 一句话 |
|------|------|--------|
| `int8_mac.sv` | 基础 | 单 MAC：activation × weight → 累加 |
| `int8_mac_pipeline.sv` | 基础 | 带 1 级流水线的 MAC |
| `dot_product_int8.sv` | 串行 | 串行点积（每拍 1 对 a,b） |
| `int8_dot_product_parallel.sv` | D1/D2 | **4-lane 并行点积**，Task B/C/D 核心 |
| `int8_gemm_serial.sv` | 串行 | 串行 GEMM 控制器（M×N×K，INT32 输出） |
| `int8_gemm_parallel.sv` | D3 | **并行 GEMM**，实例化 4-lane 点积，INT32 输出 |
| `requantize_int8.sv` | 量化 | 组合逻辑：INT32 acc × scale → INT8 |
| `requantize_int8_pipeline.sv` | 量化 | 2 拍流水线 requantize |
| `int8_weight_loader.sv` | D4 | 权重 RAM：预加载 B 矩阵，LOAD_B 时回放 |
| `int8_linear_layer.sv` | D4 | **Linear 顶层** = GEMM + loader + requantize → INT8 |
| `int8_linear_tiled.sv` | D6/D7 | **Tiled Linear**：16×(M×4×16) + QKV 多 Head 基地址 → INT8 |
| `int8_score_gemm.sv` | D8 | Score `Q@K^T` → INT32 |
| `int8_softmax_causal.sv` | D9 | Causal Softmax LUT → UQ1.15 |
| `int8_attn_v.sv` | D10 | `P@V` → INT8 |
| `int8_residual_add.sv` | D11 | 饱和 INT8 残差加 |
| `int8_silu_lut.sv` | D11 | SiLU 256-entry LUT |
| `int8_elem_mul.sv` | D11 | 逐元素 INT8 乘 `>>7` |
| `int8_transformer_block.sv` | D12 | **Block Top FSM** E2E |

**约束：** `constrs_1/new/int8_gemm_serial_100mhz.xdc` — 100 MHz 时钟

---

## 3. Testbench（`sim_1/new/` + 错放 2 个在 sources）

| 文件 | 测谁 | 说明 |
|------|------|------|
| `tb_int8_mac.sv` | `int8_mac` | 基础 MAC 定向 |
| `tb_int8_mac_pipeline.sv` | `int8_mac_pipeline` | 文件驱动 MAC 控制 |
| `tb_dot_product_int8.sv` | `dot_product_int8` | 串行点积 |
| `tb_int8_dot_product_parallel.sv` | 并行点积 | Task B/C 内联 case |
| `tb_dot_product_parallel_python_vectors.sv` | 并行点积 | **Task D**，120 case 向量 |
| `tb_int8_gemm_serial.sv` | 串行 GEMM | 24 case |
| `tb_int8_gemm_parallel.sv` | 并行 GEMM | D3 内联 24 case |
| `tb_gemm_parallel_python_vectors.sv` | 并行 GEMM | **D3**，24 case 向量 |
| `tb_requantize_int8_pipeline.sv` | requantize 流水线 | 12 case + 延迟检查 |
| `tb_linear_int8_python_vectors.sv` | Linear 层 | **D4**，24 case，INT8 期望 |
| `tb_linear_int8_gpt_vectors.sv` | Linear 层 | **D5**，12 case，MiniGPT Wq 真实权重 |
| `tb_linear_tiled_head0_q.sv` | Tiled Linear | **D6**，4 case，head0 整层 Q (N=16 K=64) |
| `tb_linear_tiled_qkv.sv` | Tiled Linear | **D7**，5 case，Q/K/V 多 Head 基地址 |
| `tb_int8_score_gemm.sv` | Score | **D8** |
| `tb_int8_softmax_causal.sv` | Softmax | **D9** |
| `tb_int8_attn_v.sv` / `tb_linear_tiled_wo.sv` | Attn / Wo | **D10** |
| `tb_int8_residual_add.sv` / silu / ffn | FFN 分层 | **D11** |
| `tb_int8_transformer_block.sv` | Block Top | **D12** E2E seq=1 |

---

## 4. 向量文件（仓库根目录）

| 文件 | 生成脚本 | 用途 |
|------|----------|------|
| `dot_product_parallel_vectors.txt` | `generate_dot_product_parallel_vectors.py` | Task D 点积 120 case |
| `gemm_parallel_vectors.txt` | `generate_gemm_parallel_vectors.py` | D3 GEMM 24 case，INT32 期望 |
| `linear_int8_vectors.txt` | `generate_linear_int8_vectors.py` | D4 Linear 24 case，INT8 期望 |
| `gpt_linear_vectors.txt` | `generate_linear_int8_gpt_vectors.py` | D5 MiniGPT Wq 12 case，INT8 期望 |
| `tiled_head0_q_vectors.txt` | `generate_linear_tiled_head0_q_vectors.py` | D6 head0 整层 4 case，tile-major WEIGHT |
| `tiled_qkv_vectors.txt` | `generate_linear_tiled_qkv_vectors.py` | D7 QKV 多 Head：PRELOAD + 5 CASE |
| `transformer_block_vectors.txt` | `generate_transformer_block_vectors.py` | **D12** E2E PRELOAD + 1 CASE |
| `mac_control_vectors.txt` | （外部） | MAC pipeline TB |

---

## 5. 脚本（`scripts/`）

### Python

| 文件 | 作用 |
|------|------|
| `generate_dot_product_parallel_vectors.py` | 生成 Task D 点积向量 |
| `generate_gemm_parallel_vectors.py` | 生成 D3 GEMM 向量 |
| `generate_linear_int8_vectors.py` | 生成 D4 Linear 向量 |
| `export_mini_gpt_wq_slice.py` | D5：MiniGPT Wq slice 导出 |
| `generate_linear_int8_gpt_vectors.py` | 生成 D5 GPT 向量 |
| `export_mini_gpt_head0_q_full.py` | D6：MiniGPT head0 Wq[16,64] 全层导出 |
| `generate_linear_tiled_head0_q_vectors.py` | 生成 D6 tiled 向量 |
| `export_mini_gpt_qkv_heads.py` | D7：Q/K/V × 4 head 银行导出 |
| `generate_linear_tiled_qkv_vectors.py` | 生成 D7 QKV 向量 |
| `generate_transformer_block_vectors.py` | **D12** E2E golden（frozen bank） |
| `verify_task_d_vcd.py` | VCD 与 Task D golden 对拍 |
| `verify_gemm_parallel_vcd.py` | VCD 与 D3 golden 对拍 |

### bat（可选，命令行一键跑）

| 文件 | 作用 |
|------|------|
| `run_transformer_block_xsim.bat` | **D12** E2E 回归 |
| `run_transformer_block_debug.bat` | D12 调试（DBG + 短 VCD） |
| `run_*_xsim.bat` | 各阶段无 GUI 快速回归 |
| `kill_xsim_locks.bat` | Vivado 锁文件时杀 xsim |

---

## 6. 日常只记这些（当前主线 D12）

```text
RTL:   int8_transformer_block.sv
TB:    tb_int8_transformer_block.sv
向量:  transformer_block_vectors.txt
生成:  python scripts/generate_transformer_block_vectors.py
CLI:   scripts/run_transformer_block_xsim.bat
进度:  PROGRESS.md / NOTES.md
```

D7 回归：`scripts/run_tiled_qkv_xsim.bat`  
D6 回归：`scripts/run_tiled_head0_q_xsim.bat`  
D4 回归：`scripts/run_linear_int8_xsim.bat`

---

## 7. 演进关系（一张图）

```text
int8_mac → dot_product_int8 ──→ int8_gemm_serial
              ↓
     int8_dot_product_parallel ──→ int8_gemm_parallel
                                        ↓
                              int8_linear_layer (+ weight_loader + requantize)  ← D4/D5
                                        ↓
                              int8_linear_tiled (16-tile + INT32 acc)          ← D6
```

---

## 8. 文档

| 文件 | 作用 |
|------|------|
| `PROGRESS.md` | 阶段进度、关账结果、常用命令 |
| `FILES.md` | **本文件**：文件地图 |
| `.cursor/rules/*.mdc` | Cursor AI 工程规范 |

---

## 9. 源码内注释

每个 `.sv` 文件开头有**文件头**（阶段/作用/验证 TB），RTL 内还有**分段注释**（FSM、流水线、握手等）。  
打开任意 `sources_1/new/*.sv` 或 `sim_1/new/tb_*.sv` 即可阅读。
