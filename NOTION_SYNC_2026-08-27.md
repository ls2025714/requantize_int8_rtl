# 并行 Dot Product 关账｜仿真 + §5 综合｜2026-08-27

> 粘贴到 Notion：[Cursor 代码学习计划书｜FPGA INT8 加速主线](https://app.notion.com/p/3c88e1b1de6f809fbd4adba7c19d34bc) 顶部 callout，或 [05｜FPGA 加速器实现](https://app.notion.com/p/20ed6b58160f41d486d3f2c1cfcc71ae) 新增一节。

---

## 仿真关账

| 阶段 | 状态 | 证据 |
|------|------|------|
| Task B | PASS | `test_count=14`，`mismatch_count=0`，`PARTIAL_LATENCY=2`，`ACC_LATENCY=3` |
| Task C | PASS | `test_count=36`，`mismatch_count=0` |
| Task D | PASS | `test_count=120`，`mismatch_count=0`，`PYTHON AND RTL ARE 120/120 EXACT`，43230 ns |

- 向量：`dot_product_parallel_vectors.txt`（seed=42）
- VCD：`task_d_full.vcd`（393771 bytes）
- VCD 验收：`python scripts/verify_task_d_vcd.py` → 120/120 `m_result` 与 Python golden 一致

---

## §5 综合（并行 standalone Top）

**器件：** `xc7z020clg484-1`  
**Top：** `int8_dot_product_parallel`  
**约束：** `create_clock -period 10.000 [get_ports clk]`（100 MHz，文件 `int8_gemm_serial_100mhz.xdc`）  
**状态：** Post-synthesis 完成；Post-route 未做

### 资源（Post-Synthesis）

| 资源 | 并行 standalone | 串行（GEMM 层次，历史记录） | 变化 |
|------|-----------------|----------------------------|------|
| LUT | **388** | 158 | 约 2.46× |
| FF | **212** | 99 | 约 2.14× |
| DSP | **0** | — | 均未用 DSP48 |
| BRAM | 0 | — | — |

> IOB 171 为顶层端口自动 IBUF/OBUF，片内对比时忽略。

### 时序（Post-Synthesis）

| 指标 | 并行 standalone |
|------|-----------------|
| WNS | **+4.619 ns** |
| TNS | 0 |
| Setup 失败端点 | 0 |
| WHS | **+0.137 ns** |
| THS | 0 |
| Hold 失败端点 | 0 |
| WPWS | +4.500 ns |

**结论：** 100 MHz post-synthesis 通过，Setup/Hold 均无失败端点。

**Hold 最差路径示例：** `valid_s3_reg/C → valid_s4_reg/D`；`beat_count_reg` / `acc_count_reg` 相关 1 级逻辑路径，Total Delay ≈ 0.38 ns。

### 吞吐（理论）

| 指标 | 串行 | 4 路并行 |
|------|------|----------|
| K=16 输入拍数 | 16 | 4 |
| 每 beat 有效 MAC | 1 | 4 |
| 流水线 | 2 级 MAC | S1→S2→S3→S4 + DRAIN |

---

## 待补

- [ ] `dot_product_int8` 单独 Top 综合（同 Part、同 xdc，公平对比）
- [ ] 并行版 Run Implementation（post-route WNS，可选）
- [ ] 进入并行 GEMM Controller（D3）

## 修改文件

- `int8_dot_product_parallel.sv`
- `tb_int8_dot_product_parallel.sv`
- `tb_dot_product_parallel_python_vectors.sv`
- `scripts/generate_dot_product_parallel_vectors.py`
- `scripts/verify_task_d_vcd.py`
