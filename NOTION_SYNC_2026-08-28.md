# D6 Full Tile head0 Q 关账｜2026-08-28

> 同步到 Notion：
> - [Cursor 代码学习计划书｜FPGA INT8 加速主线](https://app.notion.com/p/3c88e1b1de6f809fbd4adba7c19d34bc)
> - [7天仿真冲刺｜单层 Transformer Block INT8](https://app.notion.com/p/3c88e1b1de6f81059581f89001280a53)
> - [05｜FPGA 加速器实现](https://app.notion.com/p/20ed6b58160f41d486d3f2c1cfcc71ae)

---

## 仿真关账（工程里程碑 D3→D6）

| 阶段 | 状态 | 证据 |
|------|------|------|
| D3 并行 GEMM | PASS | Python 向量 24/24 exact |
| D4 Linear + Requant | PASS | seed=20260828，24/24 exact，~59390 ns |
| D5 MiniGPT Wq 子块 | PASS | seed=20260829，12/12 exact，~63880 ns |
| **D6 Full Tile head0 Q** | **PASS** | seed=20260830，**4/4 exact**，~261080 ns |

### D6 要点

- **功能：** 一次 cmd 算完 head0 Q：`[M,64] × Wq[16,64]^T → [M,16]` INT8（M≤4）
- **RTL：** 新建 `int8_linear_tiled.sv`；**未改** `int8_linear_layer`（D4/D5 回归仍 PASS）
- **Tile：** 16 次子 GEMM（4×K-tile × 4×N-tile）；K 维 INT32 累加后再 requantize
- **loader：** `DEPTH=1024` + `replay_base_addr`
- **向量：** `tiled_head0_q_vectors.txt`；VCD：`tiled_head0_q_full.vcd`（1221278 bytes）
- **VCD 验收：** `tile_idx=0..15`，`k_base={0,16,32,48}`，`n_base={0,4,8,12}`，`c_valid` 上升沿=176，`mismatch_count=0`

### 回归

- D5：12/12 PASS（loader 改口后）
- D4：24/24 PASS

---

## 下一步

- D7 候选：Wk / Wv / FFN 整层，或多 head 权重基地址表
- Attention datapath（原冲刺表「D6」）仍后置
- 综合 / Implementation / 上板仍后置

## 关键文件

- `int8_linear_tiled.sv`
- `int8_weight_loader.sv`（`replay_base_addr`）
- `tb_linear_tiled_head0_q.sv`
- `scripts/export_mini_gpt_head0_q_full.py`
- `scripts/generate_linear_tiled_head0_q_vectors.py`
- `scripts/run_tiled_head0_q_xsim.bat`
- `scripts/xsim/launch_tiled_head0_q_with_vcd.tcl`
- `scripts/verify_tiled_head0_q_vcd.py`
- `PROGRESS.md` / `NOTES.md` / `FILES.md`
