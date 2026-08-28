# D12 Transformer Block E2E 关账｜2026-08-28

> 同步到 Notion：
> - [单层 Transformer Block INT8 仿真计划｜完整版 v2](https://app.notion.com/p/ee490988723b4b169c9c3788c9ec559c)
> - [Cursor 代码学习计划书｜FPGA INT8 加速主线](https://app.notion.com/p/3c88e1b1de6f809fbd4adba7c19d34bc)
> - [7天仿真冲刺｜单层 Transformer Block INT8](https://app.notion.com/p/3c88e1b1de6f81059581f89001280a53)

---

## 仿真关账

| 阶段 | 状态 | 证据 |
|------|------|------|
| D8 Score GEMM | PASS | seed=20260903，8/8 |
| D9 Softmax causal | PASS | seed=20260902，6/6 |
| D10 Attn@V + Wo | PASS | Attn 5/5；Wo 5/5 |
| D11 Residual + FFN | PASS | Res 5/5；SiLU 3/3；FFN layered 3/3 |
| **D12 Block E2E** | **PASS** | seed=**20260908**，**1/1 EXACT**，~2716160 ns |
| 回归 D4/D6/D7 | PASS | 24/24、4/4、5/5 |

### D12 要点

- **功能：** 单层 MiniGPT Transformer Block INT8 端到端（**seq=1**）
- **规格：** `n_embd=64`，`n_head=4`，`head_dim=16`，`d_ff=256`
- **RTL：** `int8_transformer_block.sv` 编排已关账子模块
- **向量：** `transformer_block_vectors.txt`（PRELOAD ~66k + 1 CASE）
- **Golden：** EXPECT 必须用 **同一套 frozen weight/mult bank** 定点推演（与 preload 一致）；elem_mul 舍入对齐 RTL（`>=0:+64` / `<0:+63`）
- **CLI：** `scripts/run_transformer_block_xsim.bat`
- **Debug：** `scripts/run_transformer_block_debug.bat`（`TRANSFORMER_BLOCK_DBG`）

### 关键踩坑（已修）

1. feed 侧 data/地址与 `*_valid` 同拍 → **组合驱动**
2. `l_cmd_v` 仅在 `LS_CMD` 拉高，避免 linear 重复吃 cmd
3. Softmax / Attn 地址晚一拍 → seq=1 死锁
4. SiLU 1-deep：FEED 时保持 `out_ready` 策略改为 **一进一出**
5. OUTPUT 握手同拍更新 data，消除整体错位
6. residual `z_ready` 不依赖 `!z_valid`（避免 EMIT 死锁）

### 下一步

- 综合 `int8_linear_tiled` / `int8_transformer_block`
- （可选）E2E 扩到 seq=4
- 上板仍后置

## 关键文件

- `int8_transformer_block.sv`
- `tb_int8_transformer_block.sv`
- `scripts/generate_transformer_block_vectors.py`
- `scripts/run_transformer_block_xsim.bat`
- `transformer_block_vectors.txt`
- `PROGRESS.md` / `NOTES.md` / `FILES.md`
