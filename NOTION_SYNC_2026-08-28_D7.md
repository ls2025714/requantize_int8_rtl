# D7 Q/K/V 多 Head 地址关账｜2026-08-28

> 同步到 Notion：
> - [单层 Transformer Block INT8 仿真计划｜完整版 v2](https://app.notion.com/p/ee490988723b4b169c9c3788c9ec559c)
> - [Cursor 代码学习计划书｜FPGA INT8 加速主线](https://app.notion.com/p/3c88e1b1de6f809fbd4adba7c19d34bc)
> - [7天仿真冲刺｜单层 Transformer Block INT8](https://app.notion.com/p/3c88e1b1de6f81059581f89001280a53)

---

## 仿真关账

| 阶段 | 状态 | 证据 |
|------|------|------|
| D6 Full Tile head0 Q | PASS | 4/4 exact（回归仍过） |
| **D7 Q/K/V 多 Head** | **PASS** | seed=20260831，**5/5 exact**，~524300 ns |
| D5 MiniGPT Wq | PASS | 12/12（回归） |
| D4 Linear + Requant | PASS | 24/24（回归） |

### D7 要点

- **功能：** `cmd_op_type` + `cmd_head_idx` 切换权重/MULT 基地址；一次 cmd 仍出 `[M,16]` INT8
- **映射：** `op_base={0,4096,8192}`，`head_stride=1024`，`WEIGHT_DEPTH=12288`；`mult_base=op*64+head*16`，`MULT_DEPTH=192`
- **Case：** head0 WQ/WK/WV（M=4）；WQ head1（M=4）；WV head3（M=2）— 验证非零 head 与跨 op 无串读
- **RTL：** 扩展 `int8_linear_tiled.sv`；`int8_linear_layer` **未改**
- **向量：** `tiled_qkv_vectors.txt`（一次 PRELOAD + 多 CASE）
- **CLI：** `scripts/run_tiled_qkv_xsim.bat`

### 下一步

- D8：Attention Score `Q @ K^T`（先 seq=4）
- Softmax / @V / FFN / Block Top 仍后置
- 综合 / 上板仍后置

## 关键文件

- `int8_linear_tiled.sv`（`cmd_op_type` / `cmd_head_idx`）
- `tb_linear_tiled_qkv.sv`
- `scripts/export_mini_gpt_qkv_heads.py`
- `scripts/generate_linear_tiled_qkv_vectors.py`
- `scripts/run_tiled_qkv_xsim.bat`
- `scripts/xsim/launch_tiled_qkv_with_vcd.tcl`
- `PROGRESS.md` / `NOTES.md` / `FILES.md`
