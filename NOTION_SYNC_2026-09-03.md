# Notion 进度同步稿｜2026-09-03

> **状态：已写入 Notion｜2026-09-03**

目标页：
1. [单层 Transformer Block INT8 仿真计划](https://app.notion.com/p/ee490988723b4b169c9c3788c9ec559c) — `ee490988-723b-4b16-9c9c-3788c9ec559c`
2. [7天仿真冲刺](https://app.notion.com/p/3c88e1b1de6f81059581f89001280a53) — `3c88e1b1-de6f-8105-9581-f89001280a53`
3. [Cursor 代码学习计划书](https://app.notion.com/p/3c88e1b1de6f809fbd4adba7c19d34bc) — `3c88e1b1-de6f-809f-bd4a-dba7c19d34bc`

---

## 事实摘要（勿再扩写）

**仿真仍关账：** D3–D12 PASS。D12 E2E seq=1，seed=20260908，1/1 EXACT，~2716160 ns。CLI：`scripts/run_transformer_block_xsim.bat`。INT8 E2E **仅仿真**。

**学习（owner 反向学习，进行中 2026-09-02/03）：**
- Day 0 完成：CLI PASS + Vivado GUI 看 `dut/st`（PRE 最长；compute 最长 GATE/UP/DOWN）
- D2：位宽 16→17→18→32、DRAIN、`s_keep`（K=4/5/7）已懂
- D3：GEMM = 点积调度器；M=2,N=3,K=7 → 6 个 C、2 beat、keep=0111
- D4：Linear = GEMM + weight_loader + requantize（INT32→INT8，shift 24，移位后饱和）
- D6 tiling 概念：4 n_tiles × 4 k_tiles；跨 k_tile 用 INT32 `acc_mem`，每 n_tile 一次 requantize。切分细节仍部分理解中
- D7+ Attention/Block 学习所有权仍待完成（工程侧已关账）

**板级冒烟（2026-09-02 起；尚非 Transformer 上板）：**
- 器件 `xc7z020clg484-1` 与 ZedBoard 匹配
- `zedboard_pl_blink.sv` + `zedboard_pl_blink.xdc`（clk Y9，LD0–3）
- bitstream 已生成：`build/zedboard_pl_blink/zedboard_pl_blink.bit`
- 构建：`scripts/build_zedboard_blink.bat`
- 烧录（用户执行）：`scripts/program_zedboard_blink.bat` — **JTAG 烧录 / LED 闪烁尚未由用户确认**
- 下一步：确认 LD0–3 闪 → 再谈 PS+AXI INT8；完整 Block **尚未上板**

**当前下一步：**
1. ZedBoard：烧录 blink bitstream，确认 LED
2. 学习：完成 D6 tile 切分 → 再 D7 基地址
3. 工程稍后：AXI wrap + `linear_tiled`/block 综合

---

## Page 1｜仿真计划：页顶 prepend callout

```
<callout icon="📍" color="blue_bg">
	**进度同步｜2026-09-03**
	- **仿真：**仍关账。D3～D12 PASS；D12 E2E seq=1 `1/1 EXACT`（seed=`20260908`，~2716160 ns）。CLI：`scripts/run_transformer_block_xsim.bat`。INT8 仍 **仅仿真**。
	- **学习：**Day 0 完成；D2/D3/D4 已懂；D6 tiling 概念部分理解（切分仍问）；D7+ Attn/Block 学习所有权待完成。
	- **板级：**ZedBoard 冒烟已开工（非 Transformer 上板）。`zedboard_pl_blink` bitstream 已生成；`scripts/program_zedboard_blink.bat` **烧录/LED 未确认**。
	- **当前协同：**① 烧录确认 LD0–3 闪；② 学完 D6 切分再 D7 基地址；③ 工程稍后 AXI + 综合 linear_tiled/block。
</callout>
```

---

## Page 2｜7天冲刺：替换顶部「进度同步」callout +「最新进度」

**进度同步 callout（替换 2026-08-28 D12 版）：**

```
<callout icon="📍" color="blue_bg">
	**进度同步｜2026-09-03**
	**仿真：**D3～D12 仍关账；D12 Block E2E seq=1：`1/1 EXACT`（seed=`20260908`，~2716160 ns）。CLI：`scripts/run_transformer_block_xsim.bat`。
	**学习：**owner 反向学习进行中——Day 0 完成；D2 位宽/DRAIN/s_keep；D3 GEMM 调度；D4 Linear+requant；D6 tiling 概念部分懂；D7+ 学习待完成。
	**板级：**冒烟 bitstream 已生成（`build/zedboard_pl_blink/zedboard_pl_blink.bit`）；JTAG 烧录/LED **未确认**。完整 INT8 Block **未上板**。
	**当前协同任务：**① ZedBoard 烧录确认 LED；② 学习收 D6 切分→D7 基地址；③ 工程稍后 AXI wrap + 综合。
</callout>
```

**最新进度（替换 2026-08-28 段）：**

```
## 最新进度（2026-09-03）
**当前Task：**仿真已关账；主线切到 **反向学习 + ZedBoard 冒烟**。
**仿真证据：**D3～D12 PASS；D12 `1/1 EXACT`（seed=`20260908`，~2716160 ns）。
**学习证据：**Day 0（CLI+GUI `dut/st`）；D2～D4 理解到位；D6 tiling 概念部分理解；D7+ 学习所有权待完成。
**板级证据：**器件匹配 ZedBoard；blink RTL/xdc/bitstream 就绪；烧录脚本待用户确认 LED。
**下一步第一个动作：**`scripts/program_zedboard_blink.bat` 确认 LD0–3；同时学完 D6 tile 切分。
```

---

## Page 3｜学习计划书：更新「当前学习入口」callout

```
<callout icon="🚀" color="orange_bg">
	**当前学习入口｜2026-09-03**
	- **已完成：**Day 0（CLI PASS + GUI `dut/st`）；D2 位宽/DRAIN/`s_keep`；D3 GEMM 调度（M=2,N=3,K=7）；D4 Linear+requant（shift 24）。
	- **进行中：**D6 tiling（4×4 tiles、`acc_mem` 跨 k_tile、每 n_tile 一次 requant）——切分细节仍需吃透。
	- **待学：**D7 基地址 → Attention/Block 所有权（工程侧 D7–D12 已关账）。
	- **并行板级：**blink bitstream 已生成；先确认烧录 LED，再谈 PS+AXI。INT8 E2E 仍 sim-only。
</callout>
```
