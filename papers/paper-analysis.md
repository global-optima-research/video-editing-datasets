# 论文分析索引

针对 PVTT (Product Video Template Transfer) 数据集构建的论文分析汇总。

---

## 论文列表

### 数据集构建

| 论文 | 核心贡献 | 文档 |
|------|----------|------|
| **Ditto-1M** | 12K GPU-days 构建 1M 数据集，完整 Pipeline | [ditto-1m-analysis.md](ditto-1m-analysis.md) |

### 运动保持技术

| 论文 | 核心贡献 | 文档 |
|------|----------|------|
| **I2VEdit** | Motion LoRA + 注意力匹配 | [i2vedit-analysis.md](i2vedit-analysis.md) |
| **Motion LoRA** | Temporal Attention 微调详解 | [motion-lora.md](motion-lora.md) |

### 物体替换技术

| 论文 | 核心贡献 | 文档 |
|------|----------|------|
| **VideoSwap** | 语义点对应，支持形状变化 | [videoswap-analysis.md](videoswap-analysis.md) |
| **VideoAnyDoor** | 零样本插入，Box + 关键点控制 | [videoanydoor-analysis.md](videoanydoor-analysis.md) |

### Video Object Insertion

| 论文 | 核心贡献 | 文档 |
|------|----------|------|
| **综合调研** | VideoAnyDoor, Anything in Any Scene 等 | [video-object-insertion.md](video-object-insertion.md) |

---

## 快速对比

### 运动保持方案

| 方案 | 形状变化 | 自动化 | 成本 |
|------|----------|--------|------|
| 深度约束 (Ditto) | ❌ | ✅ 高 | ✅ 低 |
| Motion LoRA (I2VEdit) | ⚠️ 有限 | ⚠️ 中 | ❌ 高 (25min/视频) |
| 语义点 (VideoSwap) | ✅ | ❌ 低 | ⚠️ 中 |
| Box 序列 (VideoAnyDoor) | ✅ | ✅ 高 | ✅ 低 |

### 几何约束分析

| 方案 | 几何变化 | 原因 |
|------|----------|------|
| 深度约束 | ❌ 不支持 | 深度作为硬约束 |
| Motion LoRA | ⚠️ 有限 | 注意力倾向保持结构 |
| 语义点 | ✅ 支持 | 稀疏点不约束整体形状 |
| Box 序列 | ✅ 支持 | 只约束位置，不约束形状 |

---

## PVTT 推荐方案

基于分析，**VideoAnyDoor** 最适合 PVTT 数据集构建：

```
优势:
├── ✅ 零样本 (不需要每视频训练)
├── ✅ 形状自由 (手表→项链 可行)
├── ✅ Box 序列控制 (易自动化)
└── ✅ 已验证 virtual try-on 场景

Pipeline:
模板视频 → SAM2 分割 → Mask 序列
    ↓              ↓
    └── Inpainting → 干净背景
                        ↓
新商品图 + Mask 序列 + 背景 → VideoAnyDoor → 合成视频
```

---

## 文档结构

```
papers/
├── paper-analysis.md          # 本文件 (索引)
├── ditto-1m-analysis.md       # Ditto-1M 数据集构建
├── i2vedit-analysis.md        # I2VEdit 运动保持
├── motion-lora.md             # Motion LoRA 技术详解
├── videoswap-analysis.md      # VideoSwap 语义点替换
├── videoanydoor-analysis.md   # VideoAnyDoor 物体插入
└── video-object-insertion.md  # Video Object Insertion 综合调研
```

---

Last updated: 2026-01-20
