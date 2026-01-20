# Video Editing Dataset Construction Analysis

针对 PVTT (Product Video Template Transfer) 训练数据集构建的调研分析。

## 核心构建流程

```
源视频采集 → 预处理(分割/标注) → 编辑生成 → 质量过滤 → 数据集
```

---

## 两种主流范式

### 范式 1: Image Edit + I2V 传播 (主流)

**代表**: InsViE-1M, Ditto-1M, Señorita-2M

```
源视频 → 提取首帧 → 图像编辑器编辑首帧 → I2V模型传播到全视频 → 过滤
```

| 优点 | 缺点 |
|------|------|
| 利用成熟的图像编辑能力 | 依赖 I2V 模型的时序一致性 |
| 编辑质量高、可控性强 | 计算成本高 |
| 适合精细的局部编辑 | Pipeline 较长 |

**成本参考**: Ditto-1M 花费 12,000 GPU-days

### 范式 2: 端到端视频编辑生成

**代表**: OpenVE-3M, ReCo-Data

```
源视频 → 分割/深度图 → VLLM生成指令 → 视频编辑模型(VACE等) → 过滤
```

| 优点 | 缺点 |
|------|------|
| 流程更直接 | 视频编辑模型能力有限 |
| 适合特定任务 | 编辑精度不如图像编辑 |
| 可扩展性好 | 需要强大的视频编辑模型 |

**成本参考**: ReCo-Data 花费 ~76,800 GPU小时 + $13,600 API费用

---

## 各数据集构建方法详解

### VIVID-10M (9.7M samples)

**Pipeline**:
1. Entity Selection: SAM2 + Grounding DINO + RAM
2. Mask Propagation: RAFT 光流 + SAM2 时序一致性
3. Local Caption Generation: InternVL2

**特点**: 混合 Image-Video 数据，降低计算成本

**过滤**: 用户研究评估 (Mask质量、传播准确性、文本对齐)

---

### OpenVE-3M (3M+ triplets)

**Pipeline**:
1. Preprocessing: 视频语料构建、深度估计、分割、物体描述
2. Generation: 分类别编辑 (FLUX-Kontext, Wanxi 等)
3. Filtering: MLM 质量评分 ≥3 保留

**分类**:
- Spatially-Aligned: 全局风格、背景替换、局部修改/移除/添加、字幕编辑
- Non-Spatially-Aligned: 多镜头编辑、创意编辑

**特点**: 平均指令长度 40.6 词，最详细

---

### Señorita-2M (2M pairs)

**Pipeline**:
1. 视频标注 (BLIP-2 captioning)
2. Mask 区域识别 (CogVLM2 + Grounded-SAM2)
3. 专家模型编辑
4. 双层过滤

**专家模型** (基于 CogVideoX 训练):
- Global Stylizer (全局风格)
- Local Stylizer (局部风格)
- Inpainting Expert (物体移除)
- Super-resolution Expert
- 14 个额外任务专家

**过滤**:
- Visual level: CLIP 特征相似度
- Instruction level: 文本对齐验证

---

### Ditto-1M (1M triplets)

**Pipeline**:
1. Preprocessing: 60 GPU-days
2. Generation: 6,000 GPU-days (蒸馏后，原需 30,000)
3. Post-processing: 6,000 GPU-days

**核心创新**:
- 专业图像编辑器 + In-context 视频生成器融合
- 模型蒸馏降低 80% 成本
- Intelligent agent 驱动质量过滤

**成本**: 12,000+ GPU-days (最透明的成本披露)

---

### InsViE-1M (1M triplets)

**Pipeline**:
1. 首帧生成: 不同 CFG 强度生成多个编辑样本
2. Stage 1 过滤: GPT-4o 评估编辑首帧
3. 帧传播: 光流传播到后续帧
4. Stage 2 过滤: 运动一致性 + 帧质量评估

**特点**:
- 双阶段自动过滤，无需人工标注
- 详细的评分指南结合光流信息

---

### ReCo-Data (500K+ pairs)

**Pipeline**:
1. Raw Data Pre-processing
2. Object Segmentation
3. Instruction Generation (Gemini-2.5-Flash-Thinking)
4. Condition Pair Construction (首帧编辑 + 深度图)
5. Video Generation (VACE)
6. Quality Evaluation (VLLM)

**任务分布**:
- Object Addition: 115.6K
- Object Removal: 121.6K
- Object Replacement: 156.6K
- Video Stylization: 130.6K

**成本**:
- GPU: ~76,800 小时 (RTX 4090)
- VLLM API: ~$13,600

**验证**: 随机抽样 200 视频人工验证，>90% 高质量

---

## 关键组件对照表

| 组件类型 | 常用工具 | 用途 |
|----------|----------|------|
| 实例分割 | SAM2, Grounded-SAM2 | 提取物体 mask |
| 目标检测 | Grounding DINO | 定位编辑区域 |
| 图像编辑 | FLUX, FLUX-Fill, ControlNet | 编辑首帧 |
| 视频生成 | CogVideoX, VACE, Wan2.1 | I2V 传播 / 端到端编辑 |
| 质量过滤 | GPT-4o, Gemini | 自动评估编辑质量 |
| 时序一致性 | RAFT 光流 | 验证帧间一致性 |
| 深度估计 | Depth Anything, MiDaS | 空间约束 |
| 标注生成 | InternVL2, BLIP-2, LLaMA-3 | 自动生成指令 |

---

## 质量过滤方法对比

| 数据集 | 主过滤 | 辅助过滤 |
|--------|--------|----------|
| VIVID-10M | 用户研究指标 | Mask 增强验证 |
| OpenVE-3M | MLM 质量评分 | 多模态过滤 |
| Señorita-2M | CLIP 视觉相关性 | 文本对齐验证 |
| Ditto-1M | Intelligent Agent | 帧一致性 (99%) |
| InsViE-1M | GPT-4o 自动评估 | 光流运动评估 |
| ReCo-Data | Gemini VLLM | 人工抽样验证 (>90%) |

---

## 成本参考

| 数据集 | 规模 | 成本 |
|--------|------|------|
| Ditto-1M | 1M | 12,000+ GPU-days |
| ReCo-Data | 500K | 76,800 GPU-hours + $13,600 API |

**成本优化策略**:
- 模型蒸馏 (Ditto: 80% 成本降低)
- 混合 Image-Video 数据 (VIVID)
- 自动过滤替代人工 (InsViE, ReCo)

---

## PVTT 数据集构建建议

### 任务定位

PVTT 核心任务: **替换模板视频中的商品 (Object Replacement / Subject Swapping)**

最相关的现有数据集:
- ReCo-Data (Object Replacement: 156.6K)
- Señorita-2M (Object Swap)
- Ditto-1M (Local Editing)

### 推荐方案: Image Edit + I2V 传播

```
1. 采集商品模板视频 (电商平台/自制)
     ↓
2. SAM2 分割出商品区域 + 深度图生成
     ↓
3. 图像编辑器替换首帧商品 (FLUX-Fill / ControlNet)
     ↓
4. I2V 模型传播编辑 (CogVideoX / Wan2.1)
     ↓
5. GPT-4o / Gemini 过滤质量
```

### 关键挑战

1. **商品边界精确分割** - 需要高质量 mask
2. **替换后的视觉一致性** - 光照、阴影、反射
3. **运动传播** - 商品运动轨迹保持
4. **时序一致性** - 帧间无闪烁

### 起步路径

1. **PoC 验证** - 手动走通单个样本流程
2. **Pipeline 自动化** - 构建批量处理脚本
3. **质量评估** - 建立评估标准和过滤机制
4. **规模化** - 逐步扩展数据量

---

## 参考论文

- [VIVID-10M](https://arxiv.org/abs/2411.15260) - arXiv 2024.11
- [OpenVE-3M](https://arxiv.org/abs/2512.07826) - arXiv 2025.12
- [Señorita-2M](https://arxiv.org/abs/2502.06734) - NeurIPS D&B 2025
- [Ditto-1M](https://arxiv.org/abs/2510.15742) - arXiv 2025.10
- [InsViE-1M](https://arxiv.org/abs/2503.20287) - ICCV 2025
- [ReCo-Data](https://arxiv.org/abs/2512.17650) - arXiv 2025.12

---

Last updated: 2026-01-20
