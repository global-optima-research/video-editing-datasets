# Ditto-1M 深度分析

> Scaling Instruction-Based Video Editing with a High-Quality Synthetic Dataset (arXiv 2025.10)

**论文**: [arxiv.org/abs/2510.15742](https://arxiv.org/abs/2510.15742)
**数据集**: [huggingface.co/datasets/QingyanBai/Ditto-1M](https://huggingface.co/datasets/QingyanBai/Ditto-1M)
**代码**: [github.com/EzioBy/Ditto](https://github.com/EzioBy/Ditto)
**项目**: [editto.net](https://editto.net/)

---

## 核心贡献

1. **12,000+ GPU-days** 构建 **1M 高质量视频编辑三元组**
2. 开源数据集 + 训练好的模型 (Editto)
3. Modality Curriculum Learning 策略
4. 完整的数据生成 Pipeline

---

## 数据生成 Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Source Video Filtering                                       │
│     Pexels 200K+ 视频 → DINOv2 去重 → CoTracker3 运动分析        │
│     → 统一分辨率 + 20 FPS                                        │
├─────────────────────────────────────────────────────────────────┤
│  2. Instruction Generation                                       │
│     Qwen2.5 VL 两步生成:                                         │
│     ① 生成 dense caption 描述视频内容                            │
│     ② 基于视频+caption 生成编辑指令                              │
├─────────────────────────────────────────────────────────────────┤
│  3. Visual Context Preparation                                   │
│     ① Edited Keyframe: Qwen-Image 编辑首帧                       │
│     ② Depth Video: VideoDepthAny 生成深度视频                    │
├─────────────────────────────────────────────────────────────────┤
│  4. In-Context Video Generation                                  │
│     VACE 模型，输入:                                             │
│     - 编辑后首帧 (外观引导)                                      │
│     - 深度视频 (结构约束)                                        │
│     - 文本指令 (语义方向)                                        │
│     成本优化: 量化 + 蒸馏 → 降至原 20% 计算量                    │
├─────────────────────────────────────────────────────────────────┤
│  5. Quality Curation                                             │
│     ① VLM 过滤: 指令保真度、内容保持、视觉质量、安全合规         │
│     ② Wan2.2 fine-denoiser 增强                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 各阶段详解

### Stage 1: Source Video Filtering

**来源**: Pexels 200K+ 专业视频

**过滤步骤**:
1. **DINOv2 去重**: 提取视觉特征，过滤相似度超过阈值的视频
2. **CoTracker3 运动分析**: 跟踪网格点，计算累积位移作为运动分数，移除低运动内容
3. **标准化**: 统一分辨率 1280×720，帧率 20 FPS

### Stage 2: Instruction Generation

**工具**: Qwen2.5 VL

**两步生成策略**:
```
Step 1: Dense Caption
输入: 视频
输出: 详细描述视频内容的文本

Step 2: Edit Instruction
输入: 视频 + Dense Caption
输出: 创意且合理的编辑指令
```

**指令类型**:
- 全局风格变换
- 局部物体修改

### Stage 3: Visual Context Preparation

**两个互补的引导机制**:

| 引导类型 | 工具 | 作用 |
|----------|------|------|
| Edited Keyframe | Qwen-Image | 编辑首帧，提供外观原型 |
| Depth Video | VideoDepthAny | 提供时空结构骨架 |

### Stage 4: In-Context Video Generation

**模型**: VACE (Video Creation and Editing)

**三重条件输入**:
```
VACE 输入:
├── 编辑后首帧 → 外观引导
├── 深度视频 → 结构约束
└── 文本指令 → 语义方向
```

**成本优化**:
- 原始成本: ~50 GPU-minutes/sample
- 优化后: ~10 GPU-minutes/sample (量化 + 蒸馏)
- 降低 **80%** 计算开销

### Stage 5: Quality Curation

**VLM 自动评估维度**:
1. 指令保真度 (Instruction Fidelity)
2. 内容保持 (Content Preservation)
3. 视觉质量 (Visual Quality)
4. 安全合规 (Safety Compliance)

**后处理增强**:
- Wan2.2 fine-denoiser
- 4步逆扩散精炼

---

## 数据集规格

| 属性 | 值 |
|------|-----|
| 总量 | ~1M 三元组 |
| 全局编辑 | ~700K (风格、环境) |
| 局部编辑 | ~300K (物体替换/添加/移除) |
| 分辨率 | 1280×720 |
| 帧数 | 101 frames @ 20 FPS |
| 时长 | ~5 秒/视频 |
| 来源 | Pexels 专业素材 |
| 人物视频占比 | ~50% |

---

## 成本分解

| 阶段 | GPU-days | 占比 |
|------|----------|------|
| Preprocessing | 60 | 0.5% |
| Generation | 6,000 | 50% |
| Post-processing | 6,000 | 50% |
| **Total** | **12,000+** | 100% |

**注**: Generation 原需 30,000 GPU-days，蒸馏后降至 6,000

---

## Editto 模型训练

### 架构

基于 VACE 骨干:
- **Context Branch**: 提取源视频和参考帧的时空特征
- **Main Branch**: DiT 架构，在视觉+文本联合引导下生成编辑视频

### Modality Curriculum Learning (MCL)

逐步减少视觉依赖，过渡到纯文本引导:

```
阶段 1 (前 5,000 步):
└── 提供编辑后参考帧作为强 scaffold + 文本指令

阶段 2 (后续训练):
└── 逐步降低视觉辅助出现的概率

阶段 3 (最终):
└── 完全丢弃视觉上下文，仅依赖文本指令
```

### 训练配置

| 参数 | 值 |
|------|-----|
| GPU 集群 | 64 GPUs |
| 训练步数 | ~16,000 |
| 优化器 | AdamW |
| 学习率 | 1e-4 (constant) |
| 微调范围 | 仅 context blocks 的线性投影层 |

---

## 实验结果

### 测试集

50 个视频，每个 5 条指令，共 250 测试样例

### 定量对比

| 方法 | CLIP-T ↑ | CLIP-F ↑ | VLM Score ↑ |
|------|----------|----------|-------------|
| TokenFlow | 23.63 | 98.43 | 7.10 |
| InsV2V | 22.49 | 97.99 | 6.55 |
| InsViE | 23.56 | 98.78 | 7.35 |
| **Editto** | **25.54** | **99.03** | **8.10** |

### 用户研究 (1,000 投票)

| 方法 | Edit Accuracy | Temporal Consistency | Overall Quality |
|------|---------------|----------------------|-----------------|
| InsViE | 2.28 | 2.30 | 2.36 |
| **Editto** | **3.85** | **3.76** | **3.86** |

---

## 关键技术洞察

### 1. 深度视频约束

**核心发现**: 深度约束 = 几何不变，只改外观

> "The predicted depth video acts as a **dynamic structural scaffold**, providing an explicit, frame-by-frame guide for the **structure and geometry** of the scene."

**影响**:
- ✅ 保持场景结构稳定
- ✅ 相机运动自然保持
- ❌ **不支持物体形状变化**

### 2. VLM 自动化

**优势**:
- 指令生成: Qwen2.5 VL
- 质量过滤: VLM 多维度评估
- 无需人工标注

### 3. 蒸馏优化

**效果**: 计算成本降低 80%
**方法**: 后训练量化 + 知识蒸馏

---

## 对 PVTT 的适用性

### 优势

| 特性 | 适用性 |
|------|--------|
| Pipeline 完整 | ✅ 可参考整体流程 |
| VLM 自动化 | ✅ 降低标注成本 |
| 成本优化策略 | ✅ 蒸馏方法可复用 |
| 质量过滤 | ✅ 多维度评估框架 |

### 局限

| 特性 | 问题 |
|------|------|
| 深度约束 | ❌ 不支持形状变化 (手表→项链) |
| VACE 依赖 | ⚠️ 需替换为支持形状变化的方案 |

### 可借鉴的组件

1. **视频过滤策略**: DINOv2 去重 + CoTracker3 运动分析
2. **VLM 指令生成**: 两步生成策略
3. **质量过滤框架**: 多维度 VLM 评估
4. **成本优化**: 量化 + 蒸馏

---

## 参考文献

- [Ditto-1M](https://arxiv.org/abs/2510.15742) - arXiv 2025.10
- [VACE](https://github.com/ali-vilab/VACE) - 阿里视频生成模型
- [Qwen2.5 VL](https://github.com/QwenLM/Qwen2-VL) - 视觉语言模型
- [VideoDepthAny](https://github.com/DepthAnything/Video-Depth-Anything) - 视频深度估计

---

Last updated: 2026-01-20
