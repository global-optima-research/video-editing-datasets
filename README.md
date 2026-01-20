# Video Editing Datasets Research

PVTT (Product Video Template Transfer) 研究项目的技术调研与方案设计。

目标: CVPR 2027

> **工作方法**: 本 repo 使用 Claude Code 辅助调研，详见 [WORKFLOW.md](WORKFLOW.md) (4h 完成全部调研)

---

## 项目概览

```
目标任务: 商品视频模板迁移
├── 输入: 模板视频 (含商品 A) + 目标商品图 (商品 B)
├── 输出: 编辑视频 (商品 A → 商品 B)
└── 约束: 保持运动、光照、背景，支持形状变化 (手表→项链)
```

**核心发现**: 将 Video Editing 转化为 Video Object Insertion

```
Pipeline:
模板视频 → 镜头切分 → 单镜头片段 → SAM2 分割 → VideoPainter 修复 → 干净背景
                                       ↓                               ↓
                                   Box 序列 ────────────────→ VideoAnyDoor → 合成视频
                                                              (或 InsertAnywhere)
新商品图 (N张) ─────────────────────────────────────────────────────┘
```

---

## 文档索引

### 核心文档

| 文档 | 说明 |
|------|------|
| [pvtt-dataset-proposal.md](pvtt-dataset-proposal.md) | PVTT 数据集构建技术方案 |
| [dataset-construction-analysis.md](dataset-construction-analysis.md) | 数据集构建方法分析 |
| [awesome-video-editing-datasets.md](awesome-video-editing-datasets.md) | 公开视频编辑数据集列表 |
| [motion-lora.md](motion-lora.md) | Motion LoRA 技术详解 |

### 论文分析

详见 [papers/README.md](papers/README.md)

| 类别 | 论文 |
|------|------|
| **数据集构建** | Ditto-1M |
| **运动保持** | I2VEdit (Motion LoRA) |
| **物体替换** | VideoSwap, VideoAnyDoor |
| **Pipeline 组件** | SAM2, ProPainter |

---

## 快速对比

### 运动保持方案

| 方案 | 形状变化 | 自动化 | 成本 | 推荐 |
|------|----------|--------|------|------|
| 深度约束 (Ditto) | ❌ | ✅ 高 | ✅ 低 | - |
| Motion LoRA (I2VEdit) | ⚠️ 有限 | ⚠️ 中 | ❌ 高 | - |
| 语义点 (VideoSwap) | ✅ | ❌ 低 | ⚠️ 中 | - |
| **Box 序列 (VideoAnyDoor)** | ✅ | ✅ 高 | ✅ 低 | **PVTT 首选** |

### Pipeline 组件

| 组件 | 作用 | 发表 | 代码 |
|------|------|------|------|
| SAM2 | 视频分割 | Meta 2024 | [GitHub](https://github.com/facebookresearch/segment-anything-2) |
| **VideoPainter** | 视频修复 | SIGGRAPH 2025 | [GitHub](https://github.com/TencentARC/VideoPainter) |
| **VideoAnyDoor** | 物体插入 | SIGGRAPH 2025 | [GitHub](https://github.com/yuanpengtu/VideoAnydoor) |
| InsertAnywhere | 物体插入 (备选) | Dec 2025 | [GitHub](https://github.com/myyzzzoooo/InsertAnywhere) |

---

## 目录结构

```
video-editing-datasets/
├── README.md                          # 本文件
├── pvtt-dataset-proposal.md           # PVTT 数据集技术方案
├── dataset-construction-analysis.md   # 构建方法分析
├── awesome-video-editing-datasets.md  # 公开数据集列表
├── motion-lora.md                     # Motion LoRA 技术详解
└── papers/
    ├── README.md                      # 论文索引
    ├── ditto-1m-analysis.md           # Ditto-1M 分析
    ├── i2vedit-analysis.md            # I2VEdit 分析
    ├── videoswap-analysis.md          # VideoSwap 分析
    ├── videoanydoor-analysis.md       # VideoAnyDoor 分析
    ├── video-object-insertion.md      # Video Object Insertion 综合调研
    ├── sam2-analysis.md               # SAM2 分析
    └── propainter-analysis.md         # ProPainter 分析
```

---

## 数据规模

| 原始数据 | 数量 | 处理后 |
|----------|------|--------|
| 商品数 | 53 | - |
| 视频 | 53 | ~200 片段 (镜头切分) |
| 商品图片 | ~265 张 | 每商品 3-8 张 |

**配对数量** (交叉配对):
| 策略 | 配对数 | GPU-hours |
|------|--------|-----------|
| 单图 | ~10,400 | ~200 |
| 采样 (2张/商品) | ~20,800 | ~410 |
| 全图 | ~52,000 | ~1,000 |

---

## 相关资源

- **PVTT Benchmark**: [github.com/global-optima-research/pvtt](https://github.com/global-optima-research/pvtt)
- **数据来源**: Etsy 商品视频 (53 products, 11 categories)

---

## License

MIT

---

Last updated: 2026-01-20
