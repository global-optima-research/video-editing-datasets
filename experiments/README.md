# PVTT Pipeline 实验记录

本目录管理 PVTT 数据集构建 pipeline 的验证实验。

## 目录结构

```
experiments/
├── README.md           # 本文件（实验索引）
├── reports/            # 总结报告
│   ├── pvtt-experiments-summary_2026-01-22.md
│   └── pvtt-experiments-summary_2026-01-23.md
├── plans/              # 实验方案（已验证/已放弃）
│   └── wan_ti2v_2026-01-22.md
├── logs/               # 实验日志
│   ├── shot-detection_2026-01-20.md
│   ├── sam2-segmentation_2026-01-20.md
│   ├── propainter_2026-01-20.md
│   ├── lama_2026-01-20.md
│   ├── flux_fill_2026-01-20.md
│   ├── diffueraser_2026-01-20.md
│   ├── omnieraser_2026-01-22.md
│   ├── pipeline_redesign_2026-01-23.md
│   └── product_video_generator_2026-01-23.md
└── results/            # 实验产物（图片、视频等）
    ├── shot-detection/
    ├── sam2-segmentation/
    ├── propainter/
    ├── lama/
    ├── flux_fill/
    ├── diffueraser/
    ├── omnieraser/
    └── wan_ti2v/
```

---

## 实验日志索引

### 2026-01

| 日期 | 主题 | 文件 | 关键结论 |
|------|------|------|---------|
| 01-23 | ProductVideoGenerator | [product_video_generator_2026-01-23.md](logs/product_video_generator_2026-01-23.md) | PoC 方案：多视角商品图片 + masked_video → 商品视频生成 |
| 01-23 | Pipeline 重设计 | [pipeline_redesign_2026-01-23.md](logs/pipeline_redesign_2026-01-23.md) | 新方案：空镜视频 + 物体插入，绕过背景修复和运动迁移困境 |
| 01-22 | OmniEraser 修复 | [omnieraser_2026-01-22.md](logs/omnieraser_2026-01-22.md) | 单帧优秀 (7/10) - 阴影移除成功，但视频闪烁明显 |
| 01-20 | DiffuEraser 修复 | [diffueraser_2026-01-20.md](logs/diffueraser_2026-01-20.md) | 失败 (3/10) - 继承 ProPainter 光流局限性，阴影残留 |
| 01-20 | FLUX Fill 修复 | [flux_fill_2026-01-20.md](logs/flux_fill_2026-01-20.md) | 质量好 (8/10) - 颜色还原优秀，但依赖 prompt 不适合自动化 |
| 01-20 | LaMa 修复 | [lama_2026-01-20.md](logs/lama_2026-01-20.md) | 可用 (6/10) - 手链和阴影移除成功，但有轻微色差和闪烁 |
| 01-20 | ProPainter 修复 | [propainter_2026-01-20.md](logs/propainter_2026-01-20.md) | 失败 (2/10) - 光流方法无法移除全程存在的物体阴影 |
| 01-20 | SAM2 分割 | [sam2-segmentation_2026-01-20.md](logs/sam2-segmentation_2026-01-20.md) | 两个手链成功分割，30fps 处理速度 |
| 01-20 | 镜头切分 | [shot-detection_2026-01-20.md](logs/shot-detection_2026-01-20.md) | 检测到 3 个镜头，Scene 1 (7.67s) 作为测试用例 |

### 已放弃方案

| 日期 | 主题 | 文件 | 结论 |
|------|------|------|------|
| 01-22 | Wan2.2 TI2V 运动迁移 | [wan_ti2v_2026-01-22.md](plans/wan_ti2v_2026-01-22.md) | ❌ 失败 - 运动由首帧决定，无法跨首帧迁移 |

---

## Pipeline 验证进度

### 原方案 (已放弃)

```
Sample 001: Bracelet → Necklace (原方案：分割→修复→编辑→TI2V)
├── [x] 镜头切分 (PySceneDetect) - 3 scenes
├── [x] SAM2 分割 - 184 masks
├── [x] ProPainter 修复 - 失败 (2/10, 光流方法不适用)
├── [x] LaMa 修复 - 可用 (6/10, 有轻微色差和闪烁)
├── [x] FLUX Fill 修复 - 质量好 (8/10, 但依赖 prompt 不适合自动化)
├── [x] DiffuEraser 修复 - 失败 (3/10, 继承 ProPainter 局限性)
├── [x] OmniEraser 修复 - 单帧优秀 (7/10, 阴影移除成功但视频闪烁)
├── [x] Wan2.2 TI2V - 失败 (运动由首帧决定，无法跨首帧迁移)
└── [!] 方案遇到根本性困难，已切换到新方案
```

### 新方案：ProductVideoGenerator

```
Pipeline: masked_video + 多视角商品图片 → 商品视频
├── [ ] 数据收集 (100 商品 PoC)
├── [ ] 数据处理 (SAM2 分割 + masked_video)
├── [ ] 模型实现 (ProductEncoder + DiT 注入)
├── [ ] PoC 训练
└── [ ] 评估 & 泛化测试
```

---

## 测试样本

详见 [samples/README.md](../samples/README.md)

---

## 命名规范

### 实验日志

格式：`{主题}_{日期}.md`

示例：
- `shot-detection_2026-01-20.md`
- `sam2-segmentation_2026-01-20.md`
- `videopainter_2026-01-20.md`

---

Last updated: 2026-01-23
