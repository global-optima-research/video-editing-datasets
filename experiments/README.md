# PVTT Pipeline 实验记录

本目录管理 PVTT 数据集构建 pipeline 的验证实验。

## 目录结构

```
experiments/
├── README.md           # 本文件（实验索引）
└── logs/               # 实验日志
    ├── shot-detection_2026-01-20.md
    └── sam2-segmentation_2026-01-20.md
```

---

## 实验日志索引

### 2026-01

| 日期 | 主题 | 文件 | 关键结论 |
|------|------|------|---------|
| 01-20 | SAM2 分割 | [sam2-segmentation_2026-01-20.md](logs/sam2-segmentation_2026-01-20.md) | 两个手链成功分割，30fps 处理速度 |
| 01-20 | 镜头切分 | [shot-detection_2026-01-20.md](logs/shot-detection_2026-01-20.md) | 检测到 3 个镜头，Scene 1 (7.67s) 作为测试用例 |

---

## Pipeline 验证进度

```
Sample 001: Bracelet → Necklace
├── [x] 镜头切分 (PySceneDetect) - 3 scenes
├── [x] SAM2 分割 - 184 masks
├── [ ] VideoPainter 修复
├── [ ] VideoAnyDoor 插入
└── [ ] VLM 评估
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

Last updated: 2026-01-20
