# PVTT Training Pipeline

> Training-based methods for Product Video Template Transfer (PVTT)

## PVTT 任务定义

**输入**：
1. Template Video：一个成功的产品推广视频
2. New Product Image(s)：新产品的图片（1-N张）

**输出**：
- 新产品的推广视频
- 保持模板的视觉风格、运镜、节奏

详细任务定义见 [pvtt-training-free](https://github.com/global-optima-research/pvtt-training-free)。

## 研究定位

本项目专注于 **Training-based** 方法，与 [pvtt-training-free](https://github.com/global-optima-research/pvtt-training-free) 形成互补：

| 项目 | 方法类型 | 代表方案 |
|------|----------|----------|
| pvtt-training-free | Training-Free | Flux.2 + TI2V, RF-Solver Inversion |
| **pvtt-training-pipeline** | Training-Based | LoRA 微调, 数据集构建 |

## 当前进展

### Wan2.1-VACE Zero-Shot 实验

测试 Wan2.1-VACE-1.3B 能否 zero-shot 完成 PVTT 任务。

**结论**：Zero-shot VACE 无法实现物体替换

| 测试场景 | 结果 |
|---------|------|
| Reference-only | ✅ 参考图有效引导生成 |
| Video Inpainting | ✅ 能修复/重建视频区域 |
| **Reference + Video + Mask** | ❌ 参考图被忽略，重建原视频内容 |
| Zeroed Reactive Stream | ⚠️ 参考图生效，但丢失运动信息 |

详细实验记录：[experiments/logs/wan2.1-vace-zero-shot_2026-01-23.md](experiments/logs/wan2.1-vace-zero-shot_2026-01-23.md)

### 下一步方向

1. **LoRA 训练**：微调 VACE 增强 reference image 条件注入
2. **两阶段方案**：Reference-to-video + ControlNet 合成
3. **运动迁移**：提取原视频运动轨迹，应用到参考图生成
4. **其他模型**：探索 AnyDoor/Paint-by-Example 的视频版本

## 目录结构

```
pvtt-training-pipeline/
├── baseline/
│   └── wan2.1-vace/              # VACE baseline 测试
├── samples/
│   └── teapot/                   # 测试样本（紫砂壶）
│       ├── video_frames/         # 原视频帧
│       ├── masks/                # SAM2 分割 mask
│       └── reference_images/     # 参考图像
├── experiments/
│   ├── logs/                     # 实验日志
│   └── results/                  # 实验输出
├── scripts/                      # 工具脚本
└── papers/                       # 论文分析
```

## 相关项目

- [pvtt-training-free](https://github.com/global-optima-research/pvtt-training-free) - Training-free 方法
- [pvtt](https://github.com/global-optima-research/pvtt) - PVTT Benchmark

## 目标会议

CVPR 2027

---

Last updated: 2026-01-24
