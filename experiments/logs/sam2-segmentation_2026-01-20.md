# SAM2 视频分割实验

**日期**: 2026-01-20
**主题**: 对 Scene 1 进行手链分割
**测试用例**: bracelet_to_necklace Scene 1 (1280x1024, 184帧)

---

## 实验背景

使用 SAM2 (Segment Anything Model 2) 对视频中的手链进行分割，生成 mask 序列用于后续的 VideoPainter 修复。

---

## 实验配置

| 参数 | 值 |
|------|-----|
| 模型 | SAM2.1 Hiera Large |
| Checkpoint | sam2.1_hiera_large.pt (898MB) |
| 设备 | 5090 GPU, CUDA, bfloat16 |
| 环境 | wan conda 环境 |

---

## 输入

| 项目 | 值 |
|------|-----|
| 视频 | source_video-Scene-001.mp4 |
| 帧数 | 184 |
| 分辨率 | 1280x1024 |
| 帧格式 | JPEG (scene1_frames/) |

**输入首帧**:

![input_frame](../results/sam2-segmentation/input_frame.jpg)

---

## Point Prompts

使用多点击提示标注两个手链（经过多次迭代优化）：

| Object ID | 描述 | 点击位置 (x, y) | 数量 |
|-----------|------|----------------|------|
| 1 | 黑色手链 (Lily) | (550,620), (400,550), (650,700), (300,450) | 4 点 |
| 2 | 银色手链 (Ben) | (850,320), (950,400), (650,350) | 3 点 |

**迭代过程**:
- v1: 单点击中背景，失败
- v2: 调整点位，黑色手链成功，银色失败
- v3: 银色成功，黑色只分割内部
- **v4**: 多点覆盖手链带，两个手链均成功

---

## 实验结果

### 处理统计

| 指标 | 值 |
|------|-----|
| 帧加载速度 | 33 fps |
| 传播速度 | 30 fps |
| 总处理时间 | ~11 秒 |
| 输出 masks | 184 个 PNG 文件 |

### 分割效果

**Mask 叠加可视化** (红色区域为检测到的 mask):

![mask_overlay](../results/sam2-segmentation/mask_overlay.jpg)

**原始 Mask** (白色为前景):

![mask_frame0](../results/sam2-segmentation/mask_frame0.png)

**观察**:
- ✅ 两个手链完整分割（带+内部区域）
- ✅ 黑色手链 (Lily): 手链带、延长链、名牌均被覆盖
- ✅ 银色手链 (Ben): 编织链、装饰均被覆盖
- ⚠️ 内部有少量噪点，不影响使用

### 输出文件

```
scene1_masks/
├── 00000.png  # 首帧 mask
├── 00001.png
├── ...
└── 00183.png  # 末帧 mask (共 184 个)
```

---

## 技术细节

### SAM2 Video Predictor 工作流程

```python
# 1. 初始化
predictor = build_sam2_video_predictor(model_cfg, checkpoint, device)
inference_state = predictor.init_state(video_path=frames_dir)

# 2. 添加点击提示
predictor.add_new_points_or_box(
    inference_state, frame_idx=0, obj_id=1,
    points=[[500, 400]], labels=[1]
)

# 3. 传播到所有帧
for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
    masks[frame_idx] = (mask_logits > 0.0).cpu().numpy()
```

### 注意事项

1. **帧格式要求**: SAM2 需要 JPEG 帧目录，不能直接处理 MP4
2. **点击位置敏感**: 点击位置会影响分割结果，可能需要迭代调整
3. **多物体支持**: 每个物体需要单独的 obj_id

---

## 经验总结

### 关键发现

1. **单点容易失败**: 单个点击可能击中背景或只分割部分区域
2. **多点更可靠**: 多个点覆盖物体不同部分可确保完整分割
3. **需要迭代**: 首次尝试很难一次成功，需要查看结果并调整

### 最佳实践

1. 每个物体使用 3-4 个点击点
2. 点击点应覆盖物体的不同部分（边缘、中心、延伸部分）
3. 生成可视化叠加图检查分割效果
4. 如果效果不佳，调整点位重新运行

---

## 结论

1. **SAM2 分割基本成功**，两个手链主体被正确识别
2. **处理速度快**，184 帧仅需 11 秒
3. **mask 质量可用**，但有改进空间
4. **下一步**: 使用 VideoPainter 基于此 mask 进行视频修复

---

## 相关文件

- 分割脚本: `5090:/data/xuhao/pvtt-pipeline-test/sam2_segment_v4.py` (内联)
- 输入帧: `5090:/data/xuhao/pvtt-pipeline-test/samples/scene1_frames/`
- 输出 masks: `5090:/data/xuhao/pvtt-pipeline-test/samples/scene1_masks_v4/`
- 结果图片:
  - [input_frame.jpg](../results/sam2-segmentation/input_frame.jpg) - 输入首帧
  - [mask_overlay.jpg](../results/sam2-segmentation/mask_overlay.jpg) - Mask 叠加可视化
  - [mask_frame0.png](../results/sam2-segmentation/mask_frame0.png) - 原始 Mask
