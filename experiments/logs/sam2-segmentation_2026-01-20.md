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

使用点击提示标注两个手链：

| Object ID | 描述 | 点击位置 (x, y) | Label |
|-----------|------|----------------|-------|
| 1 | 手链 1 (黑色, "Lily") | (500, 400) | 1 (positive) |
| 2 | 手链 2 (银色, "Ben") | (780, 600) | 1 (positive) |

**注意**: 点击位置为估计值，基于图像中心附近的手链位置。

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
- ✅ 成功检测到两个手链的环形结构
- ✅ 手链主体轮廓清晰
- ⚠️ 存在一些背景噪点（玻璃碗、心形装饰、礼盒被误检）
- ⚠️ 手链内部空洞未填充（只有边缘）

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

## 问题与改进

### 当前问题

1. **背景噪点**: 一些装饰物被误识别
2. **空洞未填充**: 手链是环形，内部空洞在 mask 中显示为白色

### 改进方向

1. 添加负样本点击排除背景装饰
2. 使用 box prompt 而非 point prompt 提高精度
3. 对 mask 进行形态学处理（膨胀、填充）

---

## 结论

1. **SAM2 分割基本成功**，两个手链主体被正确识别
2. **处理速度快**，184 帧仅需 11 秒
3. **mask 质量可用**，但有改进空间
4. **下一步**: 使用 VideoPainter 基于此 mask 进行视频修复

---

## 相关文件

- 分割脚本: `5090:/data/xuhao/pvtt-pipeline-test/sam2_segment.py`
- 输入帧: `5090:/data/xuhao/pvtt-pipeline-test/samples/scene1_frames/`
- 输出 masks: `5090:/data/xuhao/pvtt-pipeline-test/samples/scene1_masks/`
- 结果图片:
  - [input_frame.jpg](../results/sam2-segmentation/input_frame.jpg) - 输入首帧
  - [mask_overlay.jpg](../results/sam2-segmentation/mask_overlay.jpg) - Mask 叠加可视化
  - [mask_frame0.png](../results/sam2-segmentation/mask_frame0.png) - 原始 Mask
