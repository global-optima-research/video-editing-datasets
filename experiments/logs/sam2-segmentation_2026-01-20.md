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

使用正负样本点击提示（经过多次迭代优化）：

| Object ID | 描述 | 正样本点 | 负样本点 |
|-----------|------|---------|---------|
| 1 | 黑色手链 (Lily) | (550,620), (400,550), (650,700), (300,450) | (500,520), (550,580) |
| 2 | 银色手链 (Ben) | (850,320), (950,400), (650,350) | (750,380) |

**关键**: 负样本点放在手链内部空白区域，排除背景

**迭代过程**:
- v1: 单点击中背景，失败
- v2: 调整点位，黑色手链成功，银色失败
- v3: 银色成功，黑色只分割内部
- v4: 多点覆盖，但包含了手链内部空白区域
- **v5**: 添加负样本排除内部，只分割手链本身 ✅

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
- ✅ 只分割手链本身，内部空白区域被正确排除
- ✅ 黑色手链 (Lily): 手链带、延长链、名牌、磁扣
- ✅ 银色手链 (Ben): 编织链、金属装饰
- ✅ 适合用于 VideoPainter 修复

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

## Box Prompt vs Point Prompt 对比实验

### 实验目的

探索自动化获取 prompt 的方案，避免手动指定坐标。

### 实验设置

对黑色手链使用 box prompt：
```python
box = [250, 400, 700, 750]  # [x_min, y_min, x_max, y_max]
predictor.add_new_points_or_box(..., box=box)
```

### 结果对比

| Prompt 方式 | 内部空白区域 | 需要负样本 | 自动化难度 |
|------------|-------------|-----------|-----------|
| Point prompt | ❌ 包含 | 是 | 高（需手动调点） |
| **Box prompt** | ✅ 排除 | 否 | **低（只需 bbox）** |

**Point prompt 结果** (包含内部):

![point_prompt](../results/sam2-segmentation/mask_overlay_v4_reference.jpg)

**Box prompt 结果** (正确排除内部):

![box_prompt](../results/sam2-segmentation/mask_box_test.jpg)

### 关键发现

**Box prompt 对环形物体效果更好**：SAM2 能正确识别手链是"带状物体"而非"填充区域"。

### 自动化方案

```
Grounding DINO ("bracelet") → bbox 坐标
            ↓
SAM2 (box prompt) → 正确的 mask
```

不需要手动指定正负样本点。

---

## 经验总结

### Point Prompt 最佳实践（手动标注场景）

1. 每个物体使用 3-4 个**正样本**点击点覆盖不同部分
2. 对于环形物体（手链、项链），添加**负样本**点排除内部空白
3. 正样本 (label=1): 点在目标物体上
4. 负样本 (label=0): 点在不想要的区域（如内部空白）

### Box Prompt 最佳实践（自动化场景）

1. 使用检测模型（Grounding DINO）获取 bbox
2. 直接用 box prompt，无需正负样本点
3. 对环形物体效果优于 point prompt

---

## 结论

1. **SAM2 分割基本成功**，两个手链主体被正确识别
2. **处理速度快**，184 帧仅需 11 秒
3. **mask 质量可用**，但有改进空间
4. **下一步**: 使用 VideoPainter 基于此 mask 进行视频修复

---

## 相关文件

- 分割脚本: 内联 Python (见上方迭代过程)
- 输入帧: `5090:/data/xuhao/pvtt-pipeline-test/samples/scene1_frames/`
- 输出 masks: `5090:/data/xuhao/pvtt-pipeline-test/samples/scene1_masks_v5/`
- 结果图片:
  - [input_frame.jpg](../results/sam2-segmentation/input_frame.jpg) - 输入首帧
  - [mask_overlay.jpg](../results/sam2-segmentation/mask_overlay.jpg) - Point prompt + 负样本 (v5)
  - [mask_overlay_v4_reference.jpg](../results/sam2-segmentation/mask_overlay_v4_reference.jpg) - Point prompt 包含内部 (v4)
  - [mask_box_test.jpg](../results/sam2-segmentation/mask_box_test.jpg) - Box prompt 测试
  - [mask_frame0.png](../results/sam2-segmentation/mask_frame0.png) - 原始 Mask
