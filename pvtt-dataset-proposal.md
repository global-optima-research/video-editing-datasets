# PVTT 训练数据集构建技术方案

Product Video Template Transfer 训练数据集构建方案。

---

## 0. 背景与动机

### 0.1 为什么需要训练数据集

我们已有 **Training-free 两阶段方法** (RF-Solver TI2V) 作为基线：

```
Training-free 基线 (RF-Solver TI2V):
源视频 → RF-Solver Inversion → 噪声 → 替换首帧 → TI2V Denoising → 编辑视频
```

该方法已在 bracelet → necklace 等任务上取得不错效果，但仍有提升空间。

**核心逻辑**：
```
要训练出超越 Training-free 基线的模型
    ↓
训练数据质量必须 >> Training-free 结果质量
    ↓
需要用更精细的 Pipeline 生成高质量 Ground Truth
```

### 0.2 数据集构建方案定位

| 方案 | 用途 | 质量要求 |
|------|------|----------|
| RF-Solver TI2V | 推理基线 | 中等 (已有) |
| **本方案 (VideoAnyDoor/InsertAnywhere)** | **生成训练数据** | **高** |
| 训练后的模型 | 推理 (目标) | 高 |

本方案的目标是构建 **高质量训练数据集**，而非作为推理方法。

---

## 1. 目标

构建用于训练 PVTT 模型的大规模视频编辑数据集，质量需显著优于 Training-free 基线。

### 1.1 任务定义

```
输入: 模板视频 V (含商品 A) + 目标商品图 T (商品 B)
输出: 编辑视频 V' (商品 A 被替换为商品 B)

约束:
├── 保持模板视频的相机运动
├── 保持模板视频的光照环境
├── 保持模板视频的背景场景
└── 支持不同形状的商品替换 (如手表→项链)
```

### 1.2 数据集规格目标

| 指标 | 目标值 |
|------|--------|
| 训练样本数 | 10K - 100K |
| 视频分辨率 | 720P (1280×720) |
| 视频时长 | 3-10 秒 |
| 商品品类 | 10+ 类 |

---

## 2. 数据来源

### 2.1 现有数据

基于 `pvtt-benchmark` 数据集：

| 资源 | 数量 | 说明 |
|------|------|------|
| 商品数 | 53 个 | Etsy 商品 |
| 模板视频 | 53 个 | 每商品 1 个展示视频 |
| 商品图片 | ~265 张 | 每商品 3-8 张产品图 (平均 ~5 张) |
| 品类 | 11 个 | Jewelry, Toys, Home, Clothing 等 |

> **注意**:
> - Etsy 视频通常为多镜头拼接，需要先进行镜头切分
> - 多张商品图提供不同角度/视角，可增加训练多样性

### 2.2 交叉配对策略

利用现有数据交叉配对生成训练样本：

```
样本 i 的视频片段 + 样本 j 的商品图 k → 训练对 (i, j, k)

配对数量计算:
├── 视频片段: 53 视频 × ~4 镜头 = ~200 片段
├── 目标商品: 52 个 (排除自身)
├── 商品图片: ~5 张/商品
└── 总配对数: 200 × 52 × 5 = ~52,000 对
```

**图片选择策略**:
| 策略 | 说明 | 配对数 |
|------|------|--------|
| 单图 (主图) | 只用每个商品的主图 | ~10,400 |
| 多图 (全部) | 用每个商品的所有图 | ~52,000 |
| 多图 (采样) | 每商品随机采样 2-3 张 | ~20,000 - 30,000 |

### 2.3 配对难度分级

| 难度 | 定义 | 示例 | 预估占比 |
|------|------|------|----------|
| **Easy** | 同子品类 | necklace → necklace | 10% |
| **Medium** | 同品类不同子品类 | necklace → bracelet | 30% |
| **Hard** | 跨品类相似形态 | jewelry → accessories | 30% |
| **Expert** | 跨品类不同形态 | jewelry → toys | 30% |

---

## 3. 技术方案

### 3.1 核心思路

**将 Video Editing 问题转化为 Video Object Insertion 问题**

```
传统思路 (Video Editing):
视频 + 编辑指令 → 编辑模型 → 编辑后视频
问题: 难以支持大幅形状变化

新思路 (Video Object Insertion):
视频 → 移除商品 → 干净背景 → 插入新商品 → 合成视频
优势: 形状完全自由
```

### 3.2 Pipeline 概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    PVTT Dataset Generation Pipeline              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1: Preprocessing (预处理)                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  模板视频 V_i (多镜头)                                   │    │
│  │       ↓                                                  │    │
│  │  镜头切分 → 单镜头片段 {V_i^1, V_i^2, ...}               │    │
│  │       ↓                                                  │    │
│  │  SAM2 分割 → 商品 Mask 序列 M_i                          │    │
│  │       ↓                                                  │    │
│  │  VideoPainter → 干净背景视频 B_i                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Stage 2: Pair Generation (配对生成)                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  For each (i, j) where i ≠ j:                            │    │
│  │       目标商品图 T_j = 样本 j 的 source_product          │    │
│  │       去背景处理 T_j → T_j' (透明背景)                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Stage 3: Video Synthesis (视频合成)                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  VideoAnyDoor(                                           │    │
│  │      background_video = B_i,                             │    │
│  │      reference_image = T_j',                             │    │
│  │      mask_sequence = M_i                                 │    │
│  │  ) → 合成视频 V'_ij                                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Stage 4: Quality Filtering (质量过滤)                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  VLM 评估:                                               │    │
│  │  ├── 商品身份一致性 (T_j vs V'_ij 中的商品)              │    │
│  │  ├── 时序连贯性 (帧间一致性)                             │    │
│  │  ├── 光照自然度                                          │    │
│  │  └── 整体质量分数                                        │    │
│  │                                                          │    │
│  │  过滤: score < threshold → 丢弃                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Output: (V_i, S_i, T_j, V'_ij, metadata)                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 各阶段技术细节

### 4.1 Stage 1: Preprocessing

#### 4.1.0 镜头切分 (Shot Detection)

**背景**: Etsy 商品视频通常由多个镜头拼接而成，包含:
- 不同角度的商品展示
- 特写镜头
- 使用场景演示
- 转场效果

**目标**: 将多镜头视频切分为单镜头片段，每个片段内相机运动连续

**工具选项**:

| 工具 | 方法 | 优势 | 劣势 |
|------|------|------|------|
| **PySceneDetect** | 内容变化检测 | 开源，简单 | 需调阈值 |
| **TransNetV2** | 深度学习 | 准确率高 | 需 GPU |
| FFmpeg | 场景变化滤镜 | 无依赖 | 精度一般 |

**推荐方案**: PySceneDetect (ContentDetector)

```python
from scenedetect import detect, ContentDetector, split_video_ffmpeg

# 检测镜头边界
scenes = detect(video_path, ContentDetector(threshold=27.0))

# 切分视频
split_video_ffmpeg(video_path, scenes, output_dir)

# 输出: video_001.mp4, video_002.mp4, ...
```

**参数调优**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `threshold` | 27.0 | 检测灵敏度，越低越敏感 |
| `min_scene_len` | 15 帧 | 最短镜头长度 |

**质量过滤**:
```python
def filter_shots(shots, min_duration=2.0, min_frames=50):
    """过滤过短的镜头"""
    return [s for s in shots if s.duration >= min_duration and s.frames >= min_frames]
```

**预期效果**:
| 原始视频 | 切分后 |
|----------|--------|
| 1 个多镜头视频 (30s) | 3-5 个单镜头片段 (5-10s 每个) |
| 53 个原始视频 | 150-250 个单镜头片段 |

> 切分后配对数量将显著增加: 200 × 199 = 39,800 对

#### 4.1.1 商品分割 (SAM2)

**工具**: Segment Anything Model 2 (SAM2)

> 详细技术分析见 [papers/sam2-analysis.md](papers/sam2-analysis.md)

**输入**: 单镜头视频片段 V_i (经过镜头切分)

**输出**:
- 逐帧 Mask 序列 M_i = {m_1, m_2, ..., m_T}
- 逐帧 Box 序列 (用于 VideoAnyDoor)

**流程**:
```python
# 伪代码
from sam2.build_sam import build_sam2_video_predictor

# 1. 初始化模型 (推荐 SAM2-Large)
predictor = build_sam2_video_predictor("sam2_hiera_large.pt")

# 2. 首帧提供 Box 提示 (从商品检测模型获得)
box = detect_product(first_frame)  # Grounding DINO
predictor.add_box_prompt(frame_idx=0, box=box, obj_id=1)

# 3. 传播到所有帧 (Memory Attention 自动跟踪)
masks, boxes = predictor.propagate()

# 输出: masks (T, H, W), boxes (T, 4)
```

**性能指标**:
| 指标 | 值 |
|------|-----|
| 速度 | 43.8 FPS (A100) |
| 显存 | ~8 GB (SAM2-Large) |
| 单视频耗时 | ~4 秒 (50帧) |

**注意事项**:
- 处理遮挡情况 (Memory Attention 可跨越短暂遮挡)
- 处理多商品情况 (每个商品分配独立 obj_id)
- Mask 平滑处理 (后处理时序平滑)

#### 4.1.2 背景修复 (Video Inpainting)

**推荐工具**: VideoPainter (TencentARC, SIGGRAPH 2025)

**代码**: [github.com/TencentARC/VideoPainter](https://github.com/TencentARC/VideoPainter)

**工具对比**:
| 工具 | 年份 | 架构 | 长视频支持 | 优势 |
|------|------|------|-----------|------|
| **VideoPainter** | **SIGGRAPH 2025** | Diffusion DiT | ✅ 原生任意长度 | SOTA，plug-and-play |
| ProPainter | ICCV 2023 | 光流 + Transformer | ⚠️ 需分块 | 稳定可靠 |
| E²FGVI | CVPR 2022 | 光流传播 | ⚠️ 需分块 | 速度快 |

**输入**: 模板视频 V_i + Mask 序列 M_i (SAM2 输出)

**输出**: 干净背景视频 B_i (商品区域被背景填充)

**流程**:
```python
# VideoPainter 使用示例
# 参考: https://github.com/TencentARC/VideoPainter

# 1. 准备输入
# - masked_video: 带 mask 的视频
# - mask_sequence: 二值 mask 序列

# 2. 推理
# VideoPainter 支持 plug-and-play 方式接入任意 Video DiT
# 使用 context encoder 注入背景信息
```

**核心技术**:
- **双流架构**: Context Encoder (仅 6% 参数) + Video DiT backbone
- **任意长度支持**: Target Region ID Resampling 技术
- **Plug-and-Play**: 可接入任意预训练 Video DiT

**性能指标**:
| 指标 | 值 |
|------|-----|
| 训练数据 | VPData (390K+ clips) |
| 评测基准 | VPBench |

> **备选方案**: 如果 VideoPainter 部署复杂，可退回使用 [ProPainter](https://github.com/sczhou/ProPainter) (ICCV 2023)

#### 4.1.3 预处理产物

每个模板视频 i 生成：
```
preprocessed/
└── {ID}/
    ├── masks/
    │   ├── 0001.png
    │   ├── 0002.png
    │   └── ...
    ├── background.mp4
    └── metadata.json  # bbox 轨迹、帧数等
```

### 4.2 Stage 2: Pair Generation

#### 4.2.1 配对规则

```python
def generate_pairs(samples, strategy="all"):
    pairs = []
    for i, sample_i in enumerate(samples):
        for j, sample_j in enumerate(samples):
            if i == j:
                continue

            if strategy == "same_category":
                if sample_i.category != sample_j.category:
                    continue

            pairs.append({
                "source_id": sample_i.id,
                "target_id": sample_j.id,
                "difficulty": compute_difficulty(sample_i, sample_j)
            })

    return pairs
```

#### 4.2.2 商品图去背景

**工具**: rembg / Segment Anything

**输入**: 商品图 S_j

**输出**: 透明背景商品图 T_j'

```python
# 伪代码
from rembg import remove

target_image = remove(source_image)
target_image.save(f"{ID}_target_nobg.png")
```

### 4.3 Stage 3: Video Synthesis

#### 4.3.1 Video Object Insertion 方案选择

**两个候选方案** (均有代码可用，建议 PoC 阶段对比测试):

| 方案 | 发表 | 核心技术 | 代码 |
|------|------|----------|------|
| **VideoAnyDoor** | SIGGRAPH 2025 | ID Extractor + Pixel Warper | [GitHub](https://github.com/yuanpengtu/VideoAnydoor) |
| **InsertAnywhere** | arXiv Dec 2025 | 4D Scene Geometry + Diffusion | [GitHub](https://github.com/myyzzzoooo/InsertAnywhere) |

**InsertAnywhere 优势**:
- 4D-aware mask generation (更好的遮挡处理)
- 显式光照适配 (illumination-aware)
- 声称在 CLIP-I, DINO-I 指标上超越 Pika-Pro, Kling

**VideoAnyDoor 优势**:
- 已有详细分析文档: [papers/videoanydoor-analysis.md](papers/videoanydoor-analysis.md)
- Box 序列控制直观
- 经过 virtual try-on 场景验证

#### 4.3.2 VideoAnyDoor 调用

**输入**:
- `background_video`: 干净背景视频 B_i (VideoPainter 输出)
- `reference_image`: 去背景商品图 T_j'
- `box_sequence`: Box 序列 (SAM2 输出，定义插入位置)

**输出**: 合成视频 V'_ij

**核心组件**:
| 组件 | 作用 |
|------|------|
| ID Extractor | DINOv2 提取商品特征 |
| Pixel Warper | 将参考图像特征 warp 到目标位置 |
| Box Control | 控制商品位置和运动轨迹 |

**配置**:
```python
config = {
    "model": "VideoAnyDoor",
    "guidance_scale": 7.5,
    "num_inference_steps": 50,
    "control_mode": "box",  # 使用 bbox 控制位置
}
```

#### 4.3.3 InsertAnywhere 调用 (备选)

**输入**:
- `video`: 原始视频 (InsertAnywhere 内部处理背景)
- `reference_image`: 商品图
- `insertion_mask`: 首帧插入位置 mask

**核心组件**:
| 组件 | 作用 |
|------|------|
| 4D Mask Generator | 重建场景几何，跨帧传播 mask |
| Diffusion Video Model | 合成物体 + 局部光照变化 |
| ROSE++ Dataset | 训练数据 (illumination-aware) |

```python
# InsertAnywhere 使用示例
# 参考: https://github.com/myyzzzoooo/InsertAnywhere
```

#### 4.3.4 Mask 到 Box 转换

VideoAnyDoor 支持 Box 序列控制，需要将 Mask 转换为 BBox：

```python
def mask_to_bbox(mask):
    """将二值 Mask 转换为 Bounding Box"""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [x_min, y_min, x_max, y_max]

def masks_to_box_sequence(masks):
    """将 Mask 序列转换为 Box 序列"""
    return [mask_to_bbox(m) for m in masks]
```

#### 4.3.5 尺寸适配

源商品和目标商品尺寸可能不同：

```python
def adapt_box_for_target(source_box, source_aspect, target_aspect):
    """
    根据目标商品的宽高比调整 Box

    策略: 保持 Box 中心不变，调整宽高
    """
    cx = (source_box[0] + source_box[2]) / 2
    cy = (source_box[1] + source_box[3]) / 2

    source_area = (source_box[2] - source_box[0]) * (source_box[3] - source_box[1])

    # 保持面积相近，调整宽高比
    new_h = np.sqrt(source_area / target_aspect)
    new_w = new_h * target_aspect

    return [cx - new_w/2, cy - new_h/2, cx + new_w/2, cy + new_h/2]
```

### 4.4 Stage 4: Quality Filtering

#### 4.4.1 评估维度

| 维度 | 评估方法 | 权重 |
|------|----------|------|
| **商品一致性** | CLIP/DINO 特征相似度 | 0.3 |
| **时序连贯性** | 帧间光流一致性 | 0.2 |
| **光照自然度** | VLM 主观评分 | 0.2 |
| **整体质量** | VLM 综合评分 | 0.3 |

#### 4.4.2 自动评估

```python
def evaluate_sample(original_video, synthesized_video, target_image):
    scores = {}

    # 商品一致性: DINO 特征相似度
    target_feat = dino_encoder(target_image)
    synth_product_feat = dino_encoder(extract_product(synthesized_video))
    scores["identity"] = cosine_similarity(target_feat, synth_product_feat)

    # 时序连贯性: 光流方差
    flow_variance = compute_flow_variance(synthesized_video)
    scores["temporal"] = 1.0 / (1.0 + flow_variance)

    # VLM 评估
    vlm_prompt = """
    评估这个视频的质量 (1-5分):
    1. 商品是否清晰可辨？
    2. 运动是否自然流畅？
    3. 光照是否协调？
    4. 整体是否真实？
    """
    scores["vlm"] = vlm_evaluate(synthesized_video, vlm_prompt)

    # 综合分数
    final_score = (
        0.3 * scores["identity"] +
        0.2 * scores["temporal"] +
        0.2 * scores["vlm"]["lighting"] +
        0.3 * scores["vlm"]["overall"]
    )

    return final_score, scores
```

#### 4.4.3 过滤阈值

```python
QUALITY_THRESHOLD = 0.7  # 保留 70% 以上质量的样本

def filter_samples(samples):
    return [s for s in samples if s["score"] >= QUALITY_THRESHOLD]
```

---

## 5. 数据集格式

### 5.1 目录结构

```
pvtt-training-dataset/
├── videos/
│   ├── source/           # 原始模板视频
│   │   └── {ID}.mp4
│   ├── background/       # Inpainted 背景视频
│   │   └── {ID}_bg.mp4
│   └── synthesized/      # 合成视频
│       └── {SOURCE_ID}_{TARGET_ID}.mp4
│
├── images/
│   ├── source/           # 源商品图
│   │   └── {ID}.jpg
│   └── target/           # 去背景目标商品图
│       └── {ID}_nobg.png
│
├── masks/
│   └── {ID}/
│       ├── 0001.png
│       └── ...
│
├── annotations/
│   ├── pairs.json        # 所有配对
│   ├── quality_scores.json
│   └── statistics.json
│
└── README.md
```

### 5.2 标注格式

**pairs.json**:
```json
{
  "version": "1.0",
  "total_pairs": 2500,
  "pairs": [
    {
      "pair_id": "JEW001_JEW002",
      "source": {
        "id": "JEW001",
        "video": "videos/source/JEW001.mp4",
        "product_image": "images/source/JEW001.jpg",
        "category": "jewelry",
        "subcategory": "necklace"
      },
      "target": {
        "id": "JEW002",
        "product_image": "images/target/JEW002_nobg.png",
        "category": "jewelry",
        "subcategory": "bracelet"
      },
      "output": {
        "video": "videos/synthesized/JEW001_JEW002.mp4",
        "background": "videos/background/JEW001_bg.mp4",
        "masks": "masks/JEW001/"
      },
      "metadata": {
        "difficulty": "medium",
        "quality_score": 0.85,
        "frame_count": 120,
        "resolution": [1280, 720]
      }
    }
  ]
}
```

---

## 6. 资源估算

### 6.1 数据规模预估

| 数据类型 | 原始 | 处理后 | 说明 |
|----------|------|--------|------|
| 视频 | 53 个 | ~200 片段 | 镜头切分 (每视频 3-4 镜头) |
| 商品图片 | ~265 张 | ~265 张 | 每商品 3-8 张 (平均 5 张) |

**配对数量** (取决于图片使用策略):

| 策略 | 计算方式 | 配对数 |
|------|----------|--------|
| 单图 | 200 片段 × 52 商品 × 1 图 | ~10,400 |
| 采样 (2张) | 200 片段 × 52 商品 × 2 图 | ~20,800 |
| 全图 | 200 片段 × 52 商品 × 5 图 | ~52,000 |

### 6.2 计算资源 (以采样策略 ~20K 对为例)

| 阶段 | 单样本耗时 | 总耗时 | GPU 需求 | 显存 |
|------|-----------|--------|----------|------|
| 镜头切分 | ~2s | ~2 min (53视频) | CPU | - |
| SAM2 分割 | ~4s (50帧) | ~13 min (200片段) | 1× A100 | 8 GB |
| ProPainter 修复 | ~25s (50帧) | ~83 min (200片段) | 1× A100 | 10 GB |
| 商品去背景 | ~5s | ~22 min (265张) | CPU | - |
| VideoAnyDoor | ~1 min | ~350 hours (20800对) | 1× A100 | 16 GB |
| 质量评估 | ~10s | ~58 hours (20800对) | 1× A100 | 8 GB |

**总计**: ~410 GPU-hours (单卡 A100，采样策略)

> 性能数据来源: [SAM2 分析](papers/sam2-analysis.md), [ProPainter 分析](papers/propainter-analysis.md)

### 6.3 存储资源

| 数据类型 | 单样本大小 | 总大小估算 |
|----------|-----------|-----------|
| 源视频片段 | ~0.8 MB | 160 MB (200片段) |
| 背景视频 | ~0.8 MB | 160 MB (200片段) |
| 合成视频 | ~0.8 MB | 17 GB (20800对) |
| Mask 序列 | ~5 MB | 1 GB (200片段) |
| 商品图片 | ~0.5 MB | 133 MB (265张) |

**总计**: ~18 GB (采样策略)

### 6.4 扩展性

| 规模 | 商品数 | 片段数 | 图片数 | 配对数 (采样2张) | GPU-hours | 存储 |
|------|--------|--------|--------|-----------------|-----------|------|
| PoC | 10 | ~40 | ~50 | ~3,600 | ~65 | 3 GB |
| Small | 53 | ~200 | ~265 | ~20,800 | ~410 | 18 GB |
| Medium | 100 | ~400 | ~500 | ~79,200 | ~1,500 | 65 GB |
| Large | 500 | ~2,000 | ~2,500 | ~1,996,000 | ~35,000 | 1.6 TB |

---

## 7. 实施计划

### Phase 1: PoC 验证 (1-2 天)

**目标**: 验证 Pipeline 可行性，选择最优模型组合

**范围**: 10 个视频 → ~40 个片段 → ~1,500 个配对

**验证点**:
- [ ] 镜头切分效果 (PySceneDetect 阈值调优)
- [ ] SAM2 分割质量
- [ ] Video Inpainting 对比: VideoPainter vs ProPainter
- [ ] Video Object Insertion 对比: **VideoAnyDoor vs InsertAnywhere**
- [ ] 光照适配是否自然
- [ ] 端到端 Pipeline 可跑通

**模型选择决策**:
| 组件 | 候选 A | 候选 B | 选择标准 |
|------|--------|--------|----------|
| Inpainting | VideoPainter | ProPainter | 质量、速度、部署难度 |
| Insertion | VideoAnyDoor | InsertAnywhere | 商品细节保持、光照适配 |

### Phase 2: 小规模构建 (1-2 周)

**目标**: 构建完整的 53 视频数据集

**范围**: 53 个视频 → ~200 个片段 → ~30,000 个有效配对

**任务**:
- [ ] 全量镜头切分
- [ ] 全量预处理 (分割 + Inpainting)
- [ ] 全量配对生成 (~770 GPU-hours)
- [ ] 质量过滤
- [ ] 数据集打包

### Phase 3: 扩展收集 (持续)

**目标**: 扩展到 100+ 视频

**任务**:
- [ ] 继续收集 Etsy 数据
- [ ] 平衡各品类样本
- [ ] 增加困难样本 (wearing, interaction)
- [ ] 优化镜头切分策略 (过滤无商品镜头)

---

## 8. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 镜头切分过细 | 片段过短无法使用 | 设置 min_scene_len 阈值 |
| 镜头切分漏检 | 多镜头混入单片段 | 降低检测阈值 / 人工复核 |
| 部分镜头无商品 | 无效片段 | 商品检测过滤 (Grounding DINO) |
| SAM2 分割不准确 | Mask 质量差 | 人工校正 / 换用 GroundedSAM |
| Inpainting 留痕 | 背景不干净 | 多模型集成 / 手动筛选 |
| VideoAnyDoor 细节丢失 | 商品不清晰 | 调整 guidance_scale / 换用 InsertAnywhere |
| 光照不匹配 | 合成不自然 | 添加光照估计模块 |
| 配对不合理 | 训练效果差 | 难度分级 / 渐进训练 |

---

## 9. 后续优化方向

1. **数据增强**: 对合成视频做随机裁剪、颜色抖动
2. **难例挖掘**: 重点生成困难配对 (跨品类、大尺寸差异)
3. **人工标注**: 对低质量样本人工标注重新生成
4. **多模型集成**: 对比 VideoAnyDoor vs 其他方法

---

## 10. 参考资料

### 论文

| 论文 | 用途 | 发表 | 链接 |
|------|------|------|------|
| **VideoPainter** | 视频修复 | SIGGRAPH 2025 | [GitHub](https://github.com/TencentARC/VideoPainter) |
| **VideoAnyDoor** | 视频物体插入 | SIGGRAPH 2025 | [GitHub](https://github.com/yuanpengtu/VideoAnydoor) / [分析](papers/videoanydoor-analysis.md) |
| **InsertAnywhere** | 视频物体插入 | arXiv Dec 2025 | [GitHub](https://github.com/myyzzzoooo/InsertAnywhere) / [论文](https://arxiv.org/abs/2512.17504) |
| **SAM2** | 视频分割 | Meta 2024 | [GitHub](https://github.com/facebookresearch/segment-anything-2) / [分析](papers/sam2-analysis.md) |
| **ProPainter** | 视频修复 (备选) | ICCV 2023 | [GitHub](https://github.com/sczhou/ProPainter) / [分析](papers/propainter-analysis.md) |

### 代码仓库

- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) - 镜头切分
- [SAM2](https://github.com/facebookresearch/segment-anything-2) - 视频分割
- [VideoPainter](https://github.com/TencentARC/VideoPainter) - 视频修复 (推荐)
- [VideoAnyDoor](https://github.com/yuanpengtu/VideoAnydoor) - 视频物体插入
- [InsertAnywhere](https://github.com/myyzzzoooo/InsertAnywhere) - 视频物体插入 (备选)
- [rembg](https://github.com/danielgatis/rembg) - 图像去背景

### 内部文档

- [papers/README.md](papers/README.md) - 论文分析索引
- [papers/video-object-insertion.md](papers/video-object-insertion.md) - Video Object Insertion 综合调研

---

Last updated: 2026-01-20
