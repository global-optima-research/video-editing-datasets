# PVTT 训练数据集构建技术方案

Product Video Template Transfer 训练数据集构建方案。

---

## 1. 目标

构建用于训练 PVTT 模型的大规模视频编辑数据集。

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
| 模板视频 | 53 个 | Etsy 商品展示视频 |
| 商品图片 | 53 张 | 对应商品的产品图 |
| 品类 | 11 个 | Jewelry, Toys, Home, Clothing 等 |

### 2.2 交叉配对策略

利用现有数据交叉配对生成训练样本：

```
样本 i 的视频 + 样本 j 的商品图 → 训练对 (i, j)

配对数量: N × (N-1) = 53 × 52 = 2,756 对
```

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
│  │  模板视频 V_i                                            │    │
│  │       ↓                                                  │    │
│  │  SAM2 分割 → 商品 Mask 序列 M_i                          │    │
│  │       ↓                                                  │    │
│  │  Video Inpainting → 干净背景视频 B_i                     │    │
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

#### 4.1.1 商品分割 (SAM2)

**工具**: Segment Anything Model 2 (SAM2)

**输入**: 模板视频 V_i

**输出**: 逐帧 Mask 序列 M_i = {m_1, m_2, ..., m_T}

**流程**:
```python
# 伪代码
1. 首帧自动检测商品 (或使用 Grounding DINO 定位)
2. SAM2 分割首帧商品区域
3. SAM2 视频模式自动传播到所有帧
4. 输出: 每帧的二值 Mask
```

**注意事项**:
- 处理遮挡情况 (商品被手遮挡)
- 处理多商品情况 (选择主商品)
- Mask 平滑处理 (避免帧间抖动)

#### 4.1.2 背景修复 (Video Inpainting)

**工具选项**:
| 工具 | 优势 | 劣势 |
|------|------|------|
| ProPainter | 开源，效果好 | 速度较慢 |
| E2FGVI | 快速 | 大区域效果一般 |
| STTN | 平衡 | 需要调参 |

**输入**: 模板视频 V_i + Mask 序列 M_i

**输出**: 干净背景视频 B_i (商品区域被背景填充)

**流程**:
```python
# 伪代码
1. 扩展 Mask (膨胀操作，覆盖商品边缘和阴影)
2. Video Inpainting 模型填充 Mask 区域
3. 时序一致性检查
4. 输出: 无商品的背景视频
```

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

#### 4.3.1 VideoAnyDoor 调用

**输入**:
- `background_video`: 干净背景视频 B_i
- `reference_image`: 去背景商品图 T_j'
- `mask_sequence`: Mask 序列 M_i (定义插入位置)

**输出**: 合成视频 V'_ij

**配置**:
```python
config = {
    "model": "VideoAnyDoor",
    "guidance_scale": 7.5,
    "num_inference_steps": 50,
    "control_mode": "box",  # 使用 bbox 而非关键点
}
```

#### 4.3.2 Mask 到 Box 转换

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

#### 4.3.3 尺寸适配

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

### 6.1 计算资源

| 阶段 | 单样本耗时 | 53 样本总耗时 | GPU 需求 |
|------|-----------|--------------|----------|
| SAM2 分割 | ~30s | ~30 min | 1× A100 |
| Video Inpainting | ~2 min | ~2 hours | 1× A100 |
| 商品去背景 | ~5s | ~5 min | CPU |
| VideoAnyDoor | ~1 min | ~45 hours (2756对) | 1× A100 |
| 质量评估 | ~10s | ~8 hours | 1× A100 |

**总计**: ~55 GPU-hours (单卡 A100)

### 6.2 存储资源

| 数据类型 | 单样本大小 | 总大小估算 |
|----------|-----------|-----------|
| 源视频 | ~1.6 MB | 85 MB |
| 背景视频 | ~1.6 MB | 85 MB |
| 合成视频 | ~1.6 MB | 4.4 GB (2756对) |
| Mask 序列 | ~10 MB | 530 MB |
| 商品图片 | ~0.5 MB | 27 MB |

**总计**: ~5.1 GB

### 6.3 扩展性

| 规模 | 源样本 | 配对数 | GPU-hours | 存储 |
|------|--------|--------|-----------|------|
| PoC | 10 | 90 | ~2 | 200 MB |
| Small | 53 | 2,756 | ~55 | 5 GB |
| Medium | 200 | 39,800 | ~800 | 70 GB |
| Large | 1,000 | 999,000 | ~20,000 | 1.7 TB |

---

## 7. 实施计划

### Phase 1: PoC 验证 (1-2 天)

**目标**: 验证 Pipeline 可行性

**范围**: 10 个样本，90 个配对

**验证点**:
- [ ] SAM2 分割质量
- [ ] Video Inpainting 效果
- [ ] VideoAnyDoor 商品细节保持
- [ ] 光照适配是否自然
- [ ] 端到端 Pipeline 可跑通

### Phase 2: 小规模构建 (3-5 天)

**目标**: 构建完整的 53 样本数据集

**范围**: 53 个样本，~2,500 个有效配对

**任务**:
- [ ] 全量预处理 (分割 + Inpainting)
- [ ] 全量配对生成
- [ ] 质量过滤
- [ ] 数据集打包

### Phase 3: 扩展收集 (持续)

**目标**: 扩展到 200+ 样本

**任务**:
- [ ] 继续收集 Etsy 数据
- [ ] 平衡各品类样本
- [ ] 增加困难样本 (wearing, interaction)

---

## 8. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| SAM2 分割不准确 | Mask 质量差 | 人工校正 / 换用 GroundedSAM |
| Inpainting 留痕 | 背景不干净 | 多模型集成 / 手动筛选 |
| VideoAnyDoor 细节丢失 | 商品不清晰 | 调整 guidance_scale / 后处理 |
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

- [VideoAnyDoor](https://arxiv.org/abs/2501.01427) - SIGGRAPH 2025
- [SAM2](https://github.com/facebookresearch/segment-anything-2) - Meta
- [ProPainter](https://github.com/sczhou/ProPainter) - Video Inpainting
- [rembg](https://github.com/danielgatis/rembg) - 图像去背景

---

Last updated: 2026-01-20
