# ProductVideoGenerator 实验方案

## 实验信息

- **日期**: 2026-01-23
- **目标**: 设计并验证"多视角商品图片 + masked_video → 商品视频"的生成方案
- **背景**: 基于 VideoPainter 数据构造思路，设计 PVTT 数据集自动化生成方案
- **基座模型**: Wan2.1-1.3B（PoC 阶段使用小模型快速验证，成功后迁移到 14B）

---

## 一、任务定义

### 1.1 输入输出

```
输入：
├── masked_video = original_video ⊙ (1 - mask)   # 商品区域置零
├── mask_sequence                                 # 二值 mask (T×H×W)
├── product_images                                # N张商品图片 (N=3-5)
└── caption (可选)                                # 文本描述

输出：
└── generated_video                               # 生成的商品视频

GT：
└── original_video                                # 原始完整视频
```

### 1.2 核心思想

借鉴 VideoPainter 的数据构造方式：
- VideoPainter：masked_video → 恢复原物体（移除任务）
- 我们：masked_video + 商品图片 → 生成商品视频（插入任务）

关键区别：我们额外输入独立的商品图片（多视角），模型需要学会把商品"放入"视频中。

### 1.3 方案优势

| 优势 | 说明 |
|------|------|
| 无需空镜视频 | 直接用 masked_video，省去 inpainting 步骤 |
| 位置明确 | mask 精确指定商品位置 |
| 多视角输入 | 多张图片提供 3D 结构信息，提升泛化能力 |
| 数据易获取 | 电商平台天然有 视频+图片 配对 |

---

## 二、数据处理 Pipeline

### 2.1 数据来源

```
电商平台商品页面：
├── 商品展示视频 (15-60s)
│   ├── 要求：单个商品、干净背景、有运镜
│   └── 来源：商品详情页视频
│
└── 商品图片集 (3-5张)
    ├── 主图 (正面)
    ├── 细节图 (侧面/背面/特写)
    └── 场景图 (可选)
```

### 2.2 处理流程

```
Step 1: 视频预处理
├── PySceneDetect 切分场景
├── 筛选：3-10s, ≥720p, 单商品
├── 统一：resize to 480×720, fps=24
└── 输出：video_clips/{product_id}_{clip_id}.mp4

Step 2: 商品分割
├── Grounding DINO: 检测商品边界框
├── SAM2: 生成 mask 序列
├── 后处理：时序平滑、去噪
└── 输出：masks/{product_id}_{clip_id}/ (T张 png)

Step 3: 生成 masked_video
├── masked = video * (1 - mask)
├── 商品区域置零（或随机噪声）
└── 输出：masked_videos/{product_id}_{clip_id}.mp4

Step 4: 商品图片处理
├── 收集 3-5 张不同角度图片
├── 去背景 (rembg / SAM)
├── Resize to 512×512, 保持宽高比
└── 输出：product_images/{product_id}/ (N张 png)

Step 5: 构建数据集
└── dataset.json
```

### 2.3 数据集格式

```json
{
  "train": [
    {
      "product_id": "JEWE001",
      "clip_id": "001",
      "video_path": "videos/JEWE001_001.mp4",
      "masked_video_path": "masked/JEWE001_001.mp4",
      "mask_dir": "masks/JEWE001_001/",
      "product_images": [
        "product_images/JEWE001/front.png",
        "product_images/JEWE001/side.png",
        "product_images/JEWE001/detail.png"
      ],
      "caption": "silver bracelet with gemstones on purple velvet",
      "num_frames": 72,
      "fps": 24
    }
  ],
  "val": [...],
  "test": [...]
}
```

### 2.4 数据规模

| 阶段 | 商品数 | 视频片段 | 目的 |
|------|--------|---------|------|
| PoC | 100-200 | 500-1000 | 验证可行性 |
| 小规模 | 1K-5K | 5K-20K | 训练初版模型 |
| 完整版 | 10K+ | 50K+ | 生产级模型 |

---

## 三、模型架构

### 3.0 基座模型选择

**PoC 阶段使用 Wan2.1-1.3B**：

| | Wan2.1-1.3B | Wan2.1-14B |
|---|---|---|
| 显存需求 | ~8-12GB | ~32-40GB |
| 训练速度 | 快 | 慢 |
| 推理速度 | 快 | 慢 |
| 生成质量 | 够用验证 | 更好 |
| 适用阶段 | PoC | 生产 |

**迁移策略**：
```
PoC (1.3B) → 验证成功 → 迁移到 14B → 正式训练
```

### 3.1 整体架构

```
┌──────────────────────────────────────────────────────────────────┐
│                      ProductVideoGenerator                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                  Product Encoder                             │ │
│  │                                                               │ │
│  │   product_images ──→ Wan2.1 Image Encoder ──→ image_tokens  │ │
│  │        (N张)              (复用，冻结)            (N×L×D)    │ │
│  │                        ↓                                     │ │
│  │              Multi-Image Attention                           │ │
│  │                        ↓                                     │ │
│  │              product_embedding (1×K×D)                       │ │
│  │              (K=64-256 tokens)                               │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              ↓                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                  Video Generator (DiT)                       │ │
│  │                                                               │ │
│  │   masked_video ──→ VAE Encoder ──→ z_masked (T×H×W×C)       │ │
│  │                                                               │ │
│  │   mask ──→ Downsample ──→ m_latent (T×H×W×1)                │ │
│  │                                                               │ │
│  │   noise ──→ z_t                                              │ │
│  │                                                               │ │
│  │   DiT Input = concat([z_t, z_masked, m_latent], dim=-1)      │ │
│  │                                                               │ │
│  │   ┌──────────────────────────────────────────────────────┐  │ │
│  │   │  DiT Block (×N)                                       │  │ │
│  │   │  ├── Self-Attention (时空)                            │  │ │
│  │   │  ├── Cross-Attention (product_embedding) ← 商品注入   │  │ │
│  │   │  └── FFN                                              │  │ │
│  │   └──────────────────────────────────────────────────────┘  │ │
│  │                         ↓                                    │ │
│  │   z_0 ──→ VAE Decoder ──→ generated_video                   │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 Product Encoder

**目标**：从多张商品图片中提取紧凑的商品表示

**设计原则**：复用 Wan2.1 已有的 Image Encoder，只新增多图融合模块

```python
class ProductEncoder(nn.Module):
    """
    输入: N张商品图片 (B, N, 3, 512, 512)
    输出: product_embedding (B, K, D)

    复用 Wan2.1 的 Image Encoder，不引入额外的 DINOv2
    """
    def __init__(self, wan_image_encoder, num_tokens=64, hidden_dim=1024):
        # 复用 Wan2.1 的 Image Encoder（冻结）
        self.image_encoder = wan_image_encoder

        # 可学习的 query tokens（用于聚合多张图片信息）
        self.query_tokens = nn.Parameter(torch.randn(num_tokens, hidden_dim))

        # Multi-Image Attention: 融合多张图片
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=16)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, images):
        # images: (B, N, 3, H, W)
        B, N = images.shape[:2]

        # 用 Wan2.1 的编码器提取每张图片特征
        image_tokens = []
        for i in range(N):
            tokens = self.image_encoder(images[:, i])  # (B, L, D)
            image_tokens.append(tokens)

        # 拼接所有图片 tokens
        all_tokens = torch.cat(image_tokens, dim=1)  # (B, N*L, D)

        # 用 query tokens 聚合多图信息
        queries = self.query_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, K, D)
        output, _ = self.cross_attn(queries, all_tokens, all_tokens)
        output = self.norm(output + queries)

        return output  # (B, K, D) - 商品的紧凑表示
```

**优势**：
- 不引入额外的大模型（DINOv2 ~300M 参数）
- 特征空间与 Wan2.1 一致，更容易融合
- 只需训练轻量级的多图融合模块

### 3.3 DiT 条件注入

**目标**：将商品特征注入到视频生成过程中

```python
class DiTBlockWithProduct(nn.Module):
    """
    在标准 DiT Block 中加入商品特征的 Cross Attention
    """
    def __init__(self, hidden_dim, num_heads):
        # 原有层
        self.self_attn = SpatioTemporalAttention(hidden_dim, num_heads)
        self.ffn = FeedForward(hidden_dim)

        # 新增：商品特征注入
        self.product_cross_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.product_norm = nn.LayerNorm(hidden_dim)
        self.product_gate = nn.Parameter(torch.zeros(1))  # 可学习的门控，初始化为0

    def forward(self, x, product_embedding, timestep):
        # x: (B, T*H*W, D) - 视频 latent tokens
        # product_embedding: (B, K, D) - 商品特征

        # 1. Self-Attention (时空)
        x = x + self.self_attn(x, timestep)

        # 2. Cross-Attention (商品特征注入)
        product_out, _ = self.product_cross_attn(x, product_embedding, product_embedding)
        x = x + self.product_gate * self.product_norm(product_out)

        # 3. FFN
        x = x + self.ffn(x)

        return x
```

### 3.4 完整前向传播

```python
class ProductVideoGenerator(nn.Module):
    def __init__(self):
        self.vae = AutoencoderKL.from_pretrained(...)       # 冻结
        self.product_encoder = ProductEncoder()              # 可训练
        self.dit = DiTWithProductInjection.from_pretrained() # 部分可训练

    def forward(self, video, masked_video, mask, product_images, timestep):
        # 1. 编码商品图片
        product_embedding = self.product_encoder(product_images)  # (B, K, D)

        # 2. VAE 编码视频
        z_gt = self.vae.encode(video)                # (B, T, H, W, C)
        z_masked = self.vae.encode(masked_video)     # (B, T, H, W, C)

        # 3. Mask 下采样
        mask_latent = F.interpolate(mask, size=z_gt.shape[1:4])  # (B, T, H, W, 1)

        # 4. 加噪声
        noise = torch.randn_like(z_gt)
        z_t = self.scheduler.add_noise(z_gt, noise, timestep)

        # 5. DiT 输入
        dit_input = torch.cat([z_t, z_masked, mask_latent], dim=-1)

        # 6. DiT 前向（带商品条件）
        noise_pred = self.dit(dit_input, timestep, product_embedding)

        return noise_pred, noise
```

---

## 四、训练策略

### 4.1 两阶段训练

**Stage 1: 训练 Adapter（冻结 DiT）**

```yaml
stage1:
  epochs: 50
  batch_size: 4
  learning_rate: 1e-4

  frozen:
    - vae
    - dit.blocks.*  # 冻结所有 DiT 原有参数
    - product_encoder.image_encoder  # 冻结 Wan2.1 Image Encoder

  trainable:
    - product_encoder.query_tokens
    - product_encoder.cross_attn
    - product_encoder.norm
    - dit.blocks.*.product_cross_attn  # 只训练新增的 cross attention
    - dit.blocks.*.product_gate
```

**Stage 2: 微调 DiT**

```yaml
stage2:
  epochs: 100
  batch_size: 2
  learning_rate: 1e-5

  frozen:
    - vae
    - product_encoder.image_encoder

  trainable:
    - product_encoder.*  # 除了 image_encoder
    - dit.blocks.*       # 全部 DiT
```

### 4.2 损失函数

```python
def compute_loss(model, batch):
    noise_pred, noise = model(
        batch['video'],
        batch['masked_video'],
        batch['mask'],
        batch['product_images'],
        batch['timestep']
    )

    # 1. 重建损失 (主要)
    loss_recon = F.mse_loss(noise_pred, noise)

    # 2. 感知损失 (可选，需要解码)
    # loss_perceptual = lpips_loss(decoded_video, batch['video'])

    return loss_recon
```

### 4.3 训练配置

```yaml
# config.yaml

model:
  base: "Wan2.1-1.3B"
  product_encoder:
    backbone: "wan2.1_image_encoder"  # 复用 Wan2.1 的 Image Encoder
    num_tokens: 64
    hidden_dim: 1024
    freeze_backbone: true

  dit_modification:
    inject_layers: "all"
    gate_init: 0.0

data:
  video_size: [49, 480, 720]  # T, H, W
  num_product_images: 4

optimizer:
  type: "AdamW"
  betas: [0.9, 0.999]
  weight_decay: 0.01

scheduler:
  type: "cosine"
  warmup_steps: 1000
```

---

## 五、推理流程

```python
@torch.no_grad()
def generate(model, masked_video, mask, product_images, num_steps=50):
    """
    生成商品视频

    Args:
        masked_video: (1, T, 3, H, W) - 被 mask 的视频
        mask: (1, T, 1, H, W) - mask 序列
        product_images: (1, N, 3, 512, 512) - 商品图片
        num_steps: 去噪步数

    Returns:
        output_video: (1, T, 3, H, W) - 生成的视频
    """
    # 1. 编码商品
    product_embedding = model.product_encoder(product_images)

    # 2. 编码 masked video
    z_masked = model.vae.encode(masked_video)
    mask_latent = F.interpolate(mask, size=z_masked.shape[1:4])

    # 3. 初始化噪声
    z_t = torch.randn_like(z_masked)

    # 4. DDIM 去噪
    for t in tqdm(reversed(range(num_steps))):
        timestep = torch.tensor([t])
        dit_input = torch.cat([z_t, z_masked, mask_latent], dim=-1)
        noise_pred = model.dit(dit_input, timestep, product_embedding)
        z_t = model.scheduler.step(noise_pred, t, z_t)

    # 5. VAE 解码
    generated_video = model.vae.decode(z_t)

    # 6. 合成最终输出
    # mask 区域用生成的，其他区域用原始的
    output_video = generated_video * mask + masked_video * (1 - mask)

    return output_video
```

---

## 六、评估指标

### 6.1 重建质量

| 指标 | 说明 | 目标 |
|------|------|------|
| PSNR | 峰值信噪比 | > 25 dB |
| SSIM | 结构相似性 | > 0.85 |
| LPIPS | 感知损失 | < 0.15 |

### 6.2 商品一致性

| 指标 | 说明 | 目标 |
|------|------|------|
| CLIP-I | 生成帧 vs 输入商品图片 | > 0.8 |
| DINO Score | 商品 ID 保持 | > 0.7 |

### 6.3 视频质量

| 指标 | 说明 | 目标 |
|------|------|------|
| FVD | Fréchet 视频距离 | < 300 |
| 时序一致性 | 光流分析 | 无明显闪烁 |

### 6.4 泛化能力

| 测试 | 说明 | 目标 |
|------|------|------|
| 测试集商品 | 训练时未见的商品 | 人工评分 ≥ 3/5 |
| 跨类别 | 不同商品类别 | 基本可用 |

---

## 七、PoC 实验计划

### Week 1: 数据准备

```
Day 1-2: 数据收集
├── 目标：100 个珠宝商品
├── 每个商品：1-3 个视频 + 3-5 张图片
├── 来源：淘宝/京东商品页
└── 工具：爬虫脚本

Day 3-4: 数据处理
├── SAM2 分割
├── 生成 masked_video
├── 图片去背景
└── 质量检查

Day 5: 数据集构建
├── 构建 dataset.json
├── Train/Val/Test 划分 (8:1:1)
└── 数据加载测试
```

### Week 2: 模型实现

```
Day 1-2: ProductEncoder
├── 复用 Wan2.1 Image Encoder
├── Multi-Image Attention
└── 单元测试

Day 3-4: DiT 修改
├── Cross Attention 注入
├── Gate 机制
└── 与 Wan2.1 集成

Day 5: 训练脚本
├── 数据加载
├── 训练循环
├── 日志 & 可视化
└── 测试前向传播
```

### Week 3: 训练 PoC

```
Day 1-3: Stage 1 训练
├── 冻结 DiT
├── 训练 adapter
├── 监控 loss 曲线
└── 每 epoch 生成样例

Day 4-5: Stage 2 训练
├── 微调 DiT
├── 继续监控
└── 保存 checkpoints
```

### Week 4: 评估 & 迭代

```
Day 1-2: 定量评估
├── 计算 PSNR/SSIM/LPIPS
├── 计算 CLIP-I/DINO
└── 计算 FVD

Day 3: 泛化测试
├── 测试集商品生成
├── 人工评估
└── 错误案例分析

Day 4-5: 总结 & 决策
├── 撰写 PoC 报告
├── 分析问题
└── 制定下一步计划
```

---

## 八、决策点

### 8.1 成功标准

| 指标 | PoC 成功标准 |
|------|-------------|
| 训练 Loss | 收敛且稳定下降 |
| 训练集 PSNR | > 25 dB |
| 训练集 CLIP-I | > 0.8 |
| 测试集泛化 | 人工评分 ≥ 3/5 |
| 时序一致性 | 无明显闪烁 |

### 8.2 决策矩阵

| 结果 | 判断 | 下一步 |
|------|------|--------|
| 全部达标 | ✅ 成功 | 扩大数据规模，训练完整模型 |
| 训练好，泛化差 | ⚠️ 过拟合 | 增加数据 / 数据增强 / 正则化 |
| 训练差，loss 收敛 | ⚠️ 容量不足 | 增加 adapter 参数 / 解冻更多层 |
| Loss 不收敛 | ❌ 失败 | 检查数据 / 调整学习率 / 检查代码 |
| 闪烁严重 | ⚠️ 时序问题 | 增加时序损失 / 调整架构 |

---

## 九、风险与备选

### 9.1 风险评估

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| 数据收集困难 | 中 | 高 | 使用公开数据集补充 |
| 训练不收敛 | 低 | 高 | 降低学习率，检查梯度 |
| 泛化能力差 | 中 | 高 | 增加数据多样性 |
| 显存不足 | 中 | 中 | 梯度累积 / 降低分辨率 |
| 时序闪烁 | 中 | 中 | 增加时序损失 |

### 9.2 备选方案

1. **如果 Wan2.1 不适合**：尝试 CogVideoX / AnimateDiff
2. **如果 Wan2.1 Image Encoder 不够**：尝试额外加 DINOv2 / CLIP
3. **如果 Cross Attention 不够**：尝试 Controlnet 风格注入

---

## 十、参考资料

### 论文

- [VideoPainter](https://arxiv.org/abs/2503.05639) - SIGGRAPH 2025
- [VideoAnyDoor](https://arxiv.org/abs/2501.01427) - SIGGRAPH 2025
- [AnyDoor](https://arxiv.org/abs/2307.09481) - CVPR 2024
- [IP-Adapter](https://arxiv.org/abs/2308.06721) - 图像特征注入
- [Wan2.1](https://github.com/Wan-Video/Wan2.1) - 视频生成基座 (阿里通义万相)

### 代码

- VideoPainter: https://github.com/TencentARC/VideoPainter
- Wan2.1: https://github.com/Wan-Video/Wan2.1
- SAM2: https://github.com/facebookresearch/sam2

---

**记录人**: Claude
**最后更新**: 2026-01-23
