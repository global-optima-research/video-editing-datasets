# Motion LoRA 技术详解

基于 I2VEdit (SIGGRAPH Asia 2024) 的运动保持技术分析。

---

## 核心思想

视频扩散模型中，不同类型的注意力层负责不同的信息：

| 注意力类型 | 负责内容 | 作用范围 |
|-----------|----------|----------|
| Spatial Attention | 外观、纹理、结构 | 单帧内 |
| **Temporal Attention** | **运动、时序关系** | **跨帧** |

**Motion LoRA 的核心洞察**：只需在 Temporal Attention 层插入 LoRA 并微调，就能捕获源视频的运动模式。

---

## 技术原理

### Temporal Attention 的作用

```
帧1特征 ──┐
帧2特征 ──┼──→ Temporal Attention ──→ 融合后的时序特征
帧3特征 ──┘

Query: 当前帧 "我要查询其他帧的什么信息？"
Key:   所有帧 "每帧有什么信息可以被查询？"
Value: 所有帧 "查询到后返回什么信息？"
```

Temporal Attention 学到的是：
- 哪些空间位置在运动
- 运动的方向和速度
- 运动的时序节奏（加速、减速）
- 帧间的对应关系

### LoRA (Low-Rank Adaptation)

LoRA 是一种参数高效的微调方法：

```
原始权重 W ∈ R^(d×k)
           ↓
W' = W + ΔW = W + BA

其中:
B ∈ R^(d×r), A ∈ R^(r×k), r << min(d,k)
```

**优势**：
- 只训练 A 和 B，参数量极小
- 原始模型权重不变
- 可以保存多个 LoRA，按需加载

### Motion LoRA 训练

```python
# 伪代码
for temporal_attention_layer in model.temporal_attentions:
    # 插入 LoRA 到 Q, K, V 投影层
    temporal_attention_layer.q_proj = LoRALayer(temporal_attention_layer.q_proj)
    temporal_attention_layer.k_proj = LoRALayer(temporal_attention_layer.k_proj)
    temporal_attention_layer.v_proj = LoRALayer(temporal_attention_layer.v_proj)

# 训练目标: 重建源视频
loss = MSE(denoise(noisy_source_video), source_video)
```

训练 Loss：

```
L_motion = E[||z₀ - z_θ(z_t, σ, t, c)||²]

z₀: 源视频的 latent
z_t: 加噪后的 latent
z_θ: 模型预测的去噪结果
```

---

## 完整流程

### 阶段 1: 运动提取 (Training)

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  源视频 (16帧)                                              │
│      ↓                                                      │
│  VAE Encoder → Latent 表示                                  │
│      ↓                                                      │
│  加噪 → Noisy Latent                                        │
│      ↓                                                      │
│  I2V UNet (只训练 Temporal Attention 的 LoRA)               │
│      ↓                                                      │
│  去噪预测                                                   │
│      ↓                                                      │
│  Loss = MSE(预测, 源视频 Latent)                            │
│      ↓                                                      │
│  保存 Motion LoRA 权重                                      │
│                                                             │
│  耗时: ~25 分钟 (A100), ~250 iterations                     │
│  LoRA rank: 通常 r=16 或 r=32                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 阶段 2: 编辑传播 (Inference)

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  编辑后首帧 (用任意图像编辑工具处理)                         │
│      ↓                                                      │
│  I2V 模型 + 加载 Motion LoRA                                │
│      ↓                                                      │
│  生成后续帧                                                 │
│      ↓                                                      │
│  输出: 保持源视频运动的编辑后视频                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 关键技术细节

### 1. 只用 Temporal Attention，不用 Spatial Attention

I2VEdit 论文指出，在 Spatial Attention 加 LoRA 会导致训练不稳定：

> "We observed that adding LoRA to spatial attention layers destabilized training."

原因：Spatial Attention 负责外观，微调它会导致过拟合到源视频的外观，破坏编辑效果。

### 2. Skip-Interval Cross-Attention (长视频)

对于超过 16 帧的视频，需要分 clip 自回归生成：

```
Clip 1 (帧 1-16):  训练 Motion LoRA，保存 K/V 矩阵
      ↓
Clip 2 (帧 17-32): 训练时，将 Clip 1 的 K/V 拼接到当前 Temporal Attention
      ↓
效果: 后续 clip 能引用首帧信息，减少累积误差
```

### 3. Smooth Area Random Perturbation

纯色背景区域（如白墙）会导致反演伪影：

```
问题: UNet 训练时从未见过无噪声的平滑区域
解决: 对平滑区域添加微小扰动 (α=0.005)
效果: 反演得到更高斯分布的 latent
```

---

## 与其他方法对比

| 方法 | 运动表示 | 训练需求 | 形状变化 |
|------|----------|----------|----------|
| **Motion LoRA** | 隐式 (注意力权重) | 每视频单独训练 | 有限 |
| 光流引导 | 显式 (像素位移) | 无需训练 | 不支持 |
| 语义点 | 显式 (稀疏轨迹) | 需要标注 | 支持 |
| 深度约束 | 显式 (几何结构) | 无需训练 | 不支持 |

### Motion LoRA 的优势

1. **隐式学习**: 自动捕获复杂运动，无需手动定义
2. **灵活性**: 可以捕获相机运动、物体运动、光影变化
3. **质量高**: 重建质量好，时序一致性强

### Motion LoRA 的局限

1. **成本高**: 每视频需 25 分钟训练
2. **不可迁移**: 每个 LoRA 只适用于对应的源视频
3. **形状约束**: 倾向保持源视频的空间结构

---

## 对 PVTT 的适用性分析

### 适合的场景

- 相似形状的商品替换 (手机A → 手机B)
- 需要高质量运动保持
- 视频数量较少，可接受逐个训练

### 不适合的场景

- 大规模数据集构建 (成本过高)
- 需要形状变化的替换
- 需要实时/快速处理

### 可能的优化方向

1. **通用 Motion LoRA**: 在大量视频上预训练，学习通用运动模式
2. **Motion Encoder**: 用 encoder 直接提取运动特征，无需微调
3. **混合方案**: Motion LoRA 提取粗运动 + BBox 轨迹处理形状变化

---

## 应用到 Wan2.1 / Wan2.2

### 架构差异

| 特性 | SVD (I2VEdit) | Wan2.1/2.2 |
|------|---------------|------------|
| 骨架 | UNet | DiT (Diffusion Transformer) |
| 范式 | DDPM | Flow Matching |
| 时序建模 | Temporal Attention 层 | Transformer Block 内的时序注意力 |
| 文本编码 | CLIP | T5 Encoder + Cross-Attention |
| VAE | 2D VAE | **3D Causal VAE (Wan-VAE)** |

### Wan 的注意力结构

```
Wan Transformer Block:
├── Spatial Self-Attention (帧内)
├── Temporal Self-Attention (跨帧) ← Motion LoRA 插入点
├── Cross-Attention (文本条件)
└── FFN
```

### Motion LoRA 在 Wan 上的实现

```python
# 伪代码: 在 Wan 的 Temporal Attention 插入 LoRA
for block in wan_model.transformer_blocks:
    # 只对 Temporal Self-Attention 加 LoRA
    temporal_attn = block.temporal_self_attention
    temporal_attn.q_proj = LoRALayer(temporal_attn.q_proj, rank=256)
    temporal_attn.k_proj = LoRALayer(temporal_attn.k_proj, rank=256)
    temporal_attn.v_proj = LoRALayer(temporal_attn.v_proj, rank=256)

    # Spatial Attention 和 Cross-Attention 不动
```

### 推荐配置

根据 Wan LoRA 微调实践:

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| LoRA Rank | 256-512 | Wan 论文建议 512 效果更好 |
| 目标层 | Temporal Self-Attention | 只训练时序注意力 |
| Learning Rate | 1e-4 ~ 5e-5 | 标准 LoRA 学习率 |
| 训练步数 | 200-500 | 单视频微调 |

### 工具支持

| 工具 | 支持情况 | 链接 |
|------|----------|------|
| **DiffSynth-Studio** | ✅ 官方支持 LoRA 训练 | [GitHub](https://github.com/modelscope/DiffSynth-Studio) |
| **Wan2GP** | ✅ 低显存优化 | [GitHub](https://github.com/deepbeepmeep/Wan2GP) |
| ComfyUI | ✅ LoRA 加载 | [Docs](https://docs.comfy.org/tutorials/video/wan/wan2_2) |

### 关键注意事项

1. **3D VAE 差异**: Wan-VAE 是 3D 因果 VAE，时序压缩比 4×，空间 8×。Latent 已包含时序信息。

2. **Flow Matching vs DDPM**: Wan 用 Flow Matching，训练目标是速度场而非噪声预测:
   ```
   DDPM Loss: ||ε - ε_θ(x_t, t)||²
   Flow Loss: ||v - v_θ(x_t, t)||²  (v = dx/dt)
   ```

3. **Wan2.2 MoE 架构**: Wan2.2 使用 Mixture of Experts，高噪声和低噪声各一个专家。LoRA 需要对两个专家都加，或只加低噪声专家（细节生成）。

4. **兼容性**: Wan2.1 训练的 LoRA 可直接用于 Wan2.2 低噪声模型。

### 示例: DiffSynth-Studio 训练 Motion LoRA

```bash
# 安装
pip install diffsynth

# 训练脚本 (简化)
python train_lora.py \
    --model Wan-AI/Wan2.1-I2V-14B-720P \
    --video_path source_video.mp4 \
    --output_dir ./motion_lora \
    --lora_rank 256 \
    --target_modules temporal_attn \
    --learning_rate 1e-4 \
    --max_steps 300
```

### 待验证问题

1. Wan 的 Temporal Attention 结构是否与 SVD 一致？
2. 3D VAE 的时序编码是否影响 Motion LoRA 效果？
3. Flow Matching 范式下，最优训练步数？

---

## 代码资源

- **官方实现**: [github.com/Vicky0522/I2VEdit](https://github.com/Vicky0522/I2VEdit)
- **论文**: [arxiv.org/abs/2405.16537](https://arxiv.org/abs/2405.16537)
- **项目主页**: [i2vedit.github.io](https://i2vedit.github.io/)
- **Wan2.1**: [github.com/Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1)
- **Wan2.2**: [github.com/Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)
- **DiffSynth-Studio**: [github.com/modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)

---

## 参考文献

- [I2VEdit: First-Frame-Guided Video Editing via Image-to-Video Diffusion Models](https://arxiv.org/abs/2405.16537) - SIGGRAPH Asia 2024
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) - ICLR 2022
- [Stable Video Diffusion](https://stability.ai/stable-video) - Stability AI

---

Last updated: 2026-01-20
