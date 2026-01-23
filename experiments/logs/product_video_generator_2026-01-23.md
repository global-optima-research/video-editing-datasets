# ProductVideoGenerator 实验方案

## 实验信息

- **日期**: 2026-01-23
- **目标**: 设计并验证"多视角商品图片 + masked_video → 商品视频"的生成方案
- **背景**: 基于 VideoPainter 数据构造思路，设计 PVTT 数据集自动化生成方案
- **基座模型**: Wan2.1-VACE-1.3B
- **训练框架**: DiffSynth-Studio

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
| 现成框架 | 使用 DiffSynth-Studio，无需从头实现 |

---

## 二、技术选型

### 2.1 为什么选择 Wan2.1-VACE-1.3B

**VACE (Visual-Aware Conditional Encoding)** 已经支持参考图像输入：

```
VACE 原生支持：
├── vace_reference_image  ← 可用于商品图片
├── vace_video            ← 可用于运动参考
└── 两者可组合使用
```

| 对比 | 自己实现 | 使用 VACE |
|------|---------|----------|
| 图像编码器 | 需要实现 ProductEncoder | 已内置 |
| Cross Attention | 需要修改 DiT | 已内置 |
| 训练代码 | 需要从头写 | DiffSynth 现成 |
| 复杂度 | 高 | 低 |

### 2.2 模型规格

| 属性 | 值 |
|------|---|
| 模型名称 | Wan2.1-VACE-1.3B |
| 参数量 | 1.3B |
| 推荐分辨率 | 480×832 |
| 帧数 | 49 帧 (~3.3s @ 15fps) |
| VRAM 需求 | ~8GB |
| 许可证 | Apache 2.0 |

### 2.3 DiffSynth-Studio

```
GitHub: https://github.com/modelscope/DiffSynth-Studio

已有的训练脚本：
├── examples/wanvideo/model_training/lora/Wan2.1-VACE-1.3B.sh
├── examples/wanvideo/model_training/full/Wan2.1-VACE-1.3B.sh
└── examples/wanvideo/model_inference/Wan2.1-VACE-1.3B.py
```

---

## 三、数据处理 Pipeline

### 3.1 数据来源

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

### 3.2 处理流程

```
Step 1: 视频预处理
├── PySceneDetect 切分场景
├── 筛选：3-10s, ≥480p, 单商品
├── 统一：resize to 480×832, fps=15
└── 输出：video_clips/{product_id}_{clip_id}.mp4

Step 2: 商品分割
├── Grounding DINO: 检测商品边界框
├── SAM2: 生成 mask 序列
├── 后处理：时序平滑、去噪
└── 输出：masks/{product_id}_{clip_id}/ (T张 png)

Step 3: 生成 masked_video（可选，训练时动态生成）
├── masked = video * (1 - mask)
├── 商品区域置零
└── 输出：masked_videos/{product_id}_{clip_id}.mp4

Step 4: 商品图片处理
├── 收集 3-5 张不同角度图片
├── 去背景 (rembg / SAM)
├── 拼接成单张图片（如 2x2 grid）
├── Resize to 480×832（与视频同尺寸）
└── 输出：product_images/{product_id}.png

Step 5: 构建 DiffSynth 数据集
└── metadata.csv
```

### 3.3 DiffSynth 数据格式

```csv
prompt,video,vace_reference_image
"silver bracelet jewelry display",videos/JEWE001_001.mp4,product_images/JEWE001.png
"gold necklace product showcase",videos/JEWE002_001.mp4,product_images/JEWE002.png
```

**说明**：
- `video`: GT 视频（原始商品视频）
- `vace_reference_image`: 商品图片（多张拼接成一张）
- `prompt`: 商品描述

### 3.4 数据目录结构

```
data/product_video_dataset/
├── metadata.csv
├── videos/
│   ├── JEWE001_001.mp4
│   ├── JEWE001_002.mp4
│   └── ...
├── masks/
│   ├── JEWE001_001/
│   │   ├── 0000.png
│   │   ├── 0001.png
│   │   └── ...
│   └── ...
└── product_images/
    ├── JEWE001.png  (多图拼接)
    ├── JEWE002.png
    └── ...
```

### 3.5 数据规模

| 阶段 | 商品数 | 视频片段 | 目的 |
|------|--------|---------|------|
| PoC | 100-200 | 500-1000 | 验证可行性 |
| 小规模 | 1K-5K | 5K-20K | 训练初版模型 |
| 完整版 | 10K+ | 50K+ | 生产级模型 |

---

## 四、训练方案

### 4.1 使用 DiffSynth LoRA 训练

**基于现有脚本修改**：

```bash
# 基于 examples/wanvideo/model_training/lora/Wan2.1-VACE-1.3B.sh

accelerate launch examples/wanvideo/model_training/train.py \
  --task "sft" \
  --dataset_base_path data/product_video_dataset \
  --dataset_metadata_path data/product_video_dataset/metadata.csv \
  --data_file_keys "video,vace_reference_image" \
  --extra_inputs "vace_reference_image" \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths \
    "Wan-AI/Wan2.1-VACE-1.3B:diffusion_pytorch_model*.safetensors,\
     Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,\
     Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 10 \
  --batch_size 1 \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --output_path "./models/product_video_lora" \
  --use_gradient_checkpointing
```

### 4.2 关键参数

| 参数 | 值 | 说明 |
|------|---|------|
| `--lora_rank` | 32 | LoRA 秩，可调 8/16/32/64 |
| `--learning_rate` | 1e-4 | 学习率 |
| `--num_epochs` | 10 | 训练轮数 |
| `--batch_size` | 1 | 受显存限制 |
| `--extra_inputs` | "vace_reference_image" | 启用参考图像输入 |

### 4.3 训练资源估算

| 资源 | PoC (100商品) | 小规模 (1K商品) |
|------|--------------|----------------|
| GPU | 1× RTX 4090 (24GB) | 1× A100 (40GB) |
| 训练时间 | ~2-4 小时 | ~1-2 天 |
| 存储 | ~50GB | ~500GB |

---

## 五、推理流程

### 5.1 推理代码

```python
from diffsynth.pipelines.wan_video import WanVideoPipeline
from PIL import Image
import torch

# 加载模型
pipe = WanVideoPipeline.from_pretrained(
    model_id="Wan-AI/Wan2.1-VACE-1.3B",
    torch_dtype=torch.bfloat16,
    device="cuda"
)

# 加载 LoRA
pipe.load_lora("./models/product_video_lora/lora.safetensors")

# 准备输入
product_image = Image.open("product_images/JEWE001.png").resize((832, 480))

# 生成视频
video = pipe(
    prompt="silver bracelet jewelry display on velvet background",
    vace_reference_image=product_image,
    height=480,
    width=832,
    num_frames=49,
    seed=42
)

video.save_video("output.mp4", fps=15)
```

### 5.2 配对视频生成

```python
# 同一个空镜 + 不同商品 = 配对视频

# 商品 A
video_a = pipe(
    prompt="jewelry display",
    vace_reference_image=product_a_image,
    seed=42  # 固定 seed
)

# 商品 B
video_b = pipe(
    prompt="jewelry display",
    vace_reference_image=product_b_image,
    seed=42  # 相同 seed
)

# 配对: (video_a, video_b)
```

**注意**：需要验证相同 seed 是否能产生相似的运动。

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
| 时序一致性 | 无明显闪烁 | 视觉检查 |

### 6.4 泛化能力

| 测试 | 说明 | 目标 |
|------|------|------|
| 测试集商品 | 训练时未见的商品 | 人工评分 ≥ 3/5 |
| 跨类别 | 不同商品类别 | 基本可用 |

---

## 七、PoC 实验计划

### Week 1: 环境搭建 & 数据准备

```
Day 1: 环境搭建
├── 克隆 DiffSynth-Studio
├── 安装依赖
├── 下载 Wan2.1-VACE-1.3B
└── 验证推理脚本

Day 2-3: 数据收集
├── 收集 100 个珠宝商品
├── 每个商品：1-3 个视频 + 3-5 张图片
└── 来源：淘宝/京东

Day 4-5: 数据处理
├── SAM2 分割
├── 商品图片拼接
├── 构建 metadata.csv
└── 数据质量检查
```

### Week 2: 训练 & 评估

```
Day 1-2: LoRA 训练
├── 运行训练脚本
├── 监控 loss
└── 保存 checkpoints

Day 3-4: 评估
├── 重建质量测试
├── 泛化测试
├── 生成样例检查
└── 人工评估

Day 5: 总结
├── 撰写 PoC 报告
├── 分析问题
└── 决定下一步
```

---

## 八、待验证问题

### 8.1 技术问题

| 问题 | 验证方法 |
|------|---------|
| VACE 是否支持 masked_video 输入？ | 检查源码 / 实验 |
| 多图拼接效果如何？ | 对比单图 vs 多图拼接 |
| 相同 seed 是否能产生相似运动？ | 实验验证 |
| LoRA rank 多少合适？ | 对比 8/16/32/64 |

### 8.2 数据问题

| 问题 | 验证方法 |
|------|---------|
| 商品图片与视频不匹配怎么办？ | 数据增强 / 筛选 |
| 100 个商品是否足够 PoC？ | 观察过拟合情况 |
| 分辨率 480p 是否足够？ | 视觉质量评估 |

---

## 九、风险与备选

### 9.1 风险评估

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| VACE 不支持我们的需求 | 中 | 高 | 回退到自定义实现 |
| 泛化能力差 | 中 | 高 | 增加数据 / 调参 |
| 训练不收敛 | 低 | 高 | 检查数据 / 降低学习率 |

### 9.2 备选方案

1. **如果 VACE 不够**：自己实现 ProductEncoder + Cross Attention
2. **如果 1.3B 不够**：迁移到 14B 版本
3. **如果 DiffSynth 有问题**：直接基于官方 Wan2.1 代码修改

---

## 十、参考资料

### 模型

- HuggingFace: https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B
- ModelScope: https://www.modelscope.cn/models/Wan-AI/Wan2.1-VACE-1.3B

### 代码

- DiffSynth-Studio: https://github.com/modelscope/DiffSynth-Studio
- Wan2.1 官方: https://github.com/Wan-Video/Wan2.1
- VACE: https://github.com/ali-vilab/VACE

### 论文

- [VideoPainter](https://arxiv.org/abs/2503.05639) - 数据构造参考
- [VACE](https://arxiv.org/abs/2503.07598) - 视觉条件编码

---

**记录人**: Claude
**最后更新**: 2026-01-23
