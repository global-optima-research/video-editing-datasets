# Wan2.1-VACE-1.3B Baseline 测试

测试 Wan2.1-VACE-1.3B 的 zero-shot 能力，验证是否需要 LoRA 训练。

## 测试目标

验证 VACE 能否 zero-shot 完成：
```
输入: masked_video + product_reference_image
输出: 商品被填入 masked 区域的视频
```

## 测试用例

| 测试 | 输入 | 目的 |
|------|------|------|
| Test 1: Inpainting | `vace_video` + `vace_video_mask` | 纯 inpainting，无参考图像 |
| Test 2: Reference | `vace_reference_image` | 纯参考图像引导生成 |
| Test 3: Combined | 全部三个输入 | **关键测试** - 参考图像引导的 inpainting |

## 运行方式

### 使用合成测试数据

```bash
cd baseline/wan2.1-vace
python test_zero_shot.py --test_case all --output_dir ./outputs
```

### 使用真实数据

```bash
python test_zero_shot.py \
    --video_path /path/to/video_frames/ \
    --mask_path /path/to/mask.png \
    --reference_path /path/to/product.png \
    --prompt "elegant jewelry product display" \
    --test_case combined
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--test_case` | `all` | 测试用例 (inpainting/reference/combined/all) |
| `--video_path` | None | 视频帧目录或视频文件 |
| `--mask_path` | None | Mask 图像路径 |
| `--reference_path` | None | 参考图像路径 |
| `--prompt` | jewelry prompt | 生成提示词 |
| `--height` | 480 | 视频高度 |
| `--width` | 832 | 视频宽度 |
| `--num_frames` | 49 | 帧数 (~3.3s @ 15fps) |
| `--seed` | 42 | 随机种子 |

## 预期结果

### 如果 Test 3 成功

- Zero-shot 即可使用
- 可跳过 LoRA 训练
- 直接用于数据生成

### 如果 Test 3 失败

- 需要 LoRA 训练
- 参考 `experiments/logs/product_video_generator_2026-01-23.md`

## 硬件需求

- GPU: ~8GB VRAM (1.3B 模型)
- 推荐: RTX 3090/4090 或更高

## 文件结构

```
baseline/wan2.1-vace/
├── README.md               # 本文件
├── test_zero_shot.py       # 测试脚本
└── outputs/                # 输出目录 (运行后生成)
    ├── test_data/          # 合成测试数据
    ├── test1_inpainting.mp4
    ├── test2_reference.mp4
    └── test3_combined.mp4
```

## 依赖

- DiffSynth-Studio (已包含在项目中)
- PyTorch >= 2.0
- CUDA >= 11.8

安装依赖：
```bash
pip install torch torchvision
pip install transformers accelerate
pip install modelscope
```
