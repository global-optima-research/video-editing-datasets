"""
VACE Reactive 流权重调整实验

如果 bbox mask 无效，测试降低 reactive 流权重是否能让参考图生效。

策略：不完全清零 reactive 流，而是降低其权重，寻找平衡点：
- 权重 = 1.0: 原始行为（参考图被忽略）
- 权重 = 0.0: 完全清零（运动丢失）
- 权重 = 0.3-0.7: 寻找甜点（参考图生效 + 保留部分运动）
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch

# Add DiffSynth-Studio to path
DIFFSYNTH_PATH = Path(__file__).parent.parent.parent / "DiffSynth-Studio"
sys.path.insert(0, str(DIFFSYNTH_PATH))

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import save_video, VideoData


def apply_reactive_weight(frames, masks, weight=0.3):
    """降低 reactive 流权重

    Args:
        frames: List of PIL Images
        masks: List of PIL Images (mask)
        weight: Reactive 流权重 (0.0 = 完全清零, 1.0 = 原始)

    Returns:
        List of numpy arrays (weighted frames)
    """
    weighted_frames = []

    for frame, mask in zip(frames, masks):
        frame_array = np.array(frame).astype(np.float32)
        mask_array = np.array(mask.convert('L')).astype(np.float32) / 255.0

        # Reactive 区域 = mask 区域
        # Inactive 区域 = 非 mask 区域

        # 降低 reactive 区域的权重
        # reactive_region = frame * mask * weight
        # inactive_region = frame * (1 - mask)
        # weighted_frame = reactive_region + inactive_region

        weighted_array = frame_array * (mask_array * weight + (1 - mask_array))
        weighted_frame = weighted_array.astype(np.uint8)

        weighted_frames.append(weighted_frame)

    return weighted_frames


def main():
    # ============================================
    # 配置
    # ============================================
    PROJECT_DIR = Path(__file__).parent.parent.parent
    SAMPLE_DIR = PROJECT_DIR / "samples/teapot"
    OUTPUT_DIR = PROJECT_DIR / "experiments/results/wan2.1-vace"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 视频参数
    WIDTH = 480
    HEIGHT = 848
    NUM_FRAMES = 49
    SEED = 42

    # 测试不同的权重
    WEIGHTS = [0.0, 0.3, 0.5, 0.7, 1.0]

    # ============================================
    # 加载数据
    # ============================================
    print("Loading data...")

    # 加载帧
    frame_files = sorted((SAMPLE_DIR / "video_frames").glob('*.jpg'))[:NUM_FRAMES]
    frames = [
        Image.open(f).convert('RGB').resize((WIDTH, HEIGHT))
        for f in frame_files
    ]

    # 加载 masks
    mask_files = sorted((SAMPLE_DIR / "masks").glob('*.png'))[:NUM_FRAMES]
    masks = [
        Image.open(f).convert('L').resize((WIDTH, HEIGHT), Image.NEAREST).convert('RGB')
        for f in mask_files
    ]

    # 加载参考图
    reference_image = Image.open(
        SAMPLE_DIR / "reference_images/ref_rubber_duck.png"
    ).convert('RGB')

    print(f"Loaded {len(frames)} frames")

    # ============================================
    # 加载 VACE pipeline
    # ============================================
    print("Loading VACE pipeline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(
                model_id="Wan-AI/Wan2.1-VACE-1.3B",
                origin_file_pattern="diffusion_pytorch_model*.safetensors"
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.1-VACE-1.3B",
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"
            ),
        ]
    )

    prompt = "yellow rubber duck toy, product display, studio lighting"

    # ============================================
    # 测试不同权重
    # ============================================
    results = {}

    for weight in WEIGHTS:
        print("\n" + "="*60)
        print(f"Testing Reactive Weight = {weight}")
        print("="*60)

        # 应用权重
        weighted_frames = apply_reactive_weight(frames, masks, weight)

        # 转换为 PIL 用于可视化
        weighted_frames_pil = [Image.fromarray(f) for f in weighted_frames]

        # 运行 VACE
        video_data = pipe(
            prompt=prompt,
            vace_video=weighted_frames,
            vace_video_mask=[np.array(m) for m in masks],
            vace_reference_image=np.array(reference_image),
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            num_inference_steps=50,
            cfg_scale=7.5,
            embedded_cfg_scale=6.0,
            seed=SEED,
        )

        # 后处理
        generated_frames = [
            Image.fromarray(frame) for frame in video_data.to_numpy_images()
        ]

        composited_frames = []
        for orig, gen, mask in zip(frames, generated_frames, masks):
            composited = Image.composite(gen, orig, mask.convert('L'))
            composited_frames.append(np.array(composited))

        # 保存视频
        output_path = OUTPUT_DIR / f"test_reactive_weight_{weight:.1f}.mp4"
        composited_video = VideoData(
            video_frames=composited_frames,
            height=HEIGHT,
            width=WIDTH,
            fps=15
        )
        save_video(composited_video, str(output_path))
        print(f"Saved: {output_path}")

        # 保存 Frame 25
        frame25_path = OUTPUT_DIR / f"reactive_weight_{weight:.1f}_frame25.jpg"
        Image.fromarray(composited_frames[24]).save(frame25_path)

        results[weight] = {
            'video_path': output_path,
            'frame25': composited_frames[24]
        }

    # ============================================
    # 生成对比图（所有权重的 Frame 25）
    # ============================================
    print("\nCreating comparison grid...")

    from PIL import ImageDraw, ImageFont

    # 创建 1x5 网格
    grid_width = WIDTH * len(WEIGHTS)
    grid_height = HEIGHT
    grid = Image.new('RGB', (grid_width, grid_height))

    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
    except:
        font = ImageFont.load_default()

    for i, weight in enumerate(WEIGHTS):
        frame = Image.fromarray(results[weight]['frame25'])
        grid.paste(frame, (i * WIDTH, 0))

        # 添加标签
        label = f"w={weight:.1f}"
        draw.text((i * WIDTH + 10, 10), label, fill='white', font=font,
                 stroke_width=3, stroke_fill='black')

    grid_path = OUTPUT_DIR / "comparison_reactive_weights_frame25.jpg"
    grid.save(grid_path)
    print(f"Saved comparison: {grid_path}")

    print("\n" + "="*60)
    print("Experiment Complete!")
    print("="*60)
    print(f"\nTested weights: {WEIGHTS}")
    print(f"Compare results to find optimal weight balance:")
    print(f"  - Weight 0.0: Full zero (reference effective, no motion)")
    print(f"  - Weight 1.0: Original (motion preserved, reference ignored)")
    print(f"  - Weight 0.3-0.7: Potential sweet spot?")
    print(f"\nCheck: {grid_path}")


if __name__ == "__main__":
    main()
