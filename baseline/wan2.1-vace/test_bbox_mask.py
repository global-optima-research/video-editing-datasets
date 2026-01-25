"""
VACE Bbox Mask 实验

测试用矩形 bbox mask 替代精确分割 mask，
验证是否能让参考图像更好地生效。

假设：精确 mask 形状成为强先验 → bbox 给模型更多自由度
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


def get_bbox_from_mask(mask_image):
    """从精确 mask 提取 bounding box

    Args:
        mask_image: PIL Image, 精确的分割 mask

    Returns:
        (x_min, y_min, x_max, y_max)
    """
    mask_array = np.array(mask_image.convert('L'))

    # 找到非零像素的坐标
    coords = np.argwhere(mask_array > 127)

    if len(coords) == 0:
        return None

    # 计算 bounding box
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    return (x_min, y_min, x_max, y_max)


def create_bbox_mask(width, height, bbox, margin=0):
    """创建矩形 bbox mask

    Args:
        width: 宽度
        height: 高度
        bbox: (x_min, y_min, x_max, y_max)
        margin: 扩展边距（像素）

    Returns:
        PIL Image (RGB), bbox 内白色，外黑色
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    x_min, y_min, x_max, y_max = bbox

    # 扩展 bbox
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(width, x_max + margin)
    y_max = min(height, y_max + margin)

    # 填充矩形
    mask[y_min:y_max, x_min:x_max] = 255

    # 转换为 RGB
    return Image.fromarray(mask).convert('RGB')


def load_frames_and_masks(video_dir, mask_dir, width, height, num_frames=49):
    """加载视频帧和对应的精确 mask"""
    frame_files = sorted(Path(video_dir).glob('*.jpg'))[:num_frames]
    mask_files = sorted(Path(mask_dir).glob('*.png'))[:num_frames]

    frames = []
    precise_masks = []

    for frame_file, mask_file in zip(frame_files, mask_files):
        # 加载帧
        frame = Image.open(frame_file).convert('RGB').resize((width, height))
        frames.append(frame)

        # 加载精确 mask
        mask = Image.open(mask_file).convert('L').resize(
            (width, height), Image.NEAREST
        ).convert('RGB')
        precise_masks.append(mask)

    return frames, precise_masks


def create_bbox_masks_from_precise(precise_masks, margin=10):
    """从精确 mask 序列生成 bbox mask 序列

    Args:
        precise_masks: List of PIL Images (RGB)
        margin: bbox 扩展边距

    Returns:
        List of PIL Images (RGB), bbox masks
    """
    bbox_masks = []
    width, height = precise_masks[0].size

    for precise_mask in precise_masks:
        # 提取 bbox
        bbox = get_bbox_from_mask(precise_mask)

        if bbox is None:
            # 如果没有检测到物体，使用空 mask
            bbox_mask = Image.new('RGB', (width, height), (0, 0, 0))
        else:
            # 创建 bbox mask
            bbox_mask = create_bbox_mask(width, height, bbox, margin)

        bbox_masks.append(bbox_mask)

    return bbox_masks


def visualize_comparison(frame, precise_mask, bbox_mask, output_path):
    """可视化对比：原帧、精确 mask、bbox mask"""
    from PIL import ImageDraw, ImageFont

    # 创建横向拼接图
    width = frame.width * 3
    height = frame.height
    combined = Image.new('RGB', (width, height))

    # 拼接三张图
    combined.paste(frame, (0, 0))
    combined.paste(precise_mask, (frame.width, 0))
    combined.paste(bbox_mask, (frame.width * 2, 0))

    # 添加标签
    draw = ImageDraw.Draw(combined)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
        except:
            font = ImageFont.load_default()

    labels = ['Original Frame', 'Precise Mask (SAM2)', 'Bbox Mask (Rectangle)']
    for i, label in enumerate(labels):
        x = i * frame.width + 10
        y = 10
        # 白色文字，黑色描边
        draw.text((x, y), label, fill='white', font=font, stroke_width=3, stroke_fill='black')

    combined.save(output_path)
    print(f"Comparison saved to {output_path}")


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

    # Bbox margin（像素）
    BBOX_MARGIN = 20  # 扩展 20 像素，给模型更多空间

    # ============================================
    # Step 1: 加载数据
    # ============================================
    print("Loading frames and masks...")
    frames, precise_masks = load_frames_and_masks(
        SAMPLE_DIR / "video_frames",
        SAMPLE_DIR / "masks",
        WIDTH, HEIGHT, NUM_FRAMES
    )

    # 加载参考图
    reference_image = Image.open(
        SAMPLE_DIR / "reference_images/ref_rubber_duck.png"
    ).convert('RGB')

    print(f"Loaded {len(frames)} frames and {len(precise_masks)} masks")

    # ============================================
    # Step 2: 生成 bbox masks
    # ============================================
    print(f"Creating bbox masks (margin={BBOX_MARGIN}px)...")
    bbox_masks = create_bbox_masks_from_precise(precise_masks, margin=BBOX_MARGIN)

    # 可视化对比（第一帧）
    visualize_comparison(
        frames[0],
        precise_masks[0],
        bbox_masks[0],
        OUTPUT_DIR / "mask_comparison_precise_vs_bbox.jpg"
    )

    # 保存 bbox mask 示例
    bbox_masks[0].save(OUTPUT_DIR / "bbox_mask_frame0.png")

    # ============================================
    # Step 3: 加载 VACE pipeline
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
            ModelConfig(
                model_id="Wan-AI/Wan2.1-VACE-1.3B",
                origin_file_pattern="Wan2.1_VAE.pth"
            ),
        ],
        tokenizer_config=ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="google/umt5-xxl/"
        ),
    )

    prompt = "yellow rubber duck toy, product display, studio lighting"

    # ============================================
    # Step 4: 测试 Bbox Mask
    # ============================================
    print("\n" + "="*50)
    print("Test: Bbox Mask + Reference Image")
    print("="*50)

    video_data = pipe(
        prompt=prompt,
        vace_video=[np.array(f) for f in frames],
        vace_video_mask=[np.array(m) for m in bbox_masks],
        vace_reference_image=np.array(reference_image),
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=50,
        cfg_scale=7.5,
                seed=SEED,
    )

    # 后处理：合成
    print("Compositing with bbox mask...")
    generated_frames = [Image.fromarray(frame) for frame in video_data.to_numpy_images()]

    composited_frames = []
    for orig, gen, mask in zip(frames, generated_frames, bbox_masks):
        # mask 是白色(255)的地方使用生成帧，黑色(0)使用原帧
        composited = Image.composite(gen, orig, mask.convert('L'))
        composited_frames.append(np.array(composited))

    # 保存视频
    output_path = OUTPUT_DIR / "test_bbox_mask.mp4"
    composited_video = VideoData(
        video_frames=composited_frames,
        height=HEIGHT,
        width=WIDTH,
        fps=15
    )
    save_video(composited_video, str(output_path))
    print(f"Saved: {output_path}")

    # 保存第 25 帧对比
    composited_frames[24].save(OUTPUT_DIR / "bbox_mask_result_frame25.jpg")

    # ============================================
    # Step 5: 对比测试（可选：重新测试精确 mask）
    # ============================================
    print("\n" + "="*50)
    print("Test: Precise Mask + Reference Image (for comparison)")
    print("="*50)

    video_data_precise = pipe(
        prompt=prompt,
        vace_video=[np.array(f) for f in frames],
        vace_video_mask=[np.array(m) for m in precise_masks],
        vace_reference_image=np.array(reference_image),
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=50,
        cfg_scale=7.5,
                seed=SEED,
    )

    # 后处理
    print("Compositing with precise mask...")
    generated_frames_precise = [
        Image.fromarray(frame) for frame in video_data_precise.to_numpy_images()
    ]

    composited_frames_precise = []
    for orig, gen, mask in zip(frames, generated_frames_precise, precise_masks):
        composited = Image.composite(gen, orig, mask.convert('L'))
        composited_frames_precise.append(np.array(composited))

    # 保存
    output_path_precise = OUTPUT_DIR / "test_precise_mask_reference.mp4"
    composited_video_precise = VideoData(
        video_frames=composited_frames_precise,
        height=HEIGHT,
        width=WIDTH,
        fps=15
    )
    save_video(composited_video_precise, str(output_path_precise))
    print(f"Saved: {output_path_precise}")

    # ============================================
    # Step 6: 生成并排对比
    # ============================================
    print("\nCreating side-by-side comparison...")

    def create_side_by_side(frame_left, frame_right, label_left, label_right):
        """创建并排对比图"""
        from PIL import ImageDraw, ImageFont

        width = frame_left.width * 2
        height = frame_left.height

        combined = Image.new('RGB', (width, height))
        combined.paste(frame_left, (0, 0))
        combined.paste(frame_right, (frame_left.width, 0))

        # 添加标签
        draw = ImageDraw.Draw(combined)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
        except:
            font = ImageFont.load_default()

        draw.text((10, 10), label_left, fill='white', font=font, stroke_width=2, stroke_fill='black')
        draw.text((frame_left.width + 10, 10), label_right, fill='white', font=font, stroke_width=2, stroke_fill='black')

        return combined

    # Frame 25 对比
    comparison_frame25 = create_side_by_side(
        Image.fromarray(composited_frames_precise[24]),
        Image.fromarray(composited_frames[24]),
        "Precise Mask",
        "Bbox Mask"
    )
    comparison_frame25.save(OUTPUT_DIR / "comparison_precise_vs_bbox_frame25.jpg")
    print(f"Saved comparison: comparison_precise_vs_bbox_frame25.jpg")

    print("\n" + "="*50)
    print("Experiment Complete!")
    print("="*50)
    print(f"\nOutputs:")
    print(f"  - Mask comparison: mask_comparison_precise_vs_bbox.jpg")
    print(f"  - Bbox mask video: test_bbox_mask.mp4")
    print(f"  - Precise mask video: test_precise_mask_reference.mp4")
    print(f"  - Frame 25 comparison: comparison_precise_vs_bbox_frame25.jpg")
    print(f"\nCheck if bbox mask allows reference image to work better!")


if __name__ == "__main__":
    main()
