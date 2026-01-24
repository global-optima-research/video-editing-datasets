"""
Wan2.1-VACE-1.3B Zero-Shot 测试脚本

测试目标：验证 VACE 能否 zero-shot 完成以下任务
- 输入：masked_video (商品区域置零) + product_reference_image (商品图片)
- 输出：生成视频，商品被填入 masked 区域

测试场景：
1. Inpainting: vace_video + vace_video_mask → 填充 masked 区域
2. Reference-guided: vace_reference_image → 参考图像引导生成
3. Combined: vace_video + vace_video_mask + vace_reference_image → 参考图像填入 masked 区域

运行方式：
    python test_zero_shot.py --test_case inpainting
    python test_zero_shot.py --test_case reference
    python test_zero_shot.py --test_case combined
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from PIL import Image
import numpy as np

# Add DiffSynth-Studio to path
DIFFSYNTH_PATH = Path(__file__).parent.parent.parent / "DiffSynth-Studio"
sys.path.insert(0, str(DIFFSYNTH_PATH))

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import save_video, VideoData


def load_pipeline(device="cuda", torch_dtype=torch.bfloat16):
    """加载 Wan2.1-VACE-1.3B pipeline"""
    print("Loading Wan2.1-VACE-1.3B pipeline...")

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype,
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

    print("Pipeline loaded successfully!")
    return pipe


def create_test_data(output_dir: Path, height=480, width=832, num_frames=49):
    """
    创建测试数据（如果没有真实数据）

    生成：
    - 纯色背景视频
    - 中心矩形 mask
    - 简单的参考图像
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建简单的背景视频帧（紫色丝绸背景）
    frames = []
    for i in range(num_frames):
        # 创建渐变紫色背景
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = 80   # R
        frame[:, :, 1] = 40   # G
        frame[:, :, 2] = 120  # B
        # 添加一些变化
        noise = np.random.randint(-10, 10, (height, width, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(Image.fromarray(frame))

    # 保存视频帧
    frames_dir = output_dir / "test_video_frames"
    frames_dir.mkdir(exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(frames_dir / f"{i:04d}.png")

    # 创建 mask（中心区域）
    mask = np.zeros((height, width), dtype=np.uint8)
    cy, cx = height // 2, width // 2
    h_box, w_box = height // 3, width // 3
    mask[cy - h_box//2:cy + h_box//2, cx - w_box//2:cx + w_box//2] = 255
    mask_img = Image.fromarray(mask)
    mask_img.save(output_dir / "test_mask.png")

    # 创建参考图像（简单的金色矩形代表商品）
    ref_img = np.zeros((height, width, 3), dtype=np.uint8)
    ref_img[cy - h_box//2:cy + h_box//2, cx - w_box//2:cx + w_box//2] = [218, 165, 32]  # 金色
    ref_img = Image.fromarray(ref_img)
    ref_img.save(output_dir / "test_reference.png")

    print(f"Test data created in {output_dir}")
    return frames, mask_img, ref_img


def load_video_frames(video_path: Path, height=480, width=832, limit=None) -> list[Image.Image]:
    """从目录或视频文件加载帧"""
    if video_path.is_dir():
        # 从目录加载帧
        frame_files = sorted(video_path.glob("*.png")) + sorted(video_path.glob("*.jpg"))
        if limit:
            frame_files = frame_files[:limit]
        frames = [Image.open(f).resize((width, height)) for f in frame_files]
    else:
        # 从视频文件加载
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if limit and len(frames) >= limit:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame).resize((width, height))
            frames.append(frame)
        cap.release()
    return frames


def load_masks(mask_path: Path, height=480, width=832, num_frames=None) -> list[Image.Image]:
    """
    加载 mask
    - 如果是目录：加载所有 mask 文件
    - 如果是单个文件：复制为 num_frames 个相同的 mask

    Note: DiffSynth expects RGB masks, so we convert L to RGB
    """
    mask_path = Path(mask_path)

    if mask_path.is_dir():
        # 从目录加载 mask 序列
        mask_files = sorted(mask_path.glob("*.png")) + sorted(mask_path.glob("*.jpg"))
        if num_frames:
            mask_files = mask_files[:num_frames]
        # Convert L to RGB for DiffSynth compatibility
        masks = [Image.open(f).convert("L").resize((width, height), Image.NEAREST).convert("RGB") for f in mask_files]
        print(f"Loaded {len(masks)} masks from directory")
    else:
        # 单个 mask 文件，复制为序列
        single_mask = Image.open(mask_path).convert("L").resize((width, height), Image.NEAREST).convert("RGB")
        masks = [single_mask] * (num_frames or 49)
        print(f"Using single mask for all {len(masks)} frames")

    return masks


def composite_with_mask(original_frames: list[Image.Image], generated_frames: list, masks: list[Image.Image]) -> list:
    """
    后处理：用 mask 合成最终视频
    最终输出 = 原视频 × (1 - mask) + 生成视频 × mask

    这样 mask 外的区域完全保持原视频不变
    """
    composited = []
    for i, (orig, gen, mask) in enumerate(zip(original_frames, generated_frames, masks)):
        # 确保尺寸一致
        if isinstance(gen, Image.Image):
            gen_img = gen
        else:
            gen_img = Image.fromarray(gen)

        orig_resized = orig.resize(gen_img.size)
        mask_resized = mask.convert("L").resize(gen_img.size)

        # 合成：mask 白色区域用生成的，黑色区域用原图
        composited_frame = Image.composite(gen_img, orig_resized, mask_resized)
        composited.append(np.array(composited_frame))

    return composited


def test_inpainting(
    pipe: WanVideoPipeline,
    video_frames: list[Image.Image],
    masks: list[Image.Image],
    prompt: str,
    output_path: Path,
    seed: int = 42,
):
    """
    测试 1: 纯 Inpainting

    输入：vace_video + vace_video_mask
    期望：模型填充 masked 区域（无参考图像引导）
    """
    print("\n" + "="*60)
    print("Test 1: Inpainting (vace_video + vace_video_mask)")
    print("="*60)

    # 确保 mask 数量与帧数一致
    mask_frames = masks[:len(video_frames)]
    if len(mask_frames) < len(video_frames):
        mask_frames = mask_frames + [mask_frames[-1]] * (len(video_frames) - len(mask_frames))

    video = pipe(
        prompt=prompt,
        negative_prompt="低质量，模糊，变形",
        vace_video=video_frames,
        vace_video_mask=mask_frames,
        height=video_frames[0].height,
        width=video_frames[0].width,
        num_frames=len(video_frames),
        seed=seed,
        tiled=True,
    )

    # 后处理：用 mask 合成，确保 mask 外区域完全保持原视频
    # 将 RGB mask 转回灰度用于合成
    gray_masks = [m.convert("L") if m.mode == "RGB" else m for m in masks[:len(video_frames)]]
    video = composite_with_mask(video_frames, video, gray_masks)

    save_video(video, str(output_path), fps=15, quality=5)
    print(f"Saved: {output_path}")
    return video


def test_reference_only(
    pipe: WanVideoPipeline,
    reference_image: Image.Image,
    prompt: str,
    output_path: Path,
    height: int = 480,
    width: int = 832,
    num_frames: int = 49,
    seed: int = 42,
):
    """
    测试 2: 纯参考图像生成

    输入：vace_reference_image
    期望：生成与参考图像风格一致的视频
    """
    print("\n" + "="*60)
    print("Test 2: Reference-guided generation (vace_reference_image)")
    print("="*60)

    video = pipe(
        prompt=prompt,
        negative_prompt="低质量，模糊，变形",
        vace_reference_image=reference_image.resize((width, height)),
        height=height,
        width=width,
        num_frames=num_frames,
        seed=seed,
        tiled=True,
    )

    save_video(video, str(output_path), fps=15, quality=5)
    print(f"Saved: {output_path}")
    return video


def test_combined(
    pipe: WanVideoPipeline,
    video_frames: list[Image.Image],
    masks: list[Image.Image],
    reference_image: Image.Image,
    prompt: str,
    output_path: Path,
    seed: int = 42,
):
    """
    测试 3: 组合测试 - 参考图像引导的 Inpainting

    输入：vace_video + vace_video_mask + vace_reference_image
    期望：将参考图像中的物体填入 masked 区域

    这是我们真正需要的功能！
    """
    print("\n" + "="*60)
    print("Test 3: Combined (vace_video + vace_video_mask + vace_reference_image)")
    print("="*60)
    print("This is the key test for ProductVideoGenerator!")

    # 确保 mask 数量与帧数一致
    mask_frames = masks[:len(video_frames)]
    if len(mask_frames) < len(video_frames):
        mask_frames = mask_frames + [mask_frames[-1]] * (len(video_frames) - len(mask_frames))

    video = pipe(
        prompt=prompt,
        negative_prompt="低质量，模糊，变形",
        vace_video=video_frames,
        vace_video_mask=mask_frames,
        vace_reference_image=reference_image.resize((video_frames[0].width, video_frames[0].height)),
        height=video_frames[0].height,
        width=video_frames[0].width,
        num_frames=len(video_frames),
        seed=seed,
        tiled=True,
    )

    # 后处理：用 mask 合成，确保 mask 外区域完全保持原视频
    gray_masks = [m.convert("L") if m.mode == "RGB" else m for m in masks[:len(video_frames)]]
    video = composite_with_mask(video_frames, video, gray_masks)

    save_video(video, str(output_path), fps=15, quality=5)
    print(f"Saved: {output_path}")
    return video


def main():
    parser = argparse.ArgumentParser(description="Wan2.1-VACE-1.3B Zero-Shot Test")
    parser.add_argument(
        "--test_case",
        type=str,
        choices=["inpainting", "reference", "combined", "all"],
        default="all",
        help="Which test case to run"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to video file or frames directory"
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default=None,
        help="Path to mask image"
    )
    parser.add_argument(
        "--reference_path",
        type=str,
        default=None,
        help="Path to reference image"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="elegant jewelry product on purple silk fabric, studio lighting, product photography",
        help="Generation prompt"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Video width"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=49,
        help="Number of frames (default: 49 = ~3.3s @ 15fps)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载或创建测试数据
    if args.video_path or args.reference_path:
        print("Loading provided test data...")

        # 加载视频帧
        if args.video_path:
            video_frames = load_video_frames(
                Path(args.video_path), args.height, args.width, limit=args.num_frames
            )
        else:
            video_frames = None

        # 加载 mask
        if args.mask_path:
            masks = load_masks(
                Path(args.mask_path), args.height, args.width, num_frames=args.num_frames
            )
        else:
            # 创建默认 mask（中心区域）
            mask = np.zeros((args.height, args.width), dtype=np.uint8)
            cy, cx = args.height // 2, args.width // 2
            h_box, w_box = args.height // 3, args.width // 3
            mask[cy - h_box//2:cy + h_box//2, cx - w_box//2:cx + w_box//2] = 255
            masks = [Image.fromarray(mask)] * args.num_frames

        # 加载参考图像
        if args.reference_path:
            reference = Image.open(args.reference_path).convert("RGB")
        else:
            reference = None
    else:
        print("No test data provided, creating synthetic test data...")
        test_data_dir = output_dir / "test_data"
        video_frames, single_mask, reference = create_test_data(
            test_data_dir, args.height, args.width, args.num_frames
        )
        masks = [single_mask] * args.num_frames

    # 限制帧数
    if video_frames:
        video_frames = video_frames[:args.num_frames]
        print(f"Video frames: {len(video_frames)}, Size: {video_frames[0].size}")
    print(f"Masks: {len(masks)}")

    # 加载模型
    pipe = load_pipeline(device=args.device)

    # 运行测试
    results = {}

    if args.test_case in ["inpainting", "all"]:
        if video_frames is None:
            print("Skipping inpainting test: no video frames provided")
        else:
            results["inpainting"] = test_inpainting(
                pipe=pipe,
                video_frames=video_frames,
                masks=masks,
                prompt=args.prompt,
                output_path=output_dir / "test1_inpainting.mp4",
                seed=args.seed,
            )

    if args.test_case in ["reference", "all"]:
        if reference is None:
            print("Skipping reference test: no reference image provided")
        else:
            results["reference"] = test_reference_only(
                pipe=pipe,
                reference_image=reference,
                prompt=args.prompt,
                output_path=output_dir / "test2_reference.mp4",
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                seed=args.seed,
            )

    if args.test_case in ["combined", "all"]:
        if video_frames is None or reference is None:
            print("Skipping combined test: need both video frames and reference image")
        else:
            results["combined"] = test_combined(
                pipe=pipe,
                video_frames=video_frames,
                masks=masks,
                reference_image=reference,
                prompt=args.prompt,
                output_path=output_dir / "test3_combined.mp4",
                seed=args.seed,
            )

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
    print(f"\nOutputs saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Review generated videos")
    print("2. If test3_combined works well → zero-shot may be sufficient")
    print("3. If not → proceed with LoRA training as planned")


if __name__ == "__main__":
    main()
