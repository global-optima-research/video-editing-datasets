"""
Grounded SAM 2 茶壶分割脚本

基于 experiments/logs/sam2-segmentation_2026-01-20.md 验证过的方案：
- Grounding DINO 检测 bbox
- SAM2 ImagePredictor 分割（效果比 VideoPredictor 的 box prompt 更好）

运行方式（需要在 GPU 服务器上运行）：
    python segment_teapot.py

依赖：
    pip install transformers
    pip install sam2  # 或 segment-anything-2
"""

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def load_grounding_dino():
    """加载 Grounding DINO 模型"""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    print("Loading Grounding DINO...")
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    return processor, model


def load_sam2():
    """加载 SAM2 模型"""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    print("Loading SAM2...")
    # 根据实际路径调整
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # 如果使用 HuggingFace 版本
    try:
        sam2_model = build_sam2(model_cfg, sam2_checkpoint)
        predictor = SAM2ImagePredictor(sam2_model)
    except Exception as e:
        print(f"Error loading SAM2 from local checkpoint: {e}")
        print("Trying HuggingFace version...")
        from transformers import Sam2Processor, Sam2Model
        sam2_processor = Sam2Processor.from_pretrained("facebook/sam2-hiera-large")
        sam2_model = Sam2Model.from_pretrained("facebook/sam2-hiera-large")
        return sam2_processor, sam2_model, "hf"

    return predictor, None, "local"


def detect_with_grounding_dino(processor, model, image, text_prompt, device, box_threshold=0.3):
    """使用 Grounding DINO 检测物体"""
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=0.25,
        target_sizes=[image.size[::-1]]  # (height, width)
    )[0]

    return results


def segment_with_sam2(predictor, image, boxes, device):
    """使用 SAM2 ImagePredictor 分割"""
    predictor.set_image(np.array(image))

    # 合并所有检测到的 box
    if len(boxes) == 0:
        return None

    # 选择最大的 box（根据实验记录，最大框效果最好）
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
    largest_idx = np.argmax(areas)
    largest_box = boxes[largest_idx]

    print(f"Using largest box: {largest_box.tolist()}, area: {areas[largest_idx]:.0f}")

    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=largest_box.cpu().numpy(),
        multimask_output=False
    )

    return masks[0]  # [H, W] boolean


def segment_with_sam2_hf(processor, model, image, boxes, device):
    """使用 HuggingFace 版本的 SAM2"""
    if len(boxes) == 0:
        return None

    # 选择最大的 box
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
    largest_idx = np.argmax(areas)
    largest_box = boxes[largest_idx].cpu().numpy().tolist()

    print(f"Using largest box: {largest_box}, area: {areas[largest_idx]:.0f}")

    inputs = processor(
        images=image,
        input_boxes=[[largest_box]],
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"],
        inputs["reshaped_input_sizes"]
    )

    mask = masks[0][0, 0].cpu().numpy()
    return mask > 0.5


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", default="video_frames")
    parser.add_argument("--output_dir", default="masks")
    parser.add_argument("--text_prompt", default="teapot.")
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--sample_interval", type=int, default=1,
                        help="Process every N frames (1=all frames)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    frames_dir = script_dir / args.frames_dir
    output_dir = script_dir / args.output_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 加载模型
    gdino_processor, gdino_model = load_grounding_dino()
    gdino_model = gdino_model.to(device)

    sam2_result = load_sam2()
    if sam2_result[2] == "hf":
        sam2_processor, sam2_model, _ = sam2_result
        sam2_model = sam2_model.to(device)
        use_hf = True
    else:
        sam2_predictor, _, _ = sam2_result
        sam2_predictor.model = sam2_predictor.model.to(device)
        use_hf = False

    # 获取所有帧
    frame_files = sorted(frames_dir.glob("*.jpg"))
    print(f"Found {len(frame_files)} frames")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 处理每一帧
    for i, frame_file in enumerate(frame_files):
        if i % args.sample_interval != 0:
            continue

        image = Image.open(frame_file).convert("RGB")

        # 1. Grounding DINO 检测
        results = detect_with_grounding_dino(
            gdino_processor, gdino_model, image,
            args.text_prompt, device, args.box_threshold
        )

        boxes = results["boxes"]
        scores = results["scores"]

        if len(boxes) == 0:
            print(f"Frame {i}: No detection, using previous mask or empty")
            # 可以选择使用上一帧的 mask 或空 mask
            mask = np.zeros((image.height, image.width), dtype=np.uint8)
        else:
            # 2. SAM2 分割
            if use_hf:
                mask = segment_with_sam2_hf(sam2_processor, sam2_model, image, boxes, device)
            else:
                mask = segment_with_sam2(sam2_predictor, image, boxes, device)

            if mask is None:
                mask = np.zeros((image.height, image.width), dtype=np.uint8)
            else:
                mask = mask.astype(np.uint8) * 255

        # 保存 mask
        mask_path = output_dir / f"{i:04d}.png"
        Image.fromarray(mask).save(mask_path)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"Processed frame {i + 1}/{len(frame_files)}, "
                  f"detections: {len(boxes)}, "
                  f"mask coverage: {mask.sum() / mask.size * 100:.1f}%")

    print(f"\nMasks saved to {output_dir}")
    print(f"Total: {len(list(output_dir.glob('*.png')))} masks")


if __name__ == "__main__":
    main()
