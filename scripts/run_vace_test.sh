#!/bin/bash
# Wan2.1-VACE Zero-Shot 测试脚本
# 在 GPU 服务器上运行

set -e

# ============================================
# 配置
# ============================================
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SAMPLE_DIR="$PROJECT_DIR/samples/teapot"
BASELINE_DIR="$PROJECT_DIR/baseline/wan2.1-vace"
OUTPUT_DIR="$PROJECT_DIR/experiments/results/wan2.1-vace"

# ============================================
# Step 0: 检查环境
# ============================================
echo "============================================"
echo "Step 0: 检查环境"
echo "============================================"

# 检查 CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found, GPU may not be available"
else
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
fi

# 检查 Python
python3 --version

# ============================================
# Step 1: 安装依赖 (如果需要)
# ============================================
echo ""
echo "============================================"
echo "Step 1: 检查/安装依赖"
echo "============================================"

# 检查关键依赖
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>/dev/null || {
    echo "Installing transformers..."
    pip install transformers
}

# 检查 SAM2
python3 -c "from sam2.build_sam import build_sam2" 2>/dev/null || {
    echo "SAM2 not found. Please install:"
    echo "  pip install segment-anything-2"
    echo "  # 或者从源码安装: https://github.com/facebookresearch/sam2"
}

# ============================================
# Step 2: 生成 Mask (Grounded SAM 2)
# ============================================
echo ""
echo "============================================"
echo "Step 2: 生成 Mask (Grounded SAM 2)"
echo "============================================"

if [ -d "$SAMPLE_DIR/masks" ] && [ "$(ls -A $SAMPLE_DIR/masks 2>/dev/null)" ]; then
    echo "Masks already exist, skipping..."
    echo "  $(ls $SAMPLE_DIR/masks | wc -l) masks found"
else
    echo "Running Grounded SAM 2 segmentation..."
    cd "$SAMPLE_DIR"
    python3 segment_teapot.py --text_prompt "teapot."
    echo "Masks generated: $(ls masks | wc -l) files"
fi

# ============================================
# Step 3: 运行 Zero-Shot 测试
# ============================================
echo ""
echo "============================================"
echo "Step 3: 运行 Zero-Shot 测试"
echo "============================================"

mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_DIR"

# Test 1: Inpainting only
echo ""
echo "--- Test 1: Inpainting (vace_video + vace_video_mask) ---"
python3 "$BASELINE_DIR/test_zero_shot.py" \
    --video_path "$SAMPLE_DIR/video_frames/" \
    --mask_path "$SAMPLE_DIR/masks/" \
    --prompt "handmade yixing zisha teapot, red clay, product display, studio lighting" \
    --output_dir "$OUTPUT_DIR" \
    --test_case inpainting \
    --num_frames 49 \
    --seed 42

# Test 2: Reference only
echo ""
echo "--- Test 2: Reference (vace_reference_image) ---"
python3 "$BASELINE_DIR/test_zero_shot.py" \
    --reference_path "$SAMPLE_DIR/reference_images/ref_side.jpg" \
    --prompt "handmade yixing zisha teapot, red clay, product display, studio lighting" \
    --output_dir "$OUTPUT_DIR" \
    --test_case reference \
    --num_frames 49 \
    --seed 42

# Test 3: Combined (关键测试)
echo ""
echo "--- Test 3: Combined (vace_video + vace_video_mask + vace_reference_image) ---"
echo "This is the KEY test for ProductVideoGenerator!"
python3 "$BASELINE_DIR/test_zero_shot.py" \
    --video_path "$SAMPLE_DIR/video_frames/" \
    --mask_path "$SAMPLE_DIR/masks/" \
    --reference_path "$SAMPLE_DIR/reference_images/ref_side.jpg" \
    --prompt "handmade yixing zisha teapot, red clay, product display, studio lighting" \
    --output_dir "$OUTPUT_DIR" \
    --test_case combined \
    --num_frames 49 \
    --seed 42

# ============================================
# Step 4: 汇总结果
# ============================================
echo ""
echo "============================================"
echo "Step 4: 测试完成"
echo "============================================"
echo ""
echo "输出文件:"
ls -lh "$OUTPUT_DIR"/*.mp4 2>/dev/null || echo "  (no videos generated)"
echo ""
echo "下一步:"
echo "1. 查看 test1_inpainting.mp4 - 纯 inpainting 效果"
echo "2. 查看 test2_reference.mp4 - 参考图像引导生成"
echo "3. 查看 test3_combined.mp4 - 关键测试：参考图像填入 mask 区域"
echo ""
echo "如果 test3_combined 效果好 → zero-shot 可用"
echo "如果效果不好 → 需要 LoRA 训练"
