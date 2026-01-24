# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **training-based** research project for Product Video Template Transfer (PVTT). It complements [pvtt-training-free](https://github.com/global-optima-research/pvtt-training-free) by exploring LoRA fine-tuning and dataset construction methods.

**Research Goal**: Given a template product video and new product images, generate a new promotional video maintaining the template's visual style, camera motion, and pacing.

**Target Conference**: CVPR 2027

## Development Workflow

**Local → Remote**: Develop/analyze locally, run GPU experiments on remote server
1. Local: Code development, analysis, documentation
2. Sync: `./scripts/sync_to_server.sh` → pushes to `5090:/data/xuhao/pvtt-training-pipeline/`
3. Remote: `ssh 5090` → run experiments with GPU

All baseline tests and model inference must run on the `5090` server (GPU required).

## Core Architecture

### DiffSynth-Studio Integration

The project uses [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) as the inference/training framework, included as a submodule at `DiffSynth-Studio/`. This is the ModelScope community's open-source diffusion model engine supporting:
- Wan2.1-VACE-1.3B (Video Aware Conditional Editing)
- FLUX.2, Qwen-Image, Z-Image models
- LoRA training infrastructure

**Important**: When importing DiffSynth modules, add it to Python path first:
```python
import sys
from pathlib import Path
DIFFSYNTH_PATH = Path(__file__).parent / "DiffSynth-Studio"
sys.path.insert(0, str(DIFFSYNTH_PATH))

from diffsynth.pipelines.wan_video import WanVideoPipeline
```

### VACE Architecture Understanding

Wan2.1-VACE uses **Concept Decoupling** with two streams:
- **Inactive Stream**: `video × (1 - mask)` - preserved regions, masked area zeroed
- **Reactive Stream**: `video × mask` - masked region only, rest zeroed

When `vace_reference_image` is provided, it's prepended to frame sequence with zero mask.

**Critical Finding**: In Combined mode (video + mask + reference), the reactive stream (original pixels) dominates generation. The model prioritizes reconstructing reactive content over reference-based replacement. See `experiments/logs/wan2.1-vace-zero-shot_2026-01-23.md` for detailed experiments.

### Mask Processing Pipeline

VACE requires specific mask handling:
```python
# CRITICAL: Use NEAREST interpolation to maintain binary values
masks = [
    Image.open(f).convert("L")
    .resize((width, height), Image.NEAREST)
    .convert("RGB")
    for f in mask_files
]

# DiffSynth doesn't auto-composite, must do manually:
def composite_with_mask(original_frames, generated_frames, masks):
    """Final output = original × (1 - mask) + generated × mask"""
    return [
        Image.composite(gen, orig, mask)
        for orig, gen, mask in zip(original_frames, generated_frames, masks)
    ]
```

## Common Commands

### Running Baseline Tests

Test Wan2.1-VACE zero-shot capabilities:
```bash
# Full automated test (segmentation + inference)
./scripts/run_vace_test.sh

# Or manual testing:
cd baseline/wan2.1-vace
python test_zero_shot.py \
    --video_path ../../samples/teapot/video_frames/ \
    --mask_path ../../samples/teapot/masks/ \
    --reference_path ../../samples/teapot/reference_images/ref_side.jpg \
    --prompt "handmade yixing zisha teapot, red clay, product display" \
    --test_case combined \
    --height 848 --width 480 --num_frames 49 --seed 42
```

Test cases:
- `inpainting`: video + mask only (no reference image)
- `reference`: reference image only (no video input)
- `combined`: video + mask + reference (key test for object replacement)
- `all`: run all three tests

### Mask Generation

Generate segmentation masks using Grounded SAM 2:
```bash
cd samples/teapot
python segment_teapot.py --text_prompt "teapot."
```

Outputs binary masks to `masks/` directory (one per video frame).

### Environment Setup

**HuggingFace Mirror** (for Chinese servers):
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**Dependencies**:
```bash
pip install torch torchvision transformers accelerate
pip install modelscope  # For Wan model downloads
pip install segment-anything-2  # For mask generation
```

**Hardware Requirements**:
- All experiments run on remote GPU server: `ssh 5090`
- Wan2.1-VACE-1.3B: ~8GB VRAM
- CUDA >= 11.8

### Syncing to Remote Server

Sync local changes to GPU server for running experiments:
```bash
./scripts/sync_to_server.sh
```

This syncs to `5090:/data/xuhao/pvtt-training-pipeline/` (excludes `.git`, `__pycache__`, result videos).

Then run experiments on server:
```bash
ssh 5090
cd /data/xuhao/pvtt-training-pipeline
bash scripts/run_vace_test.sh
```

## Directory Structure

```
pvtt-training-pipeline/
├── DiffSynth-Studio/          # Git submodule - inference/training framework
├── baseline/
│   └── wan2.1-vace/           # Wan2.1-VACE zero-shot baseline
│       ├── test_zero_shot.py  # Main test script
│       └── README.md
├── samples/
│   └── teapot/                # Test sample (purple clay teapot)
│       ├── video_frames/      # Original video frames (extracted)
│       ├── masks/             # SAM2 segmentation masks
│       ├── reference_images/  # Reference product images
│       └── segment_teapot.py  # Grounded SAM 2 script
├── experiments/
│   ├── logs/                  # Markdown experiment logs
│   └── results/               # Generated videos/images
│       └── wan2.1-vace/
├── scripts/
│   ├── run_vace_test.sh       # Automated test pipeline
│   └── sync_to_server.sh      # Remote server sync
└── papers/                    # Paper analysis (PDF analysis excluded from git)
```

## Key Findings & Design Decisions

### Zero-Shot VACE Limitations

**Conclusion**: Wan2.1-VACE cannot perform zero-shot object replacement when video + mask + reference are combined.

| Test Scenario | Reference Image Effect | Result |
|--------------|----------------------|--------|
| Reference-only | ✅ Works | Generates video guided by reference |
| Video Inpainting | ✅ Works | Reconstructs masked regions |
| **Combined (ref + video + mask)** | ❌ Ignored | Reconstructs original video content |

**Why**: The reactive stream contains original object pixels that dominate generation. Reference image acts as style guidance rather than content replacement.

**Validation**: Tested with completely different object (rubber duck) as reference - model still generated original object (teapot). See experiment log for details.

### Zeroed Reactive Stream Experiment

Attempted workaround: zero out masked regions in input frames to remove original content from reactive stream.

**Result**: Partial success
- ✅ Reference image colors/textures applied
- ❌ Shape constrained by mask (generates "teapot-shaped duck")
- ❌ Motion information lost (no rotation)

**Implication**: Need alternative approaches for proper object replacement with motion preservation.

## Next Steps & Research Directions

1. **LoRA Fine-tuning**: Train adapter to strengthen reference image conditioning while preserving motion
2. **Two-stage Pipeline**: Reference-to-video generation → ControlNet composition with depth/pose guidance
3. **Motion Transfer**: Extract motion trajectory from original video, apply to reference-guided generation
4. **Alternative Models**: Explore video versions of AnyDoor, Paint-by-Example, or VideoAnyDoor

## Experiment Logging Requirements

**CRITICAL**: All experiments MUST be documented in `experiments/logs/` with detailed Markdown logs.

### Required Log Structure

Every experiment log should follow this template (see `experiments/logs/sam2-segmentation_2026-01-20.md` as reference):

```markdown
# {Experiment Title}

**日期**: {YYYY-MM-DD}
**主题**: {Brief description}
**测试用例**: {Test case details}

---

## 实验背景
Why this experiment? What problem does it address?

## 实验配置
| 参数 | 值 |
|------|-----|
| 模型 | ... |
| 设备 | 5090 GPU |
| 环境 | ... |

## 输入
Document all input data with paths and visualizations

## 实验结果
### 处理统计
Timing, throughput, resource usage

### 效果评估
Qualitative and quantitative results with comparison tables

### 输出文件
List all generated artifacts with paths

## 技术细节
Code snippets, architectural insights, implementation notes

## 关键发现
Key insights, unexpected behaviors, limitations discovered

## 结论
Summarize findings and implications for next steps

## 相关文件
- Scripts used
- Input/output paths on 5090 server
- Result visualizations
```

### What to Include

1. **Visuals**: Embed result images/frames in the log (store in `experiments/results/{model}/images/`)
2. **Comparisons**: Use tables to compare different approaches/parameters
3. **Iterations**: Document failed attempts and why they failed (e.g., Point prompt v1-v5 iterations)
4. **Server Paths**: Include actual paths on 5090 server for reproducibility
5. **Quantitative Metrics**: Speed (fps), memory usage, quality scores
6. **Code Snippets**: Include key code that illustrates the method

### Examples of Good Logs

- `experiments/logs/sam2-segmentation_2026-01-20.md` - Comprehensive iteration documentation
- `experiments/logs/wan2.1-vace-zero-shot_2026-01-23.md` - Detailed failure analysis with validation experiments

### When to Create a Log

- After running any baseline test
- After completing a segmentation task
- After training/fine-tuning experiments
- After discovering important findings (even failed experiments)
- After pipeline component testing (SAM2, ProPainter, etc.)

**Remember**: Failed experiments are valuable! Document what didn't work and why.

## File Naming Conventions

- Experiment logs: `experiments/logs/{model}-{task}_{date}.md`
- Result videos: `experiments/results/{model}/{test_case}.mp4`
- Result images: `experiments/results/{model}/images/{description}.jpg`
- Comparison videos: `comparison_{variant}.mp4`
- Analysis docs: `{topic}-analysis.md` (root level)
- Proposals: `{topic}-proposal.md` (root level)

## Important Notes

- **Large videos excluded from git**: Files >100MB added to .gitignore (comparison_zeroed_reactive.mp4, comparison_zeroed_reactive_qt.mp4)
- **Resolution constraints**: Wan2.1 requires width/height divisible by 16. For 9:16 vertical video, use 480×848
- **Frame format**: VACE accepts directory of frame images or PIL Image list
- **Mask format**: Binary masks must be converted to RGB format for DiffSynth
- **Prompt engineering**: Use product-specific descriptive prompts, e.g., "handmade yixing zisha teapot, red clay, product display, studio lighting"
