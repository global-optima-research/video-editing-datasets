# Awesome Video Editing Datasets

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of public datasets and benchmarks for video editing research.

## Contents

- [Text/Instruction-Guided Video Editing](#textinstruction-guided-video-editing)
- [Video Subject Swapping](#video-subject-swapping)
- [Video Inpainting](#video-inpainting)
- [Video Editing Quality Assessment](#video-editing-quality-assessment)
- [Evaluation Metrics](#evaluation-metrics)

---

## Text/Instruction-Guided Video Editing

Datasets for text-driven and instruction-based video editing. TGVE uses source/target prompts; IVE uses edit instructions — both are included here as they target similar tasks.

### Evaluation Benchmarks

| Dataset | Size | Resolution | Edit Types | Venue |
|---------|------|------------|------------|-------|
| **VEditBench** | 420 videos | Varied | 6 tasks (insert/remove/swap/scene/motion/style) | ICLR 2025 |
| **IVEBench** | 600 videos | ≥2K source | 8 categories, 35 subcategories | arXiv 2025.10 |
| **FiVE-Bench** | 100 videos, 420 prompts | - | 6 fine-grained tasks | ICCV 2025 |
| **RVEBenchmark** | 100 videos, 519 queries | - | Reasoning editing (3 levels) | arXiv 2025.11 |
| **OpenVE-Bench** | - | 720P | 8 categories (SA + NSA edits) | arXiv 2025.12 |
| LOVEU-TGVE-2024 | 200 videos | Varied (2s-48s) | Insert, Remove, Change, Scene, Motion, Style | CVPR 2024 |
| LOVEU-TGVE-2023 | 76 videos | 480×480 | Style, Background, Object, Multiple | CVPR 2023 |
| BalanceCC | 100 videos | - | Creative & Controllable | CVPR 2024 |

### Training Datasets

| Dataset | Size | Resolution | Venue |
|---------|------|------------|-------|
| **OpenVE-3M** | 3M+ triplets | 720P | arXiv 2025.12 |
| **Señorita-2M** | 2M pairs | Varied | NeurIPS D&B 2025 |
| **Ditto-1M** | 1M triplets | 1280×720, 101 frames | arXiv 2025.10 |
| **InsViE-1M** | 1M triplets | High-res | ICCV 2025 |
| VIVID-10M | 9.7M samples | - | arXiv 2024.11 |

---

### VEditBench

- **Paper**: [VEditBench: Holistic Benchmark for Text-Guided Video Editing](https://openreview.net/forum?id=6325Jzc9eR) (ICLR 2025)
- **Size**: 420 real-world videos (300 short 2-4s + 120 long 10-20s)
- **Tasks**: Object Insertion, Object Removal, Object Swap, Scene Replacement, Motion Change, Style Translation
- **Metrics**: 9 evaluation dimensions for semantic fidelity and visual quality

### IVEBench

- **Paper**: [IVEBench: Modern Benchmark Suite for Instruction-Guided Video Editing Assessment](https://arxiv.org/abs/2510.11647) (arXiv Oct 2025)
- **Size**: 600 high-quality source videos, 32-1024 frames
- **Tasks**: 8 categories with 35 subcategories
- **Metrics**: 12 metrics across 3 dimensions (video quality, instruction compliance, video fidelity)
- **Source**: Pexels, Mixkit, UltraVideo (≥2K resolution)

### OpenVE-3M / OpenVE-Bench

- **Paper**: [OpenVE-3M: A Large-Scale High-Quality Dataset for Instruction-Guided Video Editing](https://arxiv.org/abs/2512.07826) (arXiv Dec 2025)
- **Dataset**: [Lewandofski/OpenVE-3M](https://huggingface.co/datasets/Lewandofski/OpenVE-3M) (HuggingFace)
- **Size**: 3,000,000+ video editing triplets, 65-129 frames per video
- **Resolution**: 720P (1280×720 / 720×1280)
- **Categories**: Spatially-aligned (Global Style, Background Change, Local Change/Remove/Add, Subtitles Edit) + Non-spatially-aligned (Camera Multi-Shot Edit, Creative Edit)
- **License**: CC-BY-NC-4.0
- **Affiliations**: ByteDance & Zhejiang University

### InsViE-1M

- **Paper**: [InsViE-1M: Effective Instruction-based Video Editing with Elaborate Dataset Construction](https://arxiv.org/abs/2503.20287) (ICCV 2025)
- **Code**: [langmanbusi/InsViE](https://github.com/langmanbusi/insvie)
- **Size**: 1M high-quality instruction-video editing triplets
- **Pipeline**: Two-stage editing-filtering with GPT-4o quality control

### Señorita-2M

- **Paper**: [Señorita-2M: A High-Quality Instruction-based Dataset for General Video Editing by Video Specialists](https://arxiv.org/abs/2502.06734) (NeurIPS D&B 2025)
- **Dataset**: [SENORITADATASET/Senorita](https://huggingface.co/datasets/SENORITADATASET/Senorita) (HuggingFace)
- **Code**: [zibojia/SENORITA](https://github.com/zibojia/SENORITA)
- **Website**: [senorita-2m-dataset.github.io](https://senorita-2m-dataset.github.io/)
- **Size**: 2M video editing pairs, 18 subcategories in 2 classes
- **Tasks**: Object removal/swap/addition, global/local stylization
- **Pipeline**: 4 trained video editing experts (CogVideoX) + 14 existing task experts + filtering

### Ditto-1M

- **Paper**: [Ditto: Scaling Instruction-Based Video Editing with a High-Quality Synthetic Dataset](https://arxiv.org/abs/2510.15742) (arXiv Oct 2025)
- **Dataset**: [QingyanBai/Ditto-1M](https://huggingface.co/datasets/QingyanBai/Ditto-1M) (HuggingFace)
- **Code**: [EzioBy/Ditto](https://github.com/EzioBy/Ditto)
- **Website**: [editto.net](https://editto.net/)
- **Size**: 1M triplets (700K global + 300K local editing), 1280×720, 101 frames @ 20 FPS
- **Categories**: Scenes (23%), Single-person (34%), Group activities (33%), Objects (10%)
- **Cost**: 12,000 GPU-days to build

### FiVE-Bench

- **Paper**: [FiVE: A Fine-grained Video Editing Benchmark](https://arxiv.org/abs/2503.13684) (ICCV 2025)
- **Dataset**: [LIMinghan/FiVE-Fine-Grained-Video-Editing-Benchmark](https://huggingface.co/datasets/LIMinghan/FiVE-Fine-Grained-Video-Editing-Benchmark) (HuggingFace)
- **Code**: [MinghanLi/FiVE-Bench](https://github.com/MinghanLi/FiVE-Bench)
- **Size**: 100 videos (74 DAVIS + 26 synthetic), 420 prompt pairs
- **Tasks**: Object replacement (rigid/non-rigid), Color alteration, Material modification, Object addition/removal
- **Metrics**: 15 metrics + FiVE-Acc (VLM-based success metric)

### RVEBenchmark

- **Paper**: [Text-Driven Reasoning Video Editing via Reinforcement Learning](https://arxiv.org/abs/2511.14100) (arXiv Nov 2025)
- **Size**: 100 videos, 519 implicit queries
- **Tasks**: 3 reasoning levels × 3 categories (semantic, spatial, temporal reasoning)
- **Details**: First benchmark for reasoning-based video editing with implicit instructions

### LOVEU-TGVE-2023

- **Paper**: [CVPR 2023 Text Guided Video Editing Competition](https://arxiv.org/abs/2310.16003)
- **Website**: [LOVEU@CVPR'23 Track4](https://sites.google.com/view/loveucvpr23/track4)
- **Code**: [showlab/loveu-tgve-2023](https://github.com/showlab/loveu-tgve-2023)
- **Details**: 76 videos (32 or 128 frames), 4 editing prompts per video. The first standardized benchmark for TGVE.

### LOVEU-TGVE-2024

- **Website**: [LOVEU@CVPR'24 Track2A](https://sites.google.com/view/loveucvpr24/track2a)
- **Details**: 200 videos across 5 categories (Animal, Food, Scenery, Sport, Vehicle). 6 editing types including Object Change.

### BalanceCC

- **Paper**: [CCEdit: Creative and Controllable Video Editing via Diffusion Models](https://openaccess.thecvf.com/content/CVPR2024/papers/Feng_CCEdit_Creative_and_Controllable_Video_Editing_via_Diffusion_Models_CVPR_2024_paper.pdf) (CVPR 2024)
- **Details**: 100 real-world videos for evaluating controllability and creativity in video editing.

### TGVE-Plus

- **Dataset**: [facebook/tgve_plus](https://huggingface.co/datasets/facebook/tgve_plus) (HuggingFace)
- **License**: CC-BY-NC 4.0
- **Paper**: Video Editing via Factorized Diffusion Distillation

### VIVID-10M

- **Paper**: [VIVID-10M: A Dataset and Baseline for Versatile and Interactive Video Local Editing](https://arxiv.org/abs/2411.15260) (Nov 2024)
- **Size**: 9.7M samples
- **Details**: First large-scale hybrid image-video local editing dataset.

---

## Video Subject Swapping

Datasets specifically designed for replacing subjects/objects in videos while preserving motion.

### VideoSwap Dataset

- **Paper**: [VideoSwap: Customized Video Subject Swapping with Interactive Semantic Point Correspondence](https://arxiv.org/abs/2312.02087) (CVPR 2024)
- **Code**: [showlab/VideoSwap](https://github.com/showlab/VideoSwap)
- **Website**: [videoswap.github.io](https://videoswap.github.io/)
- **Size**: 30 videos + 13 customized concepts → ~300 edited results
- **Categories**: Human (10), Animal (10), Object (10)
- **Source**: Shutterstock, DAVIS
- **Highlight**: Uses semantic point correspondences for motion-preserving swaps.

### DreamSwapV-Benchmark

- **Paper**: [DreamSwapV: Mask-guided Subject Swapping for Any Customized Video Editing](https://arxiv.org/abs/2508.14465)
- **Details**: First benchmark tailored to video subject swapping. Based on HumanVID dataset.
- **Metrics**: 5 indicators inherited from VBench + user study (reference detail, subject interaction, visual fidelity).

### MIVE Dataset

- **Paper**: [MIVE: New Design and Benchmark for Multi-Instance Video Editing](https://arxiv.org/html/2412.12877v1)
- **Details**: Multi-instance video editing with diverse scenarios.
- **Metric**: Cross-Instance Accuracy (CIA) Score for evaluating editing leakage.

---

## Video Inpainting

Datasets for video inpainting and object removal tasks.

| Dataset | Size | Type | Venue |
|---------|------|------|-------|
| **VPData** | 390K+ clips | Training | SIGGRAPH 2025 |
| **VPBench** | 100 videos | Evaluation | SIGGRAPH 2025 |
| **ROVI** | 5,650 videos | Training + Eval | CVPR 2024 |

### VPData / VPBench

- **Paper**: [VideoPainter: Any-length Video Inpainting and Editing with Plug-and-Play Context Control](https://arxiv.org/abs/2503.05639) (SIGGRAPH 2025)
- **Dataset**: [TencentARC/VPData](https://huggingface.co/datasets/TencentARC/VPData), [TencentARC/VPBench](https://huggingface.co/datasets/TencentARC/VPBench) (HuggingFace)
- **Code**: [TencentARC/VideoPainter](https://github.com/TencentARC/VideoPainter)
- **VPData**: 390K+ clips (866.7+ hours), largest video inpainting dataset with segmentation masks and dense captions
- **VPBench**: 100 videos (6s standard) + 16 videos (30s+ long) for evaluation
- **Affiliations**: Tencent ARC Lab, CUHK, U Tokyo, U Macau

### ROVI Dataset

- **Paper**: [Towards Language-Driven Video Inpainting via Multimodal Large Language Models](https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Towards_Language-Driven_Video_Inpainting_via_Multimodal_Large_Language_Models_CVPR_2024_paper.pdf) (CVPR 2024)
- **Code**: [jianzongwu/Language-Driven-Video-Inpainting](https://github.com/jianzongwu/Language-Driven-Video-Inpainting)
- **Size**: 5,650 videos, 9,091 inpainting results
- **Details**: First dataset for language-guided video inpainting (LVI) and interactive video inpainting (IVI)

---

## Video Editing Quality Assessment

Benchmarks focused on evaluating the quality of edited videos.

| Benchmark | Size | Metrics | Human Annotation | Venue |
|-----------|------|---------|------------------|-------|
| **EditBoard** | - | 9 metrics, 4 dimensions | - | AAAI 2025 |
| **VE-Bench** | 8 models | MOS | 24 annotators | AAAI 2025 |
| **TDVE-DB** | 3,857 videos | 3 dimensions | 173,565 ratings | arXiv 2025.05 |
| VEBench | Large-scale | VEScore (MLLM) | - | OpenReview |

### EditBoard

- **Paper**: [EditBoard: Towards a Comprehensive Evaluation Benchmark for Text-Based Video Editing Models](https://ojs.aaai.org/index.php/AAAI/article/view/33754/35909) (AAAI 2025)
- **ArXiv**: [arxiv.org/abs/2409.09668](https://arxiv.org/abs/2409.09668)
- **Details**: First comprehensive evaluation benchmark for text-based video editing models.
- **Metrics**: 9 automatic metrics across 4 dimensions, including 3 new metrics for fidelity assessment.
- **Highlight**: Task-oriented benchmark facilitating objective model comparison.

### VE-Bench

- **Paper**: [VE-Bench: Subjective-Aligned Benchmark Suite for Text-Driven Video Editing](https://ojs.aaai.org/index.php/AAAI/article/view/32763/34918) (AAAI 2025)
- **Code**: [littlespray/VE-Bench](https://github.com/littlespray/VE-Bench)
- **Details**: First quality assessment dataset for video editing. 8 models, 24 human annotators, MOS scores.
- **Categories**: Real-world scenes, CG-rendered scenes, AIGC-generated scenes.

### VEBench

- **Paper**: [VEBench: Towards Comprehensive and Automatic Evaluation for Text-guided Video Editing](https://openreview.net/forum?id=nZNWrzDBHG)
- **Details**: Large-scale benchmark with VEScore (MLLM-based automatic evaluation). Best model achieves only 3.18/5.

### TDVE-DB

- **Paper**: [TDVE-Assessor: Benchmarking and Evaluating the Quality of Text-Driven Video Editing](https://arxiv.org/html/2505.19535)
- **Size**: 3,857 edited videos from 12 models, 8 editing categories
- **Annotations**: 173,565 human ratings across 3 dimensions (quality, alignment, structural consistency)

---

## Evaluation Metrics

Common metrics used across video editing benchmarks.

### Automatic Metrics

| Metric | Description | Used In |
|--------|-------------|---------|
| **CLIP Score** | Text-video alignment via CLIP embeddings | TokenFlow, StableVideo, IVEBench |
| **LPIPS** | Perceptual similarity (LPIPS-P: vs original, LPIPS-T: temporal) | General |
| **FVD** | Fréchet Video Distance | VEditBench, IVEBench |
| **FID** | Fréchet Inception Distance (per-frame) | VEditBench |
| **CIA Score** | Cross-Instance Accuracy (editing leakage) | MIVE |
| **VEScore** | MLLM-based automatic evaluation | VEBench |

### Human Evaluation Dimensions

- **Textual Faithfulness**: How well the edit matches the text prompt
- **Temporal Consistency**: Smoothness and coherence across frames
- **Visual Quality**: Overall aesthetic and clarity
- **Structure Preservation**: Maintaining unedited regions
- **Subject Identity**: Consistency of swapped subject appearance

---

## Summary by Task Relevance

### For Product Video Template Transfer (PVTT)

Most relevant datasets for the task of transferring product video templates:

| Priority | Dataset | Relevance |
|----------|---------|-----------|
| ⭐⭐⭐⭐⭐ | VideoSwap Dataset | Subject swapping with motion preservation |
| ⭐⭐⭐⭐⭐ | DreamSwapV-Benchmark | Mask-guided subject swapping evaluation |
| ⭐⭐⭐⭐⭐ | VEditBench | Object Swap task, diverse video lengths |
| ⭐⭐⭐⭐⭐ | FiVE-Bench | Fine-grained object replacement evaluation |
| ⭐⭐⭐⭐ | OpenVE-3M | Large-scale training (3M), Local Change category |
| ⭐⭐⭐⭐ | Señorita-2M | Large-scale training (2M), object swap/removal |
| ⭐⭐⭐⭐ | Ditto-1M | High-quality training (1M), local editing |
| ⭐⭐⭐⭐ | LOVEU-TGVE-2024 | Object Change editing type |
| ⭐⭐⭐ | IVEBench | Modern evaluation protocol with MLLM |
| ⭐⭐⭐ | MIVE Dataset | Multi-instance editing |

### 2024-2025 Highlights

```
Training Data (Million-scale):
├── OpenVE-3M (3M+ triplets, 720P, ByteDance)
├── Señorita-2M (2M pairs, NeurIPS D&B 2025)
├── Ditto-1M (1M triplets, 1280×720)
├── InsViE-1M (1M triplets, ICCV 2025)
├── VIVID-10M (9.7M samples, local editing)
└── VPData (390K clips, video inpainting, SIGGRAPH 2025)

Evaluation Benchmarks:
├── VEditBench (420 videos, 6 tasks, ICLR 2025)
├── IVEBench (600 videos, 12 metrics)
├── FiVE-Bench (100 videos, fine-grained, ICCV 2025)
├── RVEBenchmark (519 queries, reasoning editing)
├── EditBoard (9 metrics, AAAI 2025)
├── TDVE-DB (3,857 videos, 173K human ratings)
└── VPBench (video inpainting, SIGGRAPH 2025)
```

### Gap Analysis

```
Current Gap: No public dataset specifically for product/e-commerce promotional videos.

Recommendation:
1. Evaluation: VideoSwap + VEditBench + FiVE-Bench (Object Swap/Replacement)
2. Training: OpenVE-3M / Señorita-2M / Ditto-1M (Local Change, Object Swap)
3. Metrics: DreamSwapV + IVEBench + FiVE-Acc
4. Build custom product video dataset from e-commerce platforms
```

---

## Contributing

Contributions welcome! Please read the contribution guidelines first.

- Add new datasets with paper links and download URLs
- Update metrics and leaderboard information
- Fix broken links or outdated information

---

## License

[![CC0](https://licensebuttons.net/p/zero/1.0/88x31.png)](https://creativecommons.org/publicdomain/zero/1.0/)

---

## Acknowledgments

This list was compiled for the PVTT (Product Video Template Transfer) research project targeting CVPR 2027.

Last updated: 2026-01-20
