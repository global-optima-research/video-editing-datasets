# Awesome Video Editing Datasets

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of public datasets and benchmarks for video editing research, including text-guided video editing, video subject swapping, and image-to-video generation.

## Contents

- [Text-Guided Video Editing (TGVE)](#text-guided-video-editing-tgve)
- [Instruction-Guided Video Editing (IVE)](#instruction-guided-video-editing-ive) 🆕
- [Video Subject Swapping](#video-subject-swapping)
- [Video Editing Quality Assessment](#video-editing-quality-assessment)
- [Video Editing Understanding](#video-editing-understanding) 🆕
- [Image-to-Video (I2V)](#image-to-video-i2v)
- [Large-Scale Video Editing](#large-scale-video-editing)
- [Foundation Video Datasets](#foundation-video-datasets)
- [Evaluation Metrics](#evaluation-metrics)

---

## Text-Guided Video Editing (TGVE)

Datasets for evaluating text-driven video editing methods.

| Dataset | Size | Resolution | Edit Types | Venue | Source |
|---------|------|------------|------------|-------|--------|
| **VEditBench** 🆕 | 420 videos | Varied | 6 tasks (insert/remove/swap/scene/motion/style) | ICLR 2025 | Real-world |
| **OpenVE-Bench** 🆕 | - | 720P | 8 categories (SA + NSA edits) | arXiv 2025.12 | OpenVE-3M |
| LOVEU-TGVE-2024 | 200 videos | Varied (2s-48s) | Insert, Remove, Change, Scene, Motion, Style | CVPR 2024 | Panda-70M |
| LOVEU-TGVE-2023 | 76 videos, 304 prompts | 480×480 | Style, Background, Object, Multiple | CVPR 2023 | DAVIS, YouTube, Videvo |
| BalanceCC | 100 videos | - | Creative & Controllable | CVPR 2024 | Real-world scenes |
| TGVE-Plus | - | - | Factorized editing | 2024 | Meta Research |

### VEditBench 🆕

- **Paper**: [VEditBench: Holistic Benchmark for Text-Guided Video Editing](https://openreview.net/forum?id=6325Jzc9eR) (ICLR 2025)
- **Size**: 420 real-world videos (300 short 2-4s + 120 long 10-20s)
- **Tasks**: Object Insertion, Object Removal, Object Swap, Scene Replacement, Motion Change, Style Translation
- **Metrics**: 9 evaluation dimensions for semantic fidelity and visual quality
- **Highlight**: Most comprehensive TGVE benchmark with diverse video lengths.

### OpenVE-Bench 🆕

- **Paper**: [OpenVE-3M: A Large-Scale High-Quality Dataset for Instruction-Guided Video Editing](https://arxiv.org/abs/2512.07826) (arXiv Dec 2025)
- **Details**: Universal, multi-category benchmark evaluating 3 key dimensions with high human alignment.
- **Categories**: Spatially-aligned (Global Style, Background Change, Local Change/Remove/Add, Subtitles) + Non-spatially-aligned (Camera Multi-Shot, Creative Edit)

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

---

## Instruction-Guided Video Editing (IVE)

Large-scale datasets for training instruction-based video editing models. 🆕

| Dataset | Size | Resolution | Categories | Venue |
|---------|------|------------|------------|-------|
| **OpenVE-3M** | 3M+ triplets | 720P (1280×720) | 8 categories | arXiv 2025.12 |
| **InsViE-1M** | 1M triplets | High-res | Instruction-based | ICCV 2025 |
| **IVEBench** | 600 videos | ≥2K source | 8 categories, 35 subcategories | arXiv 2025.10 |
| VIVID-10M | 9.7M samples | - | Local editing | arXiv 2024.11 |

### OpenVE-3M 🆕

- **Paper**: [OpenVE-3M: A Large-Scale High-Quality Dataset for Instruction-Guided Video Editing](https://arxiv.org/abs/2512.07826) (arXiv Dec 2025)
- **Dataset**: [Lewandofski/OpenVE-3M](https://huggingface.co/datasets/Lewandofski/OpenVE-3M) (HuggingFace)
- **Size**: 3,000,000+ video editing triplets, 65-129 frames per video
- **Resolution**: 720P (1280×720 / 720×1280)
- **Categories**: Spatially-aligned (Global Style, Background Change, Local Change/Remove/Add, Subtitles Edit) + Non-spatially-aligned (Camera Multi-Shot Edit, Creative Edit)
- **Avg Instruction Length**: 40.6 words
- **License**: CC-BY-NC-4.0
- **Affiliations**: ByteDance & Zhejiang University

### InsViE-1M 🆕

- **Paper**: [InsViE-1M: Effective Instruction-based Video Editing with Elaborate Dataset Construction](https://arxiv.org/abs/2503.20287) (ICCV 2025)
- **Code**: [langmanbusi/InsViE](https://github.com/langmanbusi/insvie)
- **Size**: 1M high-quality instruction-video editing triplets
- **Pipeline**: Two-stage editing-filtering with GPT-4o quality control
- **Note**: First model built upon video generation models for instruction-based editing.

### IVEBench 🆕

- **Paper**: [IVEBench: Modern Benchmark Suite for Instruction-Guided Video Editing Assessment](https://arxiv.org/abs/2510.11647) (arXiv Oct 2025)
- **Size**: 600 high-quality source videos, 32-1024 frames
- **Tasks**: 8 categories with 35 subcategories
- **Metrics**: 12 metrics across 3 dimensions (video quality, instruction compliance, video fidelity)
- **Source**: Pexels, Mixkit, UltraVideo (≥2K resolution)
- **Highlight**: Integrates both traditional metrics and MLLM-based assessments.

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

## Video Editing Quality Assessment

Benchmarks focused on evaluating the quality of edited videos.

| Benchmark | Size | Metrics | Human Annotation | Venue |
|-----------|------|---------|------------------|-------|
| **EditBoard** 🆕 | - | 9 metrics, 4 dimensions | - | AAAI 2025 |
| **VE-Bench** | 8 models | MOS | 24 annotators | AAAI 2025 |
| **TDVE-DB** | 3,857 videos | 3 dimensions | 173,565 ratings | arXiv 2025.05 |
| VEBench | Large-scale | VEScore (MLLM) | - | OpenReview |

### EditBoard 🆕

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

### FiVE (Fine-Grained Video Editing)

- **Dataset**: [LIMinghan/FiVE-Fine-Grained-Video-Editing-Benchmark](https://huggingface.co/datasets/LIMinghan/FiVE-Fine-Grained-Video-Editing-Benchmark) (HuggingFace)
- **Size**: 420 high-quality prompt pairs, 6 fine-grained editing tasks
- **Details**: Structured captions by GPT-4o (object category, action, background, camera movement)

---

## Video Editing Understanding

Benchmarks for evaluating Video LLMs on understanding video editing concepts. 🆕

### VEU-Bench 🆕

- **Paper**: [VEU-Bench: Towards Comprehensive Understanding of Video Editing](https://arxiv.org/abs/2504.17828) (CVPR 2025)
- **Website**: [CVPR 2025 Poster](https://cvpr.thecvf.com/virtual/2025/poster/34180)
- **Size**: 50K high-quality data points (45,154 train + 4,382 test)
- **Tasks**: 19 fine-grained tasks across 10 dimensions and 3 levels (Recognition, Reasoning, Judging)
- **Dimensions**: Intra-frame features (shot size) to inter-shot attributes (cut types, transitions)
- **Finding**: Current Vid-LLMs struggle significantly; some perform worse than random choice.
- **Model**: Oscars - VEU expert model outperforms open-source Vid-LLMs by 28.3%.

---

## Image-to-Video (I2V)

Benchmarks for evaluating image-conditioned video generation.

### VBench / VBench-I2V

- **Paper**: [VBench: Comprehensive Benchmark Suite for Video Generative Models](https://vchitect.github.io/VBench-project/) (CVPR 2024 Highlight)
- **Code**: [Vchitect/VBench](https://github.com/Vchitect/VBench)
- **Leaderboard**: 40 T2V models, 12 I2V models
- **I2V Metrics**: Subject Consistency, Background Consistency, Camera Motion, Motion Smoothness, Dynamic Degree, Aesthetic Quality

### AIGCBench

- **Paper**: [AIGCBench: Comprehensive Evaluation of Image-to-Video Content Generated by AI](https://www.sciencedirect.com/science/article/pii/S2772485924000048)
- **Website**: [benchcouncil.org/AIGCBench](https://www.benchcouncil.org/AIGCBench/)
- **Details**: Open-domain image-text dataset, 11 metrics across 4 dimensions (control-video alignment, motion effects, temporal consistency, video quality)

### UI2V-Bench

- **Paper**: [UI2V-Bench: An Understanding-based Image-to-Video Generation Benchmark](https://arxiv.org/html/2509.24427) (2025)
- **Details**: Tests fine-grained subject-action alignment and world knowledge for event prediction.

### Dynamic-I2V / DIVE

- **Paper**: [Dynamic-I2V: Exploring Image-to-Video Generation](https://arxiv.org/pdf/2505.19901)
- **Metric**: DIVE metric for motion diversity, controllability, and fidelity evaluation.

---

## Large-Scale Video Editing

Large-scale datasets for training video editing models. See also [Instruction-Guided Video Editing (IVE)](#instruction-guided-video-editing-ive) for OpenVE-3M (3M), InsViE-1M (1M).

| Dataset | Size | Type | Paper |
|---------|------|------|-------|
| **OpenVE-3M** 🆕 | 3M+ triplets | Instruction-based | [arXiv 2512.07826](https://arxiv.org/abs/2512.07826) |
| **InsViE-1M** 🆕 | 1M triplets | Instruction-based | [ICCV 2025](https://arxiv.org/abs/2503.20287) |
| VIVID-10M | 9.7M samples | Local editing | [arXiv 2411.15260](https://arxiv.org/abs/2411.15260) |

### VIVID-10M

- **Paper**: [VIVID-10M: A Dataset and Baseline for Versatile and Interactive Video Local Editing](https://arxiv.org/abs/2411.15260) (Nov 2024)
- **Size**: 9.7M samples
- **Details**: First large-scale hybrid image-video local editing dataset. Covers wide range of video editing tasks.

---

## Foundation Video Datasets

General video datasets commonly used as source videos for editing benchmarks.

### DAVIS (Densely Annotated VIdeo Segmentation)

- **Paper**: [The 2017 DAVIS Challenge on Video Object Segmentation](https://arxiv.org/abs/1704.00675) (CVPR 2016/2017)
- **Website**: [davischallenge.org](https://davischallenge.org/)
- **Code**: [davisvideochallenge](https://github.com/davisvideochallenge)
- **Size**: DAVIS 2016 (50 videos, single object), DAVIS 2017 (150 videos, multiple objects)
- **Resolution**: 480p (480×854)
- **Note**: In maintenance mode, results integrated into paperswithcode.

### Panda-70M

- **Details**: Source for LOVEU-TGVE-2024 videos. Large-scale video dataset.

### Shutterstock / Videvo

- **Details**: Commercial/stock video sources used in VideoSwap and LOVEU-TGVE datasets.

---

## Evaluation Metrics

Common metrics used across video editing benchmarks.

### Automatic Metrics

| Metric | Description | Used In |
|--------|-------------|---------|
| **CLIP Score** | Text-video alignment via CLIP embeddings | TokenFlow, StableVideo |
| **LPIPS** | Perceptual similarity (LPIPS-P: vs original, LPIPS-T: temporal) | General |
| **FVD** | Fréchet Video Distance | I2V-Bench |
| **FID** | Fréchet Inception Distance (per-frame) | I2V-Bench |
| **FC** | Frame Consistency | I2V-Bench |
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
| ⭐⭐⭐⭐⭐ | VEditBench 🆕 | Object Swap task, diverse video lengths |
| ⭐⭐⭐⭐ | OpenVE-3M 🆕 | Large-scale training data, Local Change category |
| ⭐⭐⭐⭐ | LOVEU-TGVE-2024 | Object Change editing type |
| ⭐⭐⭐⭐ | VBench-I2V | I2V evaluation metrics (Subject/Background Consistency) |
| ⭐⭐⭐ | IVEBench 🆕 | Modern evaluation protocol with MLLM |
| ⭐⭐⭐ | MIVE Dataset | Multi-instance editing |

### 2025 Highlights

```
🆕 Key 2025 Additions:

Training Data:
├── OpenVE-3M (3M+ triplets, 720P, ByteDance)
├── InsViE-1M (1M triplets, ICCV 2025)
└── IVEBench (600 videos, 8 categories)

Evaluation:
├── VEditBench (420 videos, 6 tasks, ICLR 2025)
├── EditBoard (9 metrics, AAAI 2025)
├── VEU-Bench (50K data, CVPR 2025, Vid-LLM understanding)
└── TDVE-DB (3,857 videos, 173K human ratings)
```

### Gap Analysis

```
Current Gap: No public dataset specifically for product/e-commerce promotional videos.

Recommendation:
1. Use VideoSwap Dataset + VEditBench (Object Swap) as baseline comparison
2. Train on OpenVE-3M (Local Change category)
3. Adopt VBench-I2V + DreamSwapV + IVEBench metrics for evaluation
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
