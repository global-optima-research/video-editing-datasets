# Video Object Insertion 调研

针对 PVTT (Product Video Template Transfer) 的 Video Object Insertion 相关工作调研。

---

## 问题定义

**Video Object Insertion** = 将物体插入到现有视频中，保持：
- 物体外观一致性
- 运动轨迹合理性
- 光照/阴影真实性
- 时序连贯性

与 Video Editing 的区别：
- Video Editing: 修改现有内容
- Video Object Insertion: 添加新内容（可完全不同形状）

---

## 核心工作

### 1. VideoAnyDoor (SIGGRAPH 2025) ⭐⭐⭐⭐⭐

> **最相关的工作**：零样本视频物体插入，支持精确运动控制

**论文**: [VideoAnydoor: High-fidelity Video Object Insertion with Precise Motion Control](https://arxiv.org/abs/2501.01427)

**代码**: [github.com/yuanpengtu/VideoAnydoor](https://github.com/yuanpengtu/VideoAnydoor)

#### 核心架构

```
┌─────────────────────────────────────────────────────────────┐
│  输入                                                        │
│  ├── 背景视频 (可选已 inpaint)                               │
│  ├── 参考物体图片 (去背景)                                   │
│  ├── Mask 序列 (物体应出现的位置)                            │
│  └── 关键点轨迹 (可选，精细运动控制)                         │
├─────────────────────────────────────────────────────────────┤
│  ID Extractor                                                │
│  ├── 提取物体的全局 identity 特征                            │
│  └── 注入到 3D UNet 的 cross-attention                       │
├─────────────────────────────────────────────────────────────┤
│  Pixel Warper (核心创新)                                     │
│  ├── 输入: 参考图 + 关键点 + 轨迹                            │
│  ├── 按轨迹 warp 像素细节                                    │
│  └── 与 diffusion UNet 融合                                  │
├─────────────────────────────────────────────────────────────┤
│  3D UNet + ControlNet                                        │
│  ├── 时序一致的视频生成                                      │
│  └── 多尺度特征注入                                          │
└─────────────────────────────────────────────────────────────┘
```

#### 运动控制方式

| 控制方式 | 精度 | 使用场景 |
|----------|------|----------|
| **Box 序列** | 粗粒度 | 整体位置/大小变化 |
| **关键点轨迹** | 细粒度 | 精确运动控制 |

#### 对 PVTT 的价值

- ✅ **零样本**: 不需要每视频训练
- ✅ **Box 序列控制**: 与我们的 BBox 轨迹想法一致！
- ✅ **细节保持**: Pixel Warper 保持物体细节
- ✅ **已验证场景**: 支持 virtual try-on（类似商品替换）

#### 局限

- 需要干净的 Mask 序列
- 物体需要去背景的参考图
- 光照适配能力待验证

---

### 2. Anything in Any Scene (arXiv 2024)

> 物理真实感导向，强调光照和阴影

**论文**: [Anything in Any Scene: Photorealistic Video Object Insertion](https://arxiv.org/abs/2401.17509)

**来源**: XPeng Motors (自动驾驶场景)

#### 三阶段 Pipeline

```
Stage 1: Geometric Placement (几何放置)
├── 确定每帧的放置位置
├── 考虑遮挡关系
└── 光流跟踪保持轨迹稳定

Stage 2: Lighting Simulation (光照模拟)
├── 估计天空和环境光分布
└── 生成真实阴影

Stage 3: Style Transfer (风格迁移)
└── 精细化调整，最大化真实感
```

#### 光流轨迹稳定

```
问题: 单帧独立估计位置 → 抖动、不自然

解决:
1. 计算连续帧间光流
2. 跟踪场景中的点运动
3. 优化相机位姿 (最小化重投影误差)
4. 稳定物体位置
```

#### 对 PVTT 的价值

- ✅ **光流稳定**: 解决物体运动抖动问题
- ✅ **光照估计**: 适配场景光照
- ⚠️ 主要针对自动驾驶场景（车辆插入）

---

### 3. ObjectDrop (ECCV 2024)

> 图像版本，但核心思想重要：反事实推理

**论文**: [ObjectDrop: Bootstrapping Counterfactuals for Photorealistic Object Removal and Insertion](https://arxiv.org/abs/2403.18818)

**来源**: Google

#### 核心思想

```
传统方法的问题:
物体插入后，缺少真实的阴影、反射、遮挡

ObjectDrop 解决方案:
1. 收集 "反事实" 数据: 移除物体前后的真实图片对
2. 训练 removal model: 学习移除物体及其效果
3. 反向应用: 用 removal 的逆过程做 insertion
```

#### 对 PVTT 的价值

- ✅ **阴影/反射处理**: 关键的真实感因素
- ⚠️ 图像版本，需扩展到视频

---

### 4. InsertAnywhere (2025)

> 4D 场景感知的视频物体插入

**特点**:
- 4D 场景几何重建
- 遮挡关系自动处理
- 跨帧一致的物体放置

---

### 5. MTV-Inpaint (arXiv 2025.03)

> 多任务长视频 Inpainting

**论文**: [MTV-Inpaint: Multi-Task Long Video Inpainting](https://arxiv.org/abs/2503.11412)

**特点**:
- 支持 Box 轨迹控制物体运动
- 长视频支持
- 统一的 inpainting + insertion

---

## 方法对比

| 方法 | 运动控制 | 光照处理 | 零样本 | 视频长度 |
|------|----------|----------|--------|----------|
| **VideoAnyDoor** | Box + 关键点 | 有限 | ✅ | 中等 |
| Anything in Any Scene | 光流稳定 | ✅ 强 | ✅ | 长 |
| ObjectDrop | - | ✅ 强 | ❌ | 图像 |
| InsertAnywhere | 4D 几何 | 有限 | ✅ | 中等 |
| MTV-Inpaint | Box 轨迹 | 有限 | ✅ | 长 |

---

## PVTT 数据集构建方案 (更新)

基于 Video Object Insertion 的新思路：

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: 准备                                                │
│                                                             │
│  模板视频 (手表) → SAM2 分割 → Mask 序列                     │
│                           ↘                                 │
│                            Video Inpainting → 干净背景视频   │
├─────────────────────────────────────────────────────────────┤
│  Step 2: Object Insertion (使用 VideoAnyDoor)                │
│                                                             │
│  输入:                                                       │
│  ├── 干净背景视频                                            │
│  ├── 新商品图片 (项链，去背景)                               │
│  ├── Mask 序列 (从源视频提取)                                │
│  └── (可选) 关键点轨迹                                       │
│                                                             │
│  输出: 项链自然融入的视频                                    │
├─────────────────────────────────────────────────────────────┤
│  Step 3: 质量过滤                                            │
│                                                             │
│  VLM 评估:                                                   │
│  ├── 商品外观一致性                                          │
│  ├── 运动自然度                                              │
│  ├── 光照匹配度                                              │
│  └── 时序连贯性                                              │
└─────────────────────────────────────────────────────────────┘
```

### 优势

1. **形状完全自由**: 手表 → 项链 ✅
2. **无需运动迁移**: 直接用 Mask 序列控制位置
3. **零样本**: VideoAnyDoor 不需要每视频训练
4. **Pipeline 简单**: 分割 → Inpainting → Insertion

### 待验证

1. VideoAnyDoor 对商品细节的保持能力？
2. 光照适配是否足够？（可能需要 Anything in Any Scene 的光照模块）
3. 长视频性能？

---

## 下一步建议

1. **PoC 验证**: 用 VideoAnyDoor 跑几个手表→项链的样例
2. **评估光照**: 检查是否需要额外的光照适配
3. **Pipeline 集成**: SAM2 + Video Inpainting + VideoAnyDoor

---

## 参考文献

- [VideoAnyDoor](https://arxiv.org/abs/2501.01427) - SIGGRAPH 2025
- [Anything in Any Scene](https://arxiv.org/abs/2401.17509) - arXiv 2024
- [ObjectDrop](https://arxiv.org/abs/2403.18818) - ECCV 2024
- [InsertAnywhere](https://liner.com/review/insertanywhere-bridging-4d-scene-geometry-and-diffusion-models-for-realistic) - 2025
- [MTV-Inpaint](https://arxiv.org/abs/2503.11412) - arXiv 2025

---

Last updated: 2026-01-20
