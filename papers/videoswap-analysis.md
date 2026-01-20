# VideoSwap 深度分析

> Customized Video Subject Swapping with Interactive Semantic Point Correspondence (CVPR 2024)

**论文**: [arxiv.org/abs/2312.02087](https://arxiv.org/abs/2312.02087)
**代码**: [github.com/showlab/VideoSwap](https://github.com/showlab/VideoSwap)
**项目**: [videoswap.github.io](https://videoswap.github.io/)

---

## 核心贡献

1. **语义点对应** (Semantic Point Correspondence): 用少量关键点描述运动
2. 支持**形状变化**的物体替换 (如飞机→直升机)
3. 交互式编辑: 点移除、拖拽调整

---

## 核心洞察

> **密集对应方法 (如光流) 会限制形状变化。**
> **少量语义点 (如飞机的机翼、机头、机尾) 足以描述运动轨迹。**

```
光流: 每个像素的对应 → 形状必须一致
       ↓
语义点: 5-10 个关键点 → 只约束关键位置，形状可变
```

---

## 三阶段 Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: Semantic Point Extraction (语义点提取)                 │
│                                                                  │
│  用户在关键帧定义 K 个语义点                                     │
│       ↓                                                          │
│  Co-Tracker 跨帧传播 → 获得 N 帧的点轨迹                         │
│       ↓                                                          │
│  DIFT 提取点嵌入 → 捕获语义信息                                  │
├─────────────────────────────────────────────────────────────────┤
│  Stage 2: Semantic Point Registration (语义点注册)               │
│                                                                  │
│  可学习 MLP 将点嵌入投影为稀疏运动特征                           │
│       ↓                                                          │
│  与 UNet 中间特征 element-wise 相加                              │
│       ↓                                                          │
│  关键: 只在高时间步 (T/2 以后) 优化，强调语义对齐                │
├─────────────────────────────────────────────────────────────────┤
│  Stage 3: Subject Swapping (主体替换)                            │
│                                                                  │
│  源视频 → VAE 编码 → DDIM 反演 → 噪声                            │
│       ↓                                                          │
│  替换文本 prompt 中的主体                                        │
│       ↓                                                          │
│  语义点引导的 DDIM 去噪                                          │
│       ↓                                                          │
│  Latent Blending: Cross-attention mask 保持背景                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 各阶段详解

### Stage 1: Semantic Point Extraction

**输入**: 源视频 + 用户标注的语义点

**处理流程**:

```python
# 伪代码
1. 用户在关键帧标注 K 个语义点 (如飞机: 机头、左翼尖、右翼尖、机尾)
2. Co-Tracker 跨帧追踪:
   - 输入: 关键帧 + K 个点坐标
   - 输出: N 帧 × K 个点的 2D 轨迹
3. DIFT 提取语义嵌入:
   - 对每帧每个点提取特征嵌入
   - 捕获点的语义信息 (不只是位置)
```

**点轨迹示例**:
```
帧 1:  (x1, y1) (x2, y2) (x3, y3) (x4, y4)  ← K=4 个点
帧 2:  (x1', y1') (x2', y2') ...
...
帧 N:  (x1'', y1'') ...
```

### Stage 2: Semantic Point Registration

**目的**: 将语义点信息注入扩散模型

**方法**:
```
点嵌入 (K × D)
     ↓
可学习 MLP
     ↓
稀疏运动特征 (与 UNet 特征同维度)
     ↓
与 UNet 中间特征 element-wise 相加
```

**关键设计**:
- 只在高时间步 (t > T/2) 进行优化
- 强调语义对齐，而非低级细节

### Stage 3: Subject Swapping

**流程**:
```
源视频 latent
     ↓
DDIM 反演 → 噪声
     ↓
替换 text prompt (如 "airplane" → "helicopter")
     ↓
语义点引导的 DDIM 去噪
     ↓
Latent Blending (用 cross-attention mask 保持背景)
     ↓
输出视频
```

---

## 用户交互方式

### 三种交互模式

| 模式 | 操作 | 适用场景 | 示例 |
|------|------|----------|------|
| **Direct adoption** | 直接使用源轨迹 | 相似形状替换 | 猫→狗 |
| **Point removal** | 移除不相关的点 | 形状差异大 | 飞机→直升机 (移除机翼点) |
| **Dragging** | 手动调整点位置 | 自定义运动 | 改变运动轨迹 |

### Point Removal 示例

```
飞机的语义点:
• 机头 ─────────────── ✅ 保留 (直升机也有)
• 左翼尖 ─────────────── ❌ 移除 (直升机没有固定翼)
• 右翼尖 ─────────────── ❌ 移除
• 机尾 ─────────────── ✅ 保留

直升机只需跟随: 机头 + 机尾 的轨迹
```

### Dragging 机制

**工具**: Layered Neural Atlas

**流程**:
1. 用户在关键帧拖拽点到新位置
2. 计算位移量
3. 通过 Jacobian 矩阵传播位移到其他帧
4. 生成新的点轨迹

**成本**: ~2 小时额外训练

---

## 关键技术

### Point Patch Loss

**问题**: 如何防止结构信息泄漏到目标物体？

**解决方案**:
```
只约束语义点周围的局部 patch 重建
     ↓
点之间的区域不受约束
     ↓
目标物体可以有不同形状
```

### 稀疏特征注入

**设计**:
```
特征图上只有点轨迹位置有嵌入
     ↓
背景区域不受引导
     ↓
减少过拟合，允许形状变化
```

### Latent Blending

**目的**: 保持背景不变

**方法**:
```
Cross-attention map → 物体区域 mask
     ↓
编辑后 latent × mask + 源 latent × (1-mask)
     ↓
只有物体区域被替换
```

---

## 实验设置

### 数据集

| 类别 | 数量 | 来源 |
|------|------|------|
| Human | 10 | Shutterstock, DAVIS |
| Animal | 10 | Shutterstock, DAVIS |
| Object | 10 | Shutterstock, DAVIS |
| **Total** | **30** | - |

### 自定义概念

13 个概念，使用 ED-LoRA 训练

### 对比方法

- Tune-A-Video
- FateZero
- Rerender-A-Video

---

## 实验结果

### 定量指标

| 方法 | CLIP-Text ↑ | CLIP-Temporal ↑ | CLIP-Image ↑ |
|------|-------------|-----------------|--------------|
| Tune-A-Video | - | - | - |
| FateZero | - | - | - |
| **VideoSwap** | **Best** | **Best** | **Best** |

### 用户研究 (1000 MTurk 问卷)

| 对比 | VideoSwap 偏好率 |
|------|------------------|
| vs. Tune-A-Video | 84% |
| vs. FateZero | 78% |
| vs. Rerender-A-Video | 82% |

**评估维度**:
- Subject Identity (主体身份)
- Motion Alignment (运动对齐)
- Temporal Consistency (时序一致性)
- Overall Preference (整体偏好)

---

## 局限性

| 局限 | 数据 |
|------|------|
| 预处理时间 | ~4 分钟/视频 |
| Drag 编辑额外成本 | ~2 小时 (Layered Neural Atlas) |
| 点跟踪失败情况 | 自遮挡、极端视角变化 |
| 推理时间 | ~50 秒/编辑 |
| 用户标注 | 需要手动定义语义点 |

---

## 对 PVTT 的适用性

### 优势

| 特性 | 价值 |
|------|------|
| 形状变化支持 | ✅ 手表→项链 可行 |
| 稀疏运动表示 | ✅ 比光流更灵活 |
| 背景保持 | ✅ Latent Blending |

### 局限

| 特性 | 问题 |
|------|------|
| 用户标注 | ❌ 需要手动定义语义点 |
| 自动化困难 | ❌ 数据集构建难以规模化 |
| 预处理成本 | ⚠️ 4分钟/视频 |

### 可能的自动化方案

1. **VLM 自动识别语义点**:
   ```
   输入: 商品图片
   VLM: "识别这个手表的关键点: 表盘中心、12点位置、6点位置、表带扣"
   输出: 4 个语义点坐标
   ```

2. **商品类别模板**:
   ```
   手表: 表盘中心、表带扣、表冠
   项链: 吊坠中心、链条两端
   手机: 四个角、摄像头
   ```

3. **通用关键点检测**:
   ```
   使用 X-Pose 或类似模型自动检测物体关键点
   ```

---

## 与其他方法对比

| 维度 | VideoSwap | VideoAnyDoor | Motion LoRA |
|------|-----------|--------------|-------------|
| 形状变化 | ✅ 支持 | ✅ 支持 | ⚠️ 有限 |
| 运动控制 | 语义点 | Box + 关键点 | 隐式学习 |
| 自动化 | ❌ 需标注 | ✅ 零样本 | ✅ 自动 |
| 训练需求 | 无 | 无 | 每视频训练 |

**结论**: 对于 PVTT，VideoAnyDoor 的 Box 序列控制可能比 VideoSwap 的语义点更易自动化。

---

## 参考文献

- [VideoSwap](https://arxiv.org/abs/2312.02087) - CVPR 2024
- [Co-Tracker](https://co-tracker.github.io/) - 点跟踪
- [DIFT](https://diffusionfeatures.github.io/) - 扩散特征
- [Layered Neural Atlas](https://layered-neural-atlases.github.io/) - 视频分解

---

Last updated: 2026-01-20
