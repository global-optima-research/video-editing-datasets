# Pipeline 重新设计：方案 B - 空镜视频 + 物体插入

## 实验信息

- **日期**: 2026-01-23
- **目标**: 基于已有实验结果，重新设计 PVTT pipeline
- **背景**: 原方案遇到根本性困难，需要 take a step back

---

## 一、原方案回顾与问题

### 1.1 原方案架构

```
源视频 (含商品A)
    → 分割 (SAM2)
    → 背景修复 (移除商品A)
    → 首帧编辑 (替换为商品B)
    → TI2V (同seed生成视频)
    → 配对视频
```

### 1.2 已证伪的假设

| 假设 | 验证结果 | 详情 |
|------|---------|------|
| TI2V 运动由 seed 控制 | ❌ 错误 | 运动由首帧内容决定，无法跨首帧迁移 |
| 光流方法能修复全程存在物体 | ❌ 错误 | 无干净背景可借用，阴影无法移除 |

### 1.3 核心矛盾

**背景修复困境**：
- 光流方法 (ProPainter/DiffuEraser)：时序一致 ✅ 但无法移除阴影 ❌
- 图像方法 (LaMa/FLUX/OmniEraser)：能移除阴影 ✅ 但帧间闪烁 ❌

**运动迁移困境**：
- TI2V 模型的运动生成主要由首帧视觉内容决定
- Prompt 对运动的控制能力有限
- 同 seed 无法保证不同首帧产生相同运动

---

## 二、新方案：空镜视频 + 物体插入

### 2.1 核心思路

**绕过问题而非解决问题**：
- 不再尝试移除物体（避免背景修复困境）
- 不再尝试迁移运动（避免运动迁移困境）
- 从空镜视频开始，分别插入不同商品

### 2.2 新方案架构

```
┌─────────────────────────────────────────────────────────┐
│  空镜视频 (无商品的背景视频)                              │
│       │                                                  │
│       ├──── 定义插入区域 (Box 序列)                       │
│       │                                                  │
│       ├──── 插入商品 X ────→ 视频 A ─┐                   │
│       │                              │                   │
│       └──── 插入商品 Y ────→ 视频 B ─┼──→ 配对数据       │
│                                      │                   │
└─────────────────────────────────────────────────────────┘
```

### 2.3 方案优势

| 优势 | 说明 |
|------|------|
| 无需背景修复 | 起点就是干净背景 |
| 无需运动迁移 | 运动由空镜视频提供，两次插入共享同一运动 |
| 问题简化 | 只需解决"视频物体插入"一个问题 |
| 易于扩展 | 一个空镜视频可配对多个商品 |

### 2.4 新的挑战

| 挑战 | 说明 |
|------|------|
| 空镜视频来源 | 需要获取或拍摄合适的空镜视频 |
| 物体插入质量 | 需要时序一致、光照匹配、形状适配 |
| Box 序列生成 | 如何定义商品在视频中的位置轨迹 |

---

## 三、视频物体插入方法调研

### 3.1 VideoAnyDoor (SIGGRAPH 2025)

**最适合方案 B，但尚未完全开源**

| 项目 | 信息 |
|------|------|
| 论文 | [arxiv.org/abs/2501.01427](https://arxiv.org/abs/2501.01427) |
| GitHub | [github.com/yuanpengtu/VideoAnydoor](https://github.com/yuanpengtu/VideoAnydoor) |
| 开源状态 | ⚠️ 代码整理中，权重未发布 |
| 会议 | SIGGRAPH 2025 (CCF A) |

**核心能力**：
- 输入：参考图 + Box 序列 + 关键点轨迹（可选）
- 输出：插入物体后的视频
- 运动控制：粗粒度 (Box) + 细粒度 (Pixel Warper)
- 零样本泛化，无需微调

**技术架构**：
```
参考图像 → ID Extractor (DINOv2) → 身份特征
                    ↓
关键点轨迹 → Pixel Warper → 像素扭曲
                    ↓
              3D UNet + ControlNet
                    ↓
               合成视频
```

**估计规格**：
- 显存：24GB+
- 分辨率：~1024×1024
- 速度：几分钟/视频

### 3.2 InsertAnywhere (arXiv 2025)

**已开源，4D 几何感知**

| 项目 | 信息 |
|------|------|
| 论文 | [arxiv.org/abs/2512.17504](https://arxiv.org/abs/2512.17504) |
| GitHub | [github.com/myyzzzoooo/InsertAnywhere](https://github.com/myyzzzoooo/InsertAnywhere) |
| 开源状态 | ✅ 代码已开源 |

**核心能力**：
- 4D 场景重建，自动处理遮挡
- 光照感知
- 用户可控位置和尺度

### 3.3 AnyDoor (CVPR 2024)

**VideoAnyDoor 的图像版基础，完全开源**

| 项目 | 信息 |
|------|------|
| 论文 | CVPR 2024 |
| GitHub | [github.com/ali-vilab/AnyDoor](https://github.com/ali-vilab/AnyDoor) |
| 开源状态 | ✅ 完全开源 |

**核心能力**：
- 零样本图像物体插入
- 细节保持好
- 可逐帧处理视频，但需后处理平滑

### 3.4 Add-it (ICLR 2025, NVIDIA)

**训练免费的图像物体插入**

| 项目 | 信息 |
|------|------|
| 论文 | [arxiv.org/abs/2411.07232](https://arxiv.org/abs/2411.07232) |
| GitHub | [github.com/NVlabs/addit](https://github.com/NVlabs/addit) |
| 开源状态 | ✅ 代码已开源 |

**核心能力**：
- 无需训练，使用预训练 diffusion 模型
- 加权 extended-attention

### 3.5 方法对比

| 方法 | 开源 | 视频原生 | Box控制 | 细节保持 | 光照 | 推荐度 |
|------|------|---------|---------|---------|------|--------|
| VideoAnyDoor | ⚠️ 整理中 | ✅ | ✅ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| InsertAnywhere | ✅ | ✅ | ✅ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| AnyDoor | ✅ | ❌ 逐帧 | ✅ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Add-it | ✅ | ❌ 仅图像 | ✅ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

---

## 四、新 Pipeline 设计

### 4.1 完整流程

```
Phase 1: 准备空镜视频
├── 来源：拍摄 / 公开数据集 / 视频生成
├── 要求：干净背景，有合适的放置区域
└── 运镜：固定/平移/旋转均可

Phase 2: 定义插入区域
├── 方法：手动标注 / SAM2 辅助 / 自动检测
├── 输出：Box 序列 (每帧的 bounding box)
└── 可选：关键点轨迹 (用于精细运动控制)

Phase 3: 物体插入
├── 输入：空镜视频 + 商品图片 (去背景) + Box 序列
├── 方法：InsertAnywhere / VideoAnyDoor / AnyDoor
└── 输出：合成视频

Phase 4: 质量评估
├── VLM 自动评估
├── 人工抽检
└── 筛选高质量配对
```

### 4.2 配对生成

```
空镜视频 V
    │
    ├── Box 序列 B (共享)
    │
    ├── 商品 X + V + B → 视频 A
    │
    └── 商品 Y + V + B → 视频 B

配对: (视频A, 视频B, "X→Y")
```

### 4.3 扩展性

一个空镜视频 + N 个商品 = N×(N-1)/2 个配对

例如：
- 10 个空镜视频
- 100 个商品图片
- = 10 × C(100,2) = 10 × 4950 = 49,500 个配对

---

## 五、待验证问题

### 5.1 技术验证

| 问题 | 验证方法 |
|------|---------|
| InsertAnywhere 效果如何？ | 在测试样本上跑 PoC |
| Box 序列如何自动生成？ | 测试固定位置 vs 跟踪 |
| 不同形状商品的适配？ | 测试手链→项链等形状变化 |
| 时序一致性如何？ | 观察视频是否闪烁 |
| 光照匹配如何？ | 观察商品与背景的融合 |

### 5.2 数据验证

| 问题 | 验证方法 |
|------|---------|
| 空镜视频哪里获取？ | 调研公开数据集 / 考虑自行拍摄 |
| 商品图片要求？ | 是否需要去背景？分辨率要求？ |
| 配对质量如何保证？ | VLM 评估 + 人工抽检 |

---

## 六、下一步计划

### 优先级 1：验证物体插入

| 任务 | 目的 |
|------|------|
| 测试 InsertAnywhere | 验证已开源方案的效果 |
| 准备测试数据 | 空镜视频 + 商品图片 |
| 评估插入质量 | 时序一致性、光照匹配、形状适配 |

### 优先级 2：完善 Pipeline

| 任务 | 目的 |
|------|------|
| Box 序列生成方案 | 自动化或半自动化 |
| 批量处理脚本 | 支持大规模生成 |
| 质量筛选机制 | VLM 自动评估 |

### 优先级 3：数据准备

| 任务 | 目的 |
|------|------|
| 空镜视频来源 | 拍摄或获取公开数据 |
| 商品图片收集 | 去背景处理 |
| 数据集规模规划 | 成本和时间估算 |

---

## 七、参考资料

### 论文

- [VideoAnyDoor: High-fidelity Video Object Insertion with Precise Motion Control](https://arxiv.org/abs/2501.01427) - SIGGRAPH 2025
- [InsertAnywhere: Bridging 4D Scene Geometry and Diffusion Models](https://arxiv.org/abs/2512.17504) - arXiv 2025
- [AnyDoor: Zero-shot Object-level Image Customization](https://arxiv.org/abs/2307.09481) - CVPR 2024
- [Add-it: Training-Free Object Insertion in Images](https://arxiv.org/abs/2411.07232) - ICLR 2025

### GitHub

- VideoAnyDoor: https://github.com/yuanpengtu/VideoAnydoor
- InsertAnywhere: https://github.com/myyzzzoooo/InsertAnywhere
- AnyDoor: https://github.com/ali-vilab/AnyDoor
- Add-it: https://github.com/NVlabs/addit

---

**记录人**: Claude
**最后更新**: 2026-01-23
