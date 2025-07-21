# Migrating Autoregressive Image Generation without Vector Quantization to the Audio Domain

## 项目简介
本项目致力于将最新的自回归扩散生成范式（MAR: Masked Autoregressive Generation without Vector Quantization）从图像领域迁移到音频领域，构建高质量、可控的文本到语音（TTS）系统。我们以Nanospeech为基础，融合了MAR的Diffusion Loss和自回归采样机制，显著提升了语音生成的自然度和灵活性。

---

## 技术路线
1. **Mel频谱建模**：以mel频谱为生成目标，兼容Nanospeech的声码器和数据流。
2. **Diffusion Loss集成**：将MAR中的DiffLoss模块迁移到Nanospeech，作为mel频谱生成的主要损失函数和采样机制。
3. **自回归掩码采样**：借鉴MAR的掩码自回归采样流程，逐步生成mel频谱帧，提升长序列建模能力。
4. **Classifier-Free Guidance**：支持CFG提升条件生成的多样性和可控性。
5. **训练与推理流程重构**：训练阶段联合优化matching flow与diffusion loss，推理阶段支持ODE采样与diffusion refinement。

---

## 主要文件结构

```
ar_tts/
├── nanospeech/                # 主体TTS系统（已集成Diffusion/AR采样）
│   ├── nanospeech_torch.py    # 主模型
│   ├── diffusion_enhanced.py  # Diffusion Loss与采样模块（迁移自mar/models/diffloss.py等）
│   ├── diffusion_utils.py     # Diffusion工具函数（迁移自mar/diffusion/）
│   ├── trainer_torch.py       # 训练器，支持联合损失与新采样流程
│   ├── generate.py            # 推理脚本，支持diffusion-enhanced采样
│   └── ...
├── mar/                      # 原MAR代码库（部分文件迁移/复用）
├── examples/                 # 训练与推理示例
└── readme.md                 # 本说明文档
```

---

## 迁移与集成细节
- **DiffLoss与采样**：直接从MAR迁移`diffloss.py`、`diffusion_utils.py`等文件，并适配mel频谱输入。
- **模型结构**：在Nanospeech的`Nanospeech`类中集成Diffusion分支，forward和sample方法均支持diffusion流程。
- **训练流程**：`trainer_torch.py`支持matching flow与diffusion loss的联合训练，损失权重可调。
- **推理流程**：`generate.py`支持ODE采样与diffusion refinement，提升生成音质与多样性。
- **配置与依赖**：保留Nanospeech原有依赖，新增MAR相关diffusion依赖。

---

## 训练与推理用法

### 安装依赖
```bash
pip install -U torch torchaudio einops tqdm soundfile sounddevice safetensors
# 如需MLX支持，安装mlx相关依赖
```

### 训练模型
```bash
python nanospeech/examples/train_nanospeech.py
# 或自定义训练脚本，参考examples/
```

### 生成语音
```bash
python -m nanospeech.generate --text "你好，世界！" --use-diffusion
```

---

## 主要参数与API说明

- `--use-diffusion`：启用diffusion增强采样
- `--cfg`：Classifier-Free Guidance强度
- `--steps`：ODE采样步数
- `--num_iter`：自回归采样迭代次数
- `--output`：输出音频文件路径
- `--voice`：参考音色选择
- `--ref-audio`/`--ref-text`：自定义参考音频与文本

**API调用示例：**
```python
from nanospeech.nanospeech_torch import Nanospeech, DiT
from nanospeech.diffusion_enhanced import DiffLoss

model = Nanospeech(..., diffusion_module=DiffLoss(...))
mel = model.sample(..., use_diffusion=True)
```

---

## 注意事项
- 迁移的Diffusion模块需确保输入输出shape与mel频谱一致。
- 训练时建议先warmup matching flow，再逐步引入diffusion loss。
- 推理时可灵活切换ODE采样与diffusion refinement，权衡速度与质量。
- 依赖项需包含MAR迁移相关的diffusion工具包。

---

## 参考文献
- [Autoregressive Image Generation without Vector Quantization (MAR)](https://arxiv.org/abs/2406.11838)
- [Nanospeech: A simple, hackable text-to-speech system](https://github.com/lucasnewman/nanospeech)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2206.08791)
