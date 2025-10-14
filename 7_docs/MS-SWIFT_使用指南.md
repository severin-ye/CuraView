# MS-SWIFT 本地模型使用指南

## 📋 目录
1. [简介](#简介)
2. [环境准备](#环境准备)
3. [安装配置](#安装配置)
4. [本地模型管理](#本地模型管理)
5. [基本使用](#基本使用)
6. [高级功能](#高级功能)
7. [常见问题与解决方案](#常见问题与解决方案)
8. [最佳实践](#最佳实践)

---

## 📖 简介

MS-SWIFT (Scalable lightWeight Infrastructure for Fine-Tuning) 是阿里巴巴 ModelScope 社区提供的官方框架，用于大语言模型和多模态大模型的微调和部署。

### 核心特性
- 🚀 支持 500+ 大语言模型和 200+ 多模态模型
- 🎯 支持训练、推理、评测、量化、部署全流程
- 💡 支持 LoRA、QLoRA、DoRA 等轻量化训练方法
- 🔧 提供 Web UI 界面和丰富的最佳实践

---

## 🛠️ 环境准备

### 系统要求
- **Python**: 3.10+ (推荐)
- **CUDA**: 支持的 CUDA 版本
- **内存**: 根据模型大小调整
- **显存**: 4B模型至少8GB，30B模型建议40GB+

### 硬件支持
- CPU、RTX系列、T4/V100、A10/A100/H100
- Ascend NPU、MPS 等

---

## 📦 安装配置

### 1. 基础安装

```bash
# 使用 pip 安装
pip install ms-swift -U

# 或者从源码安装
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .
```

### 2. 环境依赖

确保以下关键依赖已正确安装：

```bash
# 核心依赖
pip install torch>=2.0
pip install transformers>=4.33
pip install modelscope>=1.23
pip install peft>=0.11,<0.18

# 可选依赖（根据需要安装）
pip install flash_attn  # 注意力优化
pip install vllm>=0.5.1  # 推理加速
pip install deepspeed>=0.14  # 分布式训练
pip install gradio  # Web UI
```

### 3. 依赖问题解决

如果遇到 `bitsandbytes` 或 `triton` 相关错误：

```bash
# 重新安装 bitsandbytes
pip uninstall bitsandbytes
pip install bitsandbytes

# 或者禁用量化功能
export USE_BNB=0
```

---

## 📁 本地模型管理

### 当前可用模型

根据您的环境，有以下本地模型：

```
models/
├── Qwen3-4B-Thinking-2507-FP8/     # 4B 参数，FP8量化
└── Qwen3-30B-A3B-Thinking-2507/    # 30B 参数，标准精度
```

### 模型文件检查

确保每个模型目录包含以下文件：
- `config.json` - 模型配置
- `tokenizer.json` - 分词器配置  
- `model.safetensors` 或 `model-*.safetensors` - 模型权重
- `tokenizer_config.json` - 分词器设置

---

## 🚀 基本使用

### 1. 模型推理 (swift infer)

#### 推理小模型 (4B，推荐开始使用)

```bash
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model ./models/Qwen3-4B-Thinking-2507-FP8 \
    --stream true \
    --infer_backend pt \
    --max_new_tokens 2048 \
    --temperature 0.7
```

#### 推理大模型 (30B，需要更多显存)

```bash
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model ./models/Qwen3-30B-A3B-Thinking-2507 \
    --stream true \
    --infer_backend pt \
    --max_new_tokens 2048 \
    --temperature 0.7 \
    --load_in_8bit true
```

### 2. 模型微调 (swift sft)

#### 微调小模型

```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model ./models/Qwen3-4B-Thinking-2507-FP8 \
    --dataset AI-ModelScope/alpaca-gpt4-data-zh#500 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --output_dir ./output/qwen3-4b-lora \
    --max_length 2048 \
    --system 'You are a helpful assistant.'
```

#### 微调大模型 (使用量化)

```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model ./models/Qwen3-30B-A3B-Thinking-2507 \
    --dataset AI-ModelScope/alpaca-gpt4-data-zh#500 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --load_in_8bit true \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 32 \
    --output_dir ./output/qwen3-30b-lora \
    --max_length 2048
```

### 3. Web UI 界面

启动图形化界面（推荐新用户使用）：

```bash
# 中文界面
SWIFT_UI_LANG=zh swift web-ui

# 英文界面
SWIFT_UI_LANG=en swift web-ui
```

访问 `http://localhost:7860` 使用 Web 界面。

---

## 🎯 高级功能

### 1. 模型部署 (swift deploy)

将模型部署为 API 服务：

```bash
CUDA_VISIBLE_DEVICES=0 swift deploy \
    --model ./models/Qwen3-4B-Thinking-2507-FP8 \
    --infer_backend vllm \
    --port 8000
```

### 2. 模型评测 (swift eval)

评测模型性能：

```bash
CUDA_VISIBLE_DEVICES=0 swift eval \
    --model ./models/Qwen3-4B-Thinking-2507-FP8 \
    --infer_backend pt \
    --eval_backend OpenCompass \
    --eval_dataset ARC_c MMLU
```

### 3. 使用微调后的模型

训练完成后，使用 adapters 进行推理：

```bash
# 推理微调后的模型
CUDA_VISIBLE_DEVICES=0 swift infer \
    --adapters ./output/qwen3-4b-lora/vx-xxx/checkpoint-xxx \
    --stream true \
    --temperature 0.7 \
    --max_new_tokens 2048
```

### 4. 合并 LoRA 权重

```bash
# 合并 LoRA 到基础模型
CUDA_VISIBLE_DEVICES=0 swift export \
    --adapters ./output/qwen3-4b-lora/vx-xxx/checkpoint-xxx \
    --merge_lora true \
    --output_dir ./merged_model
```

### 5. 模型量化

```bash
# AWQ 量化
CUDA_VISIBLE_DEVICES=0 swift export \
    --model ./models/Qwen3-4B-Thinking-2507-FP8 \
    --quant_bits 4 \
    --quant_method awq \
    --dataset AI-ModelScope/alpaca-gpt4-data-zh \
    --output_dir ./qwen3-4b-awq
```

---

## 🔧 常见问题与解决方案

### 1. bitsandbytes 错误

**问题**: `ModuleNotFoundError: No module named 'triton.ops'`

**解决方案**:
```bash
# 方法1: 重新安装依赖
pip uninstall bitsandbytes triton
pip install bitsandbytes triton

# 方法2: 使用 CPU 模式
export USE_BNB=0

# 方法3: 避免使用量化
# 去掉 --load_in_8bit 和 --load_in_4bit 参数
```

### 2. 显存不足

**解决方案**:
```bash
# 启用量化
--load_in_8bit true
# 或者
--load_in_4bit true

# 减少批处理大小
--per_device_train_batch_size 1

# 增加梯度累积
--gradient_accumulation_steps 32

# 使用梯度检查点
--gradient_checkpointing true
```

### 3. CUDA 版本不匹配

**解决方案**:
```bash
# 检查 CUDA 版本
nvidia-smi

# 安装对应版本的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. 模型加载失败

**检查列表**:
- [ ] 模型路径是否正确
- [ ] 模型文件是否完整
- [ ] 权限是否正确
- [ ] 磁盘空间是否充足

---

## 💡 最佳实践

### 1. 硬件选择建议

| 模型大小 | 推荐显存 | 量化选项 | 批处理大小 |
|---------|---------|---------|------------|
| 4B      | 8GB+    | 可选     | 2-4        |
| 7B      | 16GB+   | 推荐     | 1-2        |
| 13B     | 24GB+   | 必需     | 1          |
| 30B+    | 40GB+   | 必需     | 1          |

### 2. 训练参数调优

```bash
# 小模型推荐配置
--lora_rank 8
--lora_alpha 32
--learning_rate 1e-4
--warmup_ratio 0.05

# 大模型推荐配置
--lora_rank 16
--lora_alpha 64
--learning_rate 5e-5
--warmup_ratio 0.1
```

### 3. 数据集建议

```bash
# 使用内置数据集
--dataset AI-ModelScope/alpaca-gpt4-data-zh#1000

# 使用多个数据集
--dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
          'AI-ModelScope/alpaca-gpt4-data-en#500' \
          'swift/self-cognition#500'

# 使用自定义数据集
--dataset /path/to/your/dataset.jsonl
```

### 4. 性能优化

```bash
# 启用 Flash Attention
--use_flash_attn true

# 使用混合精度
--torch_dtype bfloat16

# 多进程数据加载
--dataloader_num_workers 4

# 编译优化
--torch_compile true
```

### 5. 监控和日志

```bash
# 启用 TensorBoard
--logging_dir ./logs
--report_to tensorboard

# 设置保存策略
--save_steps 100
--eval_steps 100
--save_total_limit 3

# 详细日志
--logging_steps 10
```

---

## 📋 常用命令速查

```bash
# 快速推理
swift infer --model ./models/Qwen3-4B-Thinking-2507-FP8

# 快速微调
swift sft --model ./models/Qwen3-4B-Thinking-2507-FP8 --dataset alpaca

# 启动 Web UI
swift web-ui

# 部署服务
swift deploy --model ./models/Qwen3-4B-Thinking-2507-FP8

# 模型评测
swift eval --model ./models/Qwen3-4B-Thinking-2507-FP8

# 查看支持的模型
swift list-models

# 查看支持的数据集
swift list-datasets

# 查看帮助
swift --help
swift sft --help
swift infer --help
```

---

## 🔗 相关资源

- [MS-SWIFT GitHub](https://github.com/modelscope/ms-swift)
- [中文文档](https://swift.readthedocs.io/zh-cn/latest/)
- [英文文档](https://swift.readthedocs.io/en/latest/)
- [ModelScope 社区](https://modelscope.cn/)
- [问题反馈](https://github.com/modelscope/ms-swift/issues)

---

**最后更新**: 2025年10月12日

**注意**: 使用前请确保您的环境满足硬件和软件要求，并根据实际情况调整参数配置。