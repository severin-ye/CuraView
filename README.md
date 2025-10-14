# MS-Swift 模型微调完整工具包

本工具包提供了使用 MS-Swift 框架进行大语言模型微调的完整解决方案，包括训练、推理和部署的全套脚本。

## 📋 目录结构

```
/home/work/hd/
├── finetune_trainer.py      # 主微调脚本
├── inference_test.py        # 推理测试脚本
├── deploy_model.py          # 模型部署脚本
├── configs/                 # 配置文件目录
│   ├── lora_sft.json       # LoRA微调配置
│   ├── qlora_sft.json      # QLoRA微调配置
│   ├── full_sft.json       # 全参数微调配置
│   └── multimodal_sft.json # 多模态微调配置
├── test_questions.json     # 测试问题集
└── README.md               # 本文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装MS-Swift
pip install ms-swift -U

# 可选：安装推理加速后端
pip install vllm -U          # vLLM后端
pip install sglang -U        # SGlang后端
pip install lmdeploy -U      # LMDeploy后端

# 可选：安装深度并行训练
pip install deepspeed -U     # DeepSpeed
```

### 2. 基础微调

#### LoRA微调（推荐）
```bash
# 使用配置文件
python finetune_trainer.py --config configs/lora_sft.json

# 或命令行参数
python finetune_trainer.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --train_type "lora" \
    --dataset "AI-ModelScope/alpaca-gpt4-data-zh#1000" \
    --output_dir "./output/lora_test" \
    --num_train_epochs 3 \
    --learning_rate 1e-4
```

#### QLoRA微调（节省显存）
```bash
python finetune_trainer.py --config configs/qlora_sft.json
```

#### 全参数微调
```bash
python finetune_trainer.py --config configs/full_sft.json
```

### 3. 推理测试

#### 批量测试
```bash
# 测试微调后的模型
python inference_test.py --checkpoint ./output/lora_test/checkpoint-100

# 使用自定义问题集
python inference_test.py \
    --checkpoint ./output/lora_test/checkpoint-100 \
    --questions-file test_questions.json \
    --output results.json
```

#### 交互式测试
```bash
python inference_test.py \
    --checkpoint ./output/lora_test/checkpoint-100 \
    --mode interactive
```

#### 单个问题测试
```bash
python inference_test.py \
    --checkpoint ./output/lora_test/checkpoint-100 \
    --mode single \
    --question "你是谁？"
```

### 4. 模型部署

#### 部署LoRA模型
```bash
python deploy_model.py \
    --checkpoint ./output/lora_test/checkpoint-100 \
    --type lora \
    --port 8000 \
    --infer-backend vllm
```

#### 部署全参数模型
```bash
python deploy_model.py \
    --checkpoint ./output/full_test/checkpoint-100 \
    --type full \
    --port 8000
```

## 📖 详细使用说明

### 微调脚本 (finetune_trainer.py)

这是主要的微调脚本，支持多种微调方式：

#### 主要功能
- **LoRA微调**: 轻量级参数高效微调
- **QLoRA微调**: 量化+LoRA，进一步节省显存
- **全参数微调**: 更新所有模型参数
- **多模态微调**: 支持图像、视频、音频等多模态数据
- **分布式训练**: 支持多GPU、多机训练
- **流式数据**: 支持大规模数据集的流式加载

#### 使用方式

**方式1：使用配置文件（推荐）**
```bash
python finetune_trainer.py --config configs/lora_sft.json
```

**方式2：命令行参数**
```bash
python finetune_trainer.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --train_type "lora" \
    --dataset "AI-ModelScope/alpaca-gpt4-data-zh#1000" \
    --output_dir "./output/test" \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --save_steps 100 \
    --eval_steps 100
```

**方式3：Python API**
```python
from finetune_trainer import FineTuner

trainer = FineTuner("configs/lora_sft.json")
trainer.run_training()
```

#### 重要参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `model` | 基础模型路径或ID | 必需 |
| `train_type` | 训练类型：lora/qlora/full | "lora" |
| `dataset` | 训练数据集 | 必需 |
| `output_dir` | 输出目录 | "./output" |
| `num_train_epochs` | 训练轮数 | 3 |
| `learning_rate` | 学习率 | 1e-4 |
| `lora_rank` | LoRA秩 | 8 |
| `lora_alpha` | LoRA alpha | 32 |
| `max_length` | 最大序列长度 | 2048 |

### 推理测试脚本 (inference_test.py)

用于测试微调后模型的性能和效果。

#### 功能特性
- **自动检测模型类型**: 自动识别LoRA或全参数模型
- **批量测试**: 使用预定义问题集进行批量测试
- **交互式测试**: 实时对话测试
- **结果保存**: 自动保存测试结果到JSON文件
- **自定义问题**: 支持自定义测试问题集

#### 测试模式

**1. 批量测试模式**
```bash
python inference_test.py \
    --checkpoint ./output/lora_test/checkpoint-100 \
    --mode batch \
    --output test_results.json
```

**2. 交互式测试模式**
```bash
python inference_test.py \
    --checkpoint ./output/lora_test/checkpoint-100 \
    --mode interactive
```

**3. 单问题测试模式**
```bash
python inference_test.py \
    --checkpoint ./output/lora_test/checkpoint-100 \
    --mode single \
    --question "解释一下什么是人工智能"
```

### 部署脚本 (deploy_model.py)

用于将微调后的模型部署为API服务。

#### 支持的后端
- **pt**: PyTorch原生后端
- **vllm**: vLLM高性能推理后端
- **sglang**: SGLang推理后端
- **lmdeploy**: LMDeploy推理后端

#### 部署示例

**单模型部署**
```bash
python deploy_model.py \
    --checkpoint ./output/lora_test/checkpoint-100 \
    --port 8000 \
    --infer-backend vllm \
    --served-model-name "my-custom-model"
```

**多LoRA部署**
```bash
# 首先创建多LoRA配置文件 multi_lora_config.json
{
  "model1": "./output/lora1/checkpoint-100",
  "model2": "./output/lora2/checkpoint-100"
}

# 然后部署
python deploy_model.py \
    --type multi-lora \
    --multi-lora-config multi_lora_config.json \
    --port 8000 \
    --infer-backend vllm
```

#### 客户端调用示例

部署完成后，可以通过以下方式调用：

**使用curl**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-custom-model",
    "messages": [{"role": "user", "content": "你好，你是谁？"}],
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

**使用Python OpenAI客户端**
```python
from openai import OpenAI

client = OpenAI(
    api_key='EMPTY',
    base_url='http://localhost:8000/v1'
)

response = client.chat.completions.create(
    model='my-custom-model',
    messages=[{'role': 'user', 'content': '你好，你是谁？'}],
    max_tokens=512,
    temperature=0.7
)

print(response.choices[0].message.content)
```

## ⚙️ 配置文件说明

### LoRA配置 (configs/lora_sft.json)
适用于大多数微调场景，平衡效果和资源消耗。

### QLoRA配置 (configs/qlora_sft.json)
适用于显存受限的环境，使用量化技术进一步节省显存。

### 全参数配置 (configs/full_sft.json)
适用于追求最佳效果的场景，需要更多计算资源。

### 多模态配置 (configs/multimodal_sft.json)
适用于多模态模型的微调，支持图像、视频、音频数据。

## 🎯 最佳实践

### 1. 数据集准备
```python
# 自定义数据集格式
[
    {
        "system": "你是一个有用的助手。",
        "conversations": [
            {"from": "user", "value": "你好"},
            {"from": "assistant", "value": "你好！我是AI助手，有什么可以帮助你的吗？"}
        ]
    }
]
```

### 2. 超参数调优建议

| 参数 | 小模型(7B) | 大模型(30B+) | 说明 |
|------|------------|--------------|------|
| `learning_rate` | 1e-4 | 5e-5 | 大模型用更小学习率 |
| `lora_rank` | 8-16 | 32-64 | 大模型可用更大rank |
| `batch_size` | 4-8 | 1-2 | 根据显存调整 |
| `gradient_accumulation_steps` | 4-8 | 16-32 | 保证有效batch size |

### 3. 显存优化技巧

**节省显存的配置组合：**
```json
{
    "train_type": "qlora",
    "quantization_bit": 4,
    "gradient_checkpointing": true,
    "deepspeed": "zero2",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16
}
```

### 4. 多GPU训练

**数据并行**
```bash
# 使用所有可用GPU
python finetune_trainer.py --config configs/lora_sft.json

# 指定特定GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune_trainer.py --config configs/lora_sft.json
```

**DeepSpeed加速**
```bash
python finetune_trainer.py \
    --config configs/lora_sft.json \
    --deepspeed zero2
```

## 🔧 故障排除

### 常见问题

**1. 显存不足 (CUDA Out of Memory)**
- 减小 `per_device_train_batch_size`
- 增加 `gradient_accumulation_steps`
- 使用 `gradient_checkpointing`
- 尝试 QLoRA 或 DeepSpeed

**2. 训练速度慢**
- 检查是否启用了 `flash_attn`
- 使用更大的 `batch_size`
- 考虑多GPU训练
- 尝试 `pack_sequence_as_prompt` 提高GPU利用率

**3. 模型效果不理想**
- 增加训练数据量
- 调整学习率
- 增加训练轮数
- 检查数据质量

**4. 推理速度慢**
- 使用 vLLM 或其他推理加速后端
- 启用 `flash_attn`
- 考虑模型量化

### 调试技巧

**启用详细日志**
```bash
export SWIFT_DEBUG=1
python finetune_trainer.py --config configs/lora_sft.json
```

**验证数据加载**
```python
from finetune_trainer import FineTuner

trainer = FineTuner("configs/lora_sft.json")
trainer.verify_data()  # 检查数据格式
```

## 📊 性能对比

| 方法 | 显存占用 | 训练速度 | 效果质量 | 部署兼容性 |
|------|----------|----------|----------|------------|
| 全参数 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| LoRA | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| QLoRA | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个工具包！

## 📄 许可证

本项目遵循MIT许可证。

## 🔗 相关链接

- [MS-Swift官方文档](https://github.com/modelscope/ms-swift)
- [ModelScope模型库](https://modelscope.cn/models)
- [Hugging Face模型库](https://huggingface.co/models)

---

**快速开始示例：**

```bash
# 1. 克隆或下载本工具包
# 2. 安装依赖
pip install ms-swift -U

# 3. 开始LoRA微调
python finetune_trainer.py --config configs/lora_sft.json

# 4. 测试微调效果
python inference_test.py --checkpoint ./output/xxx/checkpoint-xxx

# 5. 部署模型服务
python deploy_model.py --checkpoint ./output/xxx/checkpoint-xxx
```

现在您就可以开始使用这个完整的微调工具包了！🎉