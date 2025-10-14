# 使用指南

## 快速开始

### 1. 环境准备

首先确保您的环境满足要求：

```bash
# 检查Python版本（需要3.8+）
python --version

# 检查CUDA（推荐）
nvidia-smi

# 安装依赖
pip install -r requirements.txt
```

### 2. 第一次训练

使用预设配置进行LoRA微调：

```bash
cd /home/work/hd
python 4_scripts/main.py train --preset lora --model Qwen/Qwen2.5-7B-Instruct
```

### 3. 推理测试

训练完成后测试模型：

```bash
python 4_scripts/main.py infer --checkpoint ./output --mode interactive
```

## 详细教程

### 训练工作流

#### 1. 准备数据

创建训练数据集文件（JSON格式）：

```json
[
  {
    "system": "你是一个专业的医疗助手。",
    "input": "什么是高血压？",
    "output": "高血压是指血压持续高于正常值的慢性疾病..."
  },
  {
    "system": "你是一个专业的医疗助手。",
    "input": "如何预防糖尿病？",
    "output": "预防糖尿病的主要方法包括：1. 健康饮食..."
  }
]
```

#### 2. 配置训练参数

创建训练配置文件 `train_config.json`：

```json
{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "train_type": "lora",
  "dataset": ["path/to/your/dataset.json"],
  "output_dir": "./output/my_model",
  "num_train_epochs": 3,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 8,
  "learning_rate": 1e-4,
  "max_length": 2048,
  "lora_rank": 8,
  "lora_alpha": 32,
  "save_steps": 100,
  "logging_steps": 10
}
```

#### 3. 启动训练

```bash
python 4_scripts/train.py --config train_config.json
```

或者使用命令行参数：

```bash
python 4_scripts/train.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --dataset "medical_dataset.json" \
    --train_type lora \
    --num_train_epochs 3 \
    --output_dir "./output/medical_model"
```

#### 4. 监控训练进度

训练过程中，您可以查看：

- 控制台输出的实时训练日志
- `6_output/` 目录下的详细训练日志
- 训练损失和学习率变化

### 推理工作流

#### 1. 单条推理

```bash
# 交互式对话
python 4_scripts/infer.py --checkpoint ./output/my_model --mode interactive

# 单次问答
python 4_scripts/infer.py \
    --checkpoint ./output/my_model \
    --mode single \
    --question "什么是机器学习？"
```

#### 2. 批量推理

准备问题文件 `questions.txt`：

```
什么是人工智能？
机器学习的主要方法有哪些？
深度学习和传统机器学习的区别是什么？
```

执行批量推理：

```bash
python 4_scripts/infer.py \
    --checkpoint ./output/my_model \
    --mode batch \
    --input_file questions.txt \
    --output_file answers.json
```

#### 3. 配置推理参数

```bash
python 4_scripts/infer.py \
    --checkpoint ./output/my_model \
    --mode single \
    --question "解释量子计算的原理" \
    --max_tokens 1000 \
    --temperature 0.7 \
    --top_p 0.9
```

### 部署工作流

#### 1. 本地部署

```bash
python 4_scripts/deploy.py \
    --checkpoint ./output/my_model \
    --preset local \
    --port 8000
```

#### 2. 服务器部署

```bash
python 4_scripts/deploy.py \
    --checkpoint ./output/my_model \
    --preset server \
    --host 0.0.0.0 \
    --port 8000 \
    --infer_backend vllm
```

#### 3. 测试部署的服务

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "max_tokens": 100
    }'
```

### 评估工作流

#### 1. 标准评估

```bash
python 4_scripts/evaluate.py \
    --checkpoint ./output/my_model \
    --preset standard \
    --test_data test_dataset.json
```

#### 2. 基准测试

```bash
python 4_scripts/evaluate.py \
    --checkpoint ./output/my_model \
    --benchmark \
    --dataset_name "commonsense_qa"
```

#### 3. 自定义评估

```bash
python 4_scripts/evaluate.py \
    --checkpoint ./output/my_model \
    --custom_metrics "bleu,rouge,bertscore" \
    --test_data custom_test.json
```

## 智能体使用

### 1. RAG智能体

创建RAG智能体配置：

```json
{
  "agent_type": "rag",
  "knowledge_base_path": "./knowledge_base.json",
  "embedding_model": "all-MiniLM-L6-v2",
  "top_k": 5,
  "similarity_threshold": 0.5
}
```

启动RAG智能体：

```python
from agents.rag_agent import RAGAgent

rag_agent = RAGAgent(
    agent_name="医疗助手",
    checkpoint_path="./output/medical_model",
    knowledge_base_path="./medical_kb.json"
)

# 添加文档到知识库
rag_agent.add_document(
    content="阿司匹林是一种常用的解热镇痛药...",
    title="阿司匹林使用指南",
    source="医疗手册"
)

# 开始对话
response = rag_agent.chat("阿司匹林的副作用有哪些？")
print(response)
```

### 2. 偏好学习智能体

```python
from agents.preference_agent import PreferenceAgent

pref_agent = PreferenceAgent(
    agent_name="个性化助手",
    checkpoint_path="./output/my_model"
)

# 设置用户
pref_agent.set_current_user("user_123")

# 对话并收集反馈
response = pref_agent.chat("推荐一些好书")
print(response)

# 处理用户反馈
feedback = "我更喜欢科幻小说，不太喜欢历史书籍"
pref_agent.process_feedback(feedback)

# 后续对话会考虑用户偏好
response2 = pref_agent.chat("再推荐一些书")
print(response2)
```

## 常见使用场景

### 1. 领域特定模型训练

#### 医疗助手训练

```bash
# 1. 准备医疗数据集
python 4_scripts/train.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --dataset "medical_qa_dataset.json" \
    --train_type lora \
    --output_dir "./output/medical_assistant" \
    --num_train_epochs 5 \
    --learning_rate 5e-5

# 2. 评估医疗知识
python 4_scripts/evaluate.py \
    --checkpoint "./output/medical_assistant" \
    --test_data "medical_test.json" \
    --custom_metrics "medical_accuracy"

# 3. 部署医疗助手
python 4_scripts/deploy.py \
    --checkpoint "./output/medical_assistant" \
    --preset server \
    --port 8001
```

#### 代码助手训练

```bash
# 1. 代码训练
python 4_scripts/train.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --dataset "code_dataset.json" \
    --train_type qlora \
    --output_dir "./output/code_assistant" \
    --max_length 4096

# 2. 代码生成测试
python 4_scripts/infer.py \
    --checkpoint "./output/code_assistant" \
    --mode single \
    --question "写一个Python快速排序算法"
```

### 2. 多模态模型训练

```bash
python 4_scripts/train.py \
    --model "Qwen/Qwen2-VL-7B-Instruct" \
    --dataset "multimodal_dataset.json" \
    --train_type multimodal \
    --output_dir "./output/vision_assistant"
```

### 3. 大规模部署

```bash
# 多GPU部署
python 4_scripts/deploy.py \
    --checkpoint "./output/my_model" \
    --preset distributed \
    --tensor_parallel_size 4 \
    --host 0.0.0.0 \
    --port 8000
```

## 配置优化

### 1. 训练配置优化

#### 内存优化

```json
{
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 16,
  "gradient_checkpointing": true,
  "dataloader_pin_memory": false,
  "fp16": true
}
```

#### 速度优化

```json
{
  "per_device_train_batch_size": 4,
  "gradient_accumulation_steps": 4,
  "dataloader_num_workers": 4,
  "group_by_length": true,
  "bf16": true
}
```

### 2. 推理配置优化

#### 质量优先

```json
{
  "temperature": 0.3,
  "top_p": 0.8,
  "repetition_penalty": 1.1,
  "max_new_tokens": 2048
}
```

#### 速度优先

```json
{
  "temperature": 0.7,
  "top_p": 0.9,
  "max_new_tokens": 512,
  "do_sample": true
}
```

## 故障排除

### 1. 训练问题

#### 内存不足

```bash
# 减少批次大小
--per_device_train_batch_size 1
--gradient_accumulation_steps 16

# 启用梯度检查点
--gradient_checkpointing true

# 使用混合精度
--fp16 true
```

#### 训练不收敛

```bash
# 调整学习率
--learning_rate 5e-5

# 增加warmup步数
--warmup_ratio 0.1

# 检查数据质量
python 4_scripts/tools.py --validate_dataset your_dataset.json
```

### 2. 推理问题

#### 推理速度慢

```bash
# 使用更高效的后端
--infer_backend vllm

# 减少最大token数
--max_tokens 512

# 使用量化模型
--quantization int8
```

#### 输出质量差

```bash
# 调整生成参数
--temperature 0.5
--top_p 0.8
--repetition_penalty 1.1

# 检查模型是否正确加载
python 4_scripts/tools.py --check_model your_checkpoint
```

### 3. 部署问题

#### 服务无法启动

```bash
# 检查端口占用
lsof -i :8000

# 检查模型路径
python 4_scripts/tools.py --validate_checkpoint your_checkpoint

# 查看详细错误日志
python 4_scripts/deploy.py --checkpoint your_checkpoint --debug
```

## 性能调优

### 1. 硬件利用率优化

```bash
# 查看GPU使用情况
python 4_scripts/main.py info --gpu

# 优化批次大小
python 4_scripts/tools.py --find_optimal_batch_size

# 内存使用分析
python 4_scripts/tools.py --memory_profile
```

### 2. 训练效率优化

```bash
# 数据加载优化
--dataloader_num_workers 8
--dataloader_pin_memory true

# 模型并行
--tensor_parallel_size 2

# 使用更高效的优化器
--optimizer adamw_torch_fused
```

### 3. 推理效率优化

```bash
# 批量推理
python 4_scripts/infer.py \
    --mode batch \
    --batch_size 32 \
    --input_file large_questions.txt

# 使用KV缓存
--use_kv_cache true

# 推理加速
--infer_backend lmdeploy
```

## 最佳实践总结

### 1. 项目组织

- 为每个项目创建独立的输出目录
- 使用有意义的模型和数据集命名
- 定期备份重要的检查点

### 2. 实验管理

- 记录每次实验的配置参数
- 使用版本控制跟踪配置变更
- 保存实验结果用于对比分析

### 3. 资源管理

- 监控GPU内存使用情况
- 及时清理不需要的模型文件
- 合理安排训练和推理任务

### 4. 质量保证

- 在小数据集上验证配置
- 定期评估模型性能
- 建立自动化测试流程