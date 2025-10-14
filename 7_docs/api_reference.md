# API 参考文档

## 核心API接口

### TrainingManager

训练管理器，提供统一的模型训练接口。

#### 初始化

```python
from training.trainer import TrainingManager

trainer = TrainingManager(config_path=None)
```

**参数：**
- `config_path` (str, optional): 配置文件路径

#### 方法

##### `train_lora(**kwargs) -> str`

执行LoRA微调训练。

**参数：**
- `model` (str): 基础模型路径
- `dataset` (List[str]): 训练数据集列表
- `output_dir` (str): 输出目录
- `num_train_epochs` (int): 训练轮数
- `learning_rate` (float): 学习率
- `lora_rank` (int): LoRA秩
- `lora_alpha` (int): LoRA alpha参数

**返回：**
- `str`: 训练输出目录路径

**示例：**
```python
result = trainer.train_lora(
    model="Qwen/Qwen2.5-7B-Instruct",
    dataset=["medical_qa_dataset"],
    output_dir="./output",
    num_train_epochs=3,
    learning_rate=1e-4
)
```

##### `train_qlora(**kwargs) -> str`

执行QLoRA微调训练。

##### `train_full_params(**kwargs) -> str`

执行全参数微调训练。

##### `train_multimodal(**kwargs) -> str`

执行多模态微调训练。

### InferenceManager

推理管理器，提供统一的模型推理接口。

#### 初始化

```python
from inference.inference import InferenceManager

inference_manager = InferenceManager(
    checkpoint_path="./model_checkpoint",
    config_path=None
)
```

#### 方法

##### `load_model(checkpoint_path=None)`

加载模型用于推理。

##### `infer_single(question, **config) -> str`

单条推理。

**参数：**
- `question` (str): 输入问题
- `max_tokens` (int): 最大生成token数
- `temperature` (float): 生成温度
- `top_p` (float): 核采样概率

**返回：**
- `str`: 推理结果

##### `infer_batch(questions, **config) -> List[str]`

批量推理。

##### `interactive_chat()`

启动交互式对话。

### DeploymentManager

部署管理器，提供统一的模型部署接口。

#### 初始化

```python
from deployment.deploy import DeploymentManager

deployment_manager = DeploymentManager(config_path=None)
```

#### 方法

##### `deploy_single_model(checkpoint_path, deploy_config=None) -> Dict[str, Any]`

部署单个模型。

**参数：**
- `checkpoint_path` (str): 检查点路径
- `deploy_config` (dict): 部署配置

**返回：**
- `dict`: 部署信息

##### `stop_deployment()`

停止部署服务。

##### `get_deployment_status() -> Dict[str, Any]`

获取部署状态。

### EvaluationManager

评估管理器，提供统一的模型评估接口。

#### 方法

##### `evaluate_model_performance(inference_func, test_data, task_type) -> Dict[str, Any]`

评估模型性能。

##### `benchmark_on_dataset(inference_func, dataset_name, sample_size=None) -> Dict[str, Any]`

在数据集上执行基准测试。

## 智能体API

### BaseAgent

基础智能体类，所有专业智能体的父类。

#### 初始化

```python
from base_agent import BaseAgent

class MyAgent(BaseAgent):
    def get_system_prompt(self) -> str:
        return "你是一个helpful assistant"
    
    def process_input(self, user_input: str, context=None) -> str:
        return self.infer(user_input)

agent = MyAgent("MyAgent")
```

#### 方法

##### `load_model(checkpoint_path=None)`

加载模型。

##### `chat(user_input, context=None, save_history=True) -> str`

对话接口。

##### `get_conversation_history(limit=None) -> List[Dict[str, Any]]`

获取对话历史。

##### `interactive_session()`

启动交互式会话。

### RAGAgent

RAG智能体，支持知识库检索的对话智能体。

#### 初始化

```python
from rag_agent import RAGAgent

rag_agent = RAGAgent(
    agent_name="RAG助手",
    checkpoint_path="./model",
    knowledge_base_path="./knowledge_base.json"
)
```

#### 方法

##### `load_knowledge_base(kb_path)`

加载知识库。

##### `retrieve_documents(query, top_k=None) -> List[Dict[str, Any]]`

检索相关文档。

##### `add_document(content, title, source="manual_add")`

添加新文档到知识库。

### PreferenceAgent

偏好学习智能体，基于用户反馈进行偏好学习。

#### 方法

##### `set_current_user(user_id)`

设置当前用户。

##### `process_feedback(feedback, interaction_index=-1) -> str`

处理用户反馈。

##### `get_preference_summary() -> Dict[str, Any]`

获取偏好学习摘要。

## 工具类API

### ConfigLoader

配置加载器。

```python
from config_loader import ConfigLoader

loader = ConfigLoader()
config = loader.load_config("config.json")
```

### Logger

日志系统。

```python
from logger import Logger

logger = Logger("MyApp").get_logger()
logger.info("这是一条信息日志")
logger.error("这是一条错误日志")
```

### GPUManager

GPU管理器。

```python
from gpu_manager import GPUManager

gpu_manager = GPUManager()
gpu_info = gpu_manager.get_gpu_info()
available_gpus = gpu_manager.get_available_gpus()
```

## 命令行接口

### 主入口

```bash
python 4_scripts/main.py <command> [options]
```

#### 训练命令

```bash
# 基础训练
python 4_scripts/main.py train --preset lora --model Qwen/Qwen2.5-7B-Instruct

# 自定义训练
python 4_scripts/main.py train \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset medical_qa \
    --train_type lora \
    --num_train_epochs 3 \
    --output_dir ./output
```

#### 推理命令

```bash
# 交互式推理
python 4_scripts/main.py infer --checkpoint ./output --mode interactive

# 单条推理
python 4_scripts/main.py infer \
    --checkpoint ./output \
    --mode single \
    --question "什么是人工智能？"

# 批量推理
python 4_scripts/main.py infer \
    --checkpoint ./output \
    --mode batch \
    --input_file questions.txt \
    --output_file answers.json
```

#### 部署命令

```bash
# 本地部署
python 4_scripts/main.py deploy --checkpoint ./output --preset local

# 服务器部署
python 4_scripts/main.py deploy \
    --checkpoint ./output \
    --preset server \
    --port 8000 \
    --host 0.0.0.0
```

#### 评估命令

```bash
# 标准评估
python 4_scripts/main.py evaluate \
    --checkpoint ./output \
    --preset standard \
    --test_data test.json

# 基准测试
python 4_scripts/main.py evaluate \
    --checkpoint ./output \
    --benchmark
```

## 配置文件格式

### 训练配置

```json
{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "train_type": "lora",
  "dataset": ["medical_qa_dataset"],
  "output_dir": "./output",
  "num_train_epochs": 3,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 8,
  "learning_rate": 1e-4,
  "max_length": 2048,
  "lora_rank": 8,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "save_steps": 100,
  "logging_steps": 10,
  "warmup_ratio": 0.03,
  "weight_decay": 0.1,
  "gradient_checkpointing": true
}
```

### 部署配置

```json
{
  "host": "0.0.0.0",
  "port": 8000,
  "infer_backend": "vllm",
  "temperature": 0.7,
  "max_new_tokens": 2048,
  "top_p": 0.9,
  "tensor_parallel_size": 1,
  "max_model_len": 4096
}
```

### 智能体配置

```json
{
  "agent_type": "rag",
  "knowledge_base_path": "./knowledge_base.json",
  "embedding_model": "all-MiniLM-L6-v2",
  "top_k": 5,
  "similarity_threshold": 0.5,
  "max_context_length": 2000
}
```

## 错误处理

### 异常类型

- `ConfigError`: 配置相关错误
- `ModelError`: 模型相关错误
- `TrainingError`: 训练相关错误
- `InferenceError`: 推理相关错误
- `DeploymentError`: 部署相关错误

### 错误处理示例

```python
try:
    result = trainer.train_lora(**config)
except TrainingError as e:
    logger.error(f"训练失败: {e}")
    # 处理错误
except ConfigError as e:
    logger.error(f"配置错误: {e}")
    # 处理错误
```

## 最佳实践

### 1. 配置管理

- 使用配置文件而不是硬编码参数
- 为不同环境准备不同的配置文件
- 使用环境变量覆盖敏感配置

### 2. 错误处理

- 总是处理可能的异常
- 记录详细的错误信息
- 提供用户友好的错误消息

### 3. 资源管理

- 及时释放不需要的模型
- 监控GPU内存使用
- 使用合适的批次大小

### 4. 日志记录

- 记录关键操作的开始和结束
- 使用适当的日志级别
- 包含足够的上下文信息

### 5. 性能优化

- 使用缓存避免重复计算
- 选择合适的推理后端
- 优化数据加载流程