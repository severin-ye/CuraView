# 故障排除指南

## 常见问题解决方案

### 1. 环境问题

#### Python版本不兼容

**问题描述：**
```
TypeError: 'type' object is not subscriptable
```

**解决方案：**
```bash
# 检查Python版本
python --version

# 如果版本低于3.8，请升级
conda install python=3.9
# 或者
pyenv install 3.9.0
pyenv global 3.9.0
```

#### 依赖包冲突

**问题描述：**
```
ImportError: cannot import name 'xxx' from 'yyy'
```

**解决方案：**
```bash
# 重新安装依赖
pip uninstall torch transformers
pip install torch transformers --upgrade

# 或者使用conda
conda update --all

# 检查包版本兼容性
python 4_scripts/tools.py --check_dependencies
```

#### CUDA版本不匹配

**问题描述：**
```
RuntimeError: CUDA out of memory
AssertionError: Torch not compiled with CUDA support
```

**解决方案：**
```bash
# 检查CUDA版本
nvcc --version
nvidia-smi

# 安装匹配的PyTorch版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 验证CUDA可用性
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. 训练问题

#### 内存不足错误

**问题描述：**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**解决方案：**

1. 减少批次大小：
```bash
python 4_scripts/train.py \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16
```

2. 启用梯度检查点：
```bash
python 4_scripts/train.py \
    --gradient_checkpointing true
```

3. 使用混合精度训练：
```bash
python 4_scripts/train.py \
    --fp16 true
    # 或者
    --bf16 true
```

4. 清理GPU缓存：
```python
import torch
torch.cuda.empty_cache()
```

#### 训练不收敛

**问题描述：**
- 损失不下降
- 验证指标没有改善
- 梯度消失或爆炸

**解决方案：**

1. 调整学习率：
```bash
# 降低学习率
python 4_scripts/train.py --learning_rate 1e-5

# 增加warmup
python 4_scripts/train.py --warmup_ratio 0.1
```

2. 检查数据质量：
```bash
python 4_scripts/tools.py --validate_dataset your_dataset.json
```

3. 调整训练策略：
```bash
# 增加训练轮数
python 4_scripts/train.py --num_train_epochs 10

# 降低权重衰减
python 4_scripts/train.py --weight_decay 0.01
```

#### 数据加载错误

**问题描述：**
```
JSONDecodeError: Expecting value: line 1 column 1 (char 0)
FileNotFoundError: Dataset file not found
```

**解决方案：**

1. 验证数据格式：
```bash
python 4_scripts/tools.py --validate_dataset your_dataset.json
```

2. 检查文件路径：
```bash
# 使用绝对路径
python 4_scripts/train.py --dataset /absolute/path/to/dataset.json
```

3. 数据格式示例：
```json
[
  {
    "system": "你是一个helpful assistant",
    "input": "用户输入",
    "output": "期望输出"
  }
]
```

### 3. 推理问题

#### 模型加载失败

**问题描述：**
```
OSError: Can't load tokenizer for 'your_model_path'
RuntimeError: Error(s) in loading state_dict
```

**解决方案：**

1. 检查模型路径：
```bash
python 4_scripts/tools.py --validate_checkpoint your_checkpoint
```

2. 验证模型完整性：
```bash
ls -la your_checkpoint/
# 应该包含：config.json, pytorch_model.bin, tokenizer.json等
```

3. 重新下载基础模型：
```bash
python 4_scripts/tools.py --download_model Qwen/Qwen2.5-7B-Instruct
```

#### 推理速度过慢

**问题描述：**
- 单次推理时间过长
- 批量推理效率低

**解决方案：**

1. 使用更高效的推理后端：
```bash
python 4_scripts/infer.py \
    --infer_backend vllm \
    --checkpoint your_model
```

2. 调整生成参数：
```bash
python 4_scripts/infer.py \
    --max_tokens 512 \
    --temperature 0.7
```

3. 使用量化模型：
```bash
python 4_scripts/infer.py \
    --quantization int8 \
    --checkpoint your_model
```

#### 输出质量差

**问题描述：**
- 生成内容不相关
- 重复输出
- 格式错误

**解决方案：**

1. 调整生成参数：
```bash
python 4_scripts/infer.py \
    --temperature 0.5 \
    --top_p 0.8 \
    --repetition_penalty 1.1
```

2. 优化提示词：
```python
# 使用更清晰的系统提示
system_prompt = "你是一个专业的助手，请根据用户问题提供准确、简洁的回答。"
```

3. 检查训练数据质量：
```bash
python 4_scripts/tools.py --analyze_dataset your_training_data.json
```

### 4. 部署问题

#### 服务启动失败

**问题描述：**
```
Address already in use
Permission denied
ModuleNotFoundError: No module named 'xxx'
```

**解决方案：**

1. 检查端口占用：
```bash
lsof -i :8000
# 杀死占用进程
kill -9 <PID>
```

2. 更换端口：
```bash
python 4_scripts/deploy.py \
    --port 8001 \
    --checkpoint your_model
```

3. 检查权限：
```bash
# 使用非特权端口（>1024）
python 4_scripts/deploy.py --port 8000

# 或者使用sudo（不推荐）
sudo python 4_scripts/deploy.py --port 80
```

#### API请求失败

**问题描述：**
```
ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))
```

**解决方案：**

1. 检查服务状态：
```bash
curl http://localhost:8000/health
```

2. 验证请求格式：
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 100
    }'
```

3. 检查服务日志：
```bash
python 4_scripts/deploy.py --debug --checkpoint your_model
```

### 5. 智能体问题

#### RAG智能体知识库问题

**问题描述：**
- 检索结果不相关
- 知识库加载失败

**解决方案：**

1. 验证知识库格式：
```json
[
  {
    "id": "doc_1",
    "title": "文档标题",
    "content": "文档内容",
    "source": "来源"
  }
]
```

2. 重建索引：
```python
from agents.rag_agent import RAGAgent

agent = RAGAgent("RAG助手")
agent.rebuild_index("knowledge_base.json")
```

3. 调整检索参数：
```python
agent.set_retrieval_params(
    top_k=10,
    similarity_threshold=0.3
)
```

#### 偏好学习异常

**问题描述：**
- 偏好学习不生效
- 用户偏好混乱

**解决方案：**

1. 重置用户偏好：
```python
from agents.preference_agent import PreferenceAgent

agent = PreferenceAgent("偏好助手")
agent.reset_user_preferences("user_id")
```

2. 验证反馈格式：
```python
# 正确的反馈格式
feedback = {
    "rating": 5,  # 1-5评分
    "comment": "很好的回答",
    "preferred_style": "简洁"
}
```

### 6. 系统性能问题

#### 系统资源不足

**问题描述：**
- CPU使用率过高
- 内存不足
- 磁盘空间不够

**解决方案：**

1. 监控系统资源：
```bash
python 4_scripts/main.py info --system
```

2. 清理临时文件：
```bash
python 4_scripts/tools.py --cleanup
```

3. 优化并发数：
```bash
python 4_scripts/deploy.py \
    --max_workers 4 \
    --checkpoint your_model
```

#### 网络连接问题

**问题描述：**
- 模型下载失败
- API请求超时

**解决方案：**

1. 设置代理：
```bash
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
```

2. 增加超时时间：
```bash
python 4_scripts/train.py \
    --download_timeout 3600 \
    --model your_model
```

3. 使用本地模型：
```bash
# 下载到本地后使用本地路径
python 4_scripts/train.py --model /local/path/to/model
```

## 调试技巧

### 1. 日志分析

#### 启用详细日志

```bash
# 设置日志级别
export LOG_LEVEL=DEBUG

# 或者在代码中设置
python 4_scripts/train.py --log_level debug
```

#### 查看训练日志

```bash
# 实时查看日志
tail -f 6_output/*/logs/training.log

# 搜索错误信息
grep -i error 6_output/*/logs/training.log
```

### 2. 渐进式调试

#### 最小化测试

```bash
# 使用小数据集测试
python 4_scripts/train.py \
    --dataset small_test.json \
    --num_train_epochs 1 \
    --max_steps 10
```

#### 分步验证

```bash
# 1. 验证数据加载
python 4_scripts/tools.py --test_dataloader your_dataset.json

# 2. 验证模型加载
python 4_scripts/tools.py --test_model_loading your_model

# 3. 验证训练步骤
python 4_scripts/tools.py --test_training_step
```

### 3. 性能分析

#### 内存使用分析

```bash
python 4_scripts/tools.py --memory_profile --function train_lora
```

#### GPU利用率监控

```bash
# 实时监控
nvidia-smi -l 1

# 记录GPU使用情况
python 4_scripts/tools.py --gpu_monitor --duration 3600
```

## 自动化故障诊断

### 系统健康检查

创建健康检查脚本：

```python
# health_check.py
import subprocess
import sys

def check_environment():
    """检查环境配置"""
    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
        print(f"✓ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ GPU数量: {torch.cuda.device_count()}")
    except ImportError:
        print("✗ PyTorch未安装")
        return False
    return True

def check_models():
    """检查模型文件"""
    model_paths = ["5_models/", "6_output/"]
    for path in model_paths:
        if os.path.exists(path):
            print(f"✓ 目录存在: {path}")
        else:
            print(f"✗ 目录不存在: {path}")

def check_services():
    """检查服务状态"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✓ API服务正常")
        else:
            print("✗ API服务异常")
    except:
        print("✗ API服务未启动")

if __name__ == "__main__":
    check_environment()
    check_models()
    check_services()
```

### 自动修复脚本

```bash
#!/bin/bash
# auto_fix.sh

# 清理GPU缓存
python -c "import torch; torch.cuda.empty_cache()"

# 重启服务
pkill -f "python 4_scripts/deploy.py"
sleep 5
python 4_scripts/deploy.py --checkpoint default_model &

# 检查磁盘空间
df -h | grep -E '9[0-9]%|100%' && echo "警告：磁盘空间不足"

# 清理临时文件
find /tmp -name "*.tmp" -delete
find 6_output -name "*.log" -mtime +7 -delete
```

## 获取帮助

### 1. 内置帮助

```bash
# 查看命令帮助
python 4_scripts/main.py --help
python 4_scripts/train.py --help

# 查看系统信息
python 4_scripts/main.py info --all
```

### 2. 日志收集

```bash
# 收集诊断信息
python 4_scripts/tools.py --collect_logs --output diagnostic_info.zip
```

### 3. 社区支持

- 查看项目文档：`7_docs/README.md`
- 提交问题时请包含：
  - 错误信息的完整日志
  - 系统环境信息
  - 重现步骤
  - 使用的配置文件

### 4. 常用工具命令

```bash
# 验证安装
python 4_scripts/tools.py --validate_installation

# 性能基准测试
python 4_scripts/tools.py --benchmark

# 清理环境
python 4_scripts/tools.py --cleanup --force

# 重置配置
python 4_scripts/tools.py --reset_config
```

记住：大多数问题都可以通过仔细检查错误信息、验证配置参数和逐步调试来解决。如果遇到持续性问题，请收集完整的日志信息并寻求社区帮助。