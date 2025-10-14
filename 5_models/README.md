# 模型层说明文档

## 5_models/ 目录结构

这个目录用于存储和管理所有的模型文件，包括预训练模型、微调后的模型等。

### 目录组织结构

```
5_models/
├── pretrained/          # 预训练模型
│   ├── qwen/           # Qwen系列模型
│   ├── llama/          # LLaMA系列模型
│   └── custom/         # 自定义模型
├── finetuned/          # 微调后的模型
│   ├── lora/           # LoRA微调模型
│   ├── qlora/          # QLoRA微调模型
│   ├── full/           # 全参数微调模型
│   └── multimodal/     # 多模态微调模型
├── checkpoints/        # 训练检查点
│   ├── training_logs/  # 训练日志
│   └── intermediate/   # 中间检查点
└── exports/            # 导出的模型
    ├── onnx/          # ONNX格式
    ├── tensorrt/      # TensorRT格式
    └── safetensors/   # SafeTensors格式
```

### 模型命名规范

#### 预训练模型
- 格式：`{model_family}-{size}-{variant}`
- 示例：`Qwen2.5-7B-Instruct`、`LLaMA-13B-Chat`

#### 微调模型
- 格式：`{base_model}-{task}-{train_type}-{timestamp}`
- 示例：`Qwen2.5-7B-medical-lora-20241014`

#### 检查点
- 格式：`checkpoint-{step}`
- 示例：`checkpoint-1000`、`checkpoint-final`

### 模型版本管理

#### 版本号规范
- 主版本.次版本.修订版本
- 示例：`v1.0.0`、`v1.1.0`、`v1.0.1`

#### 模型元数据
每个模型目录应包含以下文件：
- `model_info.json`：模型基本信息
- `training_config.json`：训练配置
- `performance_metrics.json`：性能指标
- `README.md`：模型说明文档

### 模型信息示例

```json
{
  "model_name": "Qwen2.5-7B-medical-lora",
  "base_model": "Qwen/Qwen2.5-7B-Instruct",
  "model_type": "lora",
  "task": "medical_qa",
  "version": "v1.0.0",
  "created_time": "2024-10-14T10:00:00Z",
  "training_dataset": "medical_qa_dataset",
  "training_steps": 1000,
  "performance": {
    "bleu": 0.75,
    "rouge-l": 0.68,
    "f1": 0.72
  },
  "file_size": "15.2GB",
  "checksum": "sha256:abc123...",
  "tags": ["medical", "qa", "chinese"],
  "description": "基于Qwen2.5-7B的医疗问答LoRA微调模型"
}
```

### 存储最佳实践

#### 1. 空间管理
- 定期清理不需要的中间检查点
- 使用符号链接避免重复存储
- 压缩长期存储的模型

#### 2. 备份策略
- 重要模型定期备份到云存储
- 使用增量备份减少存储开销
- 记录备份和恢复日志

#### 3. 访问控制
- 设置适当的文件权限
- 对敏感模型进行加密存储
- 记录模型访问日志

### 工具支持

#### 模型管理脚本
- `model_registry.py`：模型注册和查询
- `model_converter.py`：模型格式转换
- `model_validator.py`：模型完整性验证

#### 使用示例

```bash
# 列出所有模型
python model_registry.py list

# 注册新模型
python model_registry.py register --path ./finetuned/lora/my_model

# 转换模型格式
python model_converter.py --input ./model.bin --output ./model.onnx --format onnx

# 验证模型完整性
python model_validator.py --model_path ./model
```

### 注意事项

1. **存储路径**：确保有足够的磁盘空间
2. **权限设置**：正确设置文件和目录权限
3. **版本控制**：不要将大型模型文件提交到Git
4. **文档更新**：及时更新模型文档和元数据
5. **清理策略**：定期清理不再需要的模型文件