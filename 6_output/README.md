# 输出层说明文档

## 6_output/ 目录结构

这个目录用于存储所有的输出结果，包括训练输出、推理结果、评估报告、日志等。

### 目录组织结构

```
6_output/
├── training/           # 训练输出
│   ├── logs/          # 训练日志
│   ├── checkpoints/   # 检查点文件
│   ├── metrics/       # 训练指标
│   └── configs/       # 使用的配置文件
├── inference/         # 推理输出
│   ├── single/        # 单条推理结果
│   ├── batch/         # 批量推理结果
│   ├── interactive/   # 交互式对话记录
│   └── api_logs/      # API调用日志
├── evaluation/        # 评估输出
│   ├── reports/       # 评估报告
│   ├── metrics/       # 评估指标
│   ├── comparisons/   # 模型对比结果
│   └── benchmarks/    # 基准测试结果
├── deployment/        # 部署输出
│   ├── logs/          # 部署日志
│   ├── configs/       # 部署配置
│   └── monitoring/    # 监控数据
├── agents/            # 智能体输出
│   ├── conversations/ # 对话记录
│   ├── preferences/   # 偏好学习数据
│   ├── knowledge/     # 知识库更新
│   └── analytics/     # 分析报告
└── exports/           # 导出数据
    ├── datasets/      # 生成的数据集
    ├── models/        # 导出的模型文件
    └── reports/       # 汇总报告
```

### 文件命名规范

#### 时间戳格式
- 格式：`YYYYMMDD-HHMMSS`
- 示例：`20241014-143000`

#### 训练输出
- 日志：`training_log_{timestamp}.log`
- 指标：`training_metrics_{timestamp}.json`
- 配置：`training_config_{timestamp}.json`

#### 推理输出
- 单条：`inference_single_{timestamp}.json`
- 批量：`inference_batch_{timestamp}.json`
- 对话：`conversation_{session_id}_{timestamp}.json`

#### 评估输出
- 报告：`evaluation_report_{model_name}_{timestamp}.json`
- 对比：`comparison_{model_a}_vs_{model_b}_{timestamp}.json`
- 基准：`benchmark_{dataset}_{timestamp}.json`

### 输出文件格式

#### 训练日志格式
```json
{
  "training_id": "train_20241014_143000",
  "model_name": "Qwen2.5-7B-medical-lora",
  "start_time": "2024-10-14T14:30:00Z",
  "end_time": "2024-10-14T16:45:00Z",
  "config": {
    "train_type": "lora",
    "epochs": 3,
    "learning_rate": 1e-4,
    "batch_size": 8
  },
  "metrics": {
    "final_loss": 0.45,
    "best_epoch": 2,
    "total_steps": 1500
  },
  "checkpoints": [
    "checkpoint-500",
    "checkpoint-1000", 
    "checkpoint-final"
  ],
  "status": "completed"
}
```

#### 推理结果格式
```json
{
  "inference_id": "infer_20241014_143000",
  "model_checkpoint": "./models/checkpoint-1000",
  "timestamp": "2024-10-14T14:30:00Z",
  "config": {
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9
  },
  "results": [
    {
      "input": "什么是人工智能？",
      "output": "人工智能是计算机科学的一个分支...",
      "metrics": {
        "response_time": 2.3,
        "token_count": 128
      }
    }
  ]
}
```

#### 评估报告格式
```json
{
  "evaluation_id": "eval_20241014_143000",
  "model_checkpoint": "./models/checkpoint-1000",
  "dataset": "medical_qa_test",
  "timestamp": "2024-10-14T14:30:00Z",
  "metrics": {
    "bleu": 0.75,
    "rouge-1": 0.68,
    "rouge-2": 0.45,
    "rouge-l": 0.62,
    "f1": 0.72,
    "exact_match": 0.45
  },
  "samples": 1000,
  "detailed_results": "evaluation_details_20241014_143000.json"
}
```

### 日志管理

#### 日志级别
- **DEBUG**：详细的调试信息
- **INFO**：一般信息记录
- **WARNING**：警告信息
- **ERROR**：错误信息
- **CRITICAL**：严重错误

#### 日志格式
```
[2024-10-14 14:30:00] [INFO] [TrainingManager] 开始LoRA训练...
[2024-10-14 14:30:05] [DEBUG] [DataLoader] 加载数据集: medical_qa
[2024-10-14 14:30:10] [WARNING] [GPU] GPU内存使用率超过80%
[2024-10-14 14:35:00] [ERROR] [Training] 训练过程中出现错误: CUDA out of memory
```

### 数据保留策略

#### 短期存储（1-7天）
- 调试日志
- 临时推理结果
- 测试输出

#### 中期存储（1个月）
- 训练日志
- 评估报告
- 对话记录

#### 长期存储（永久）
- 重要的模型输出
- 基准测试结果
- 最终评估报告

### 自动化清理

#### 清理规则
```bash
# 清理7天前的调试日志
find 6_output/*/logs/ -name "*debug*.log" -mtime +7 -delete

# 清理30天前的临时文件
find 6_output/temp/ -type f -mtime +30 -delete

# 压缩3个月前的日志文件
find 6_output/*/logs/ -name "*.log" -mtime +90 -exec gzip {} \;
```

### 监控和报警

#### 磁盘空间监控
- 当6_output目录占用超过80%时发出警告
- 当剩余空间小于10GB时发出紧急警告

#### 文件数量监控
- 监控各子目录的文件数量
- 防止产生过多小文件影响性能

### 使用工具

#### 输出管理脚本
- `output_manager.py`：输出文件管理
- `log_analyzer.py`：日志分析工具
- `report_generator.py`：报告生成器

#### 使用示例

```bash
# 查看最近的训练输出
python output_manager.py list --type training --limit 10

# 分析训练日志
python log_analyzer.py --log_file training_log_20241014.log

# 生成评估报告
python report_generator.py --eval_results evaluation/ --output final_report.html

# 清理过期文件
python output_manager.py cleanup --older_than 30d
```

### 注意事项

1. **存储配额**：定期检查磁盘使用情况
2. **备份策略**：重要输出定期备份
3. **权限管理**：确保输出文件有正确的权限
4. **格式统一**：保持输出格式的一致性
5. **索引管理**：为大量输出文件建立索引便于查找