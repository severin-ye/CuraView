# MS-Swift 微调工具套件 - 7层架构专业版

## 项目概述

这是一个基于MS-Swift框架的专业级大语言模型微调工具套件，采用7层架构设计，提供完整的训练、推理、部署、评估和智能体开发能力。

## 🏗️ 架构设计

本项目采用分层架构设计，每一层都有明确的职责和边界：

```
📁 项目根目录/
├── 0_configs/          # 📋 配置层：统一配置管理
├── 1_utils/            # 🔧 工具层：通用工具函数
├── 2_core/             # ⚙️  核心层：业务逻辑实现
├── 3_agents/           # 🤖 智能体层：专业智能体
├── 4_scripts/          # 💻 脚本层：命令行入口
├── 5_models/           # 📦 模型层：模型存储管理
├── 6_output/           # 📊 输出层：结果输出管理
└── 7_docs/             # 📚 文档层：文档和说明
```

## ✨ 核心功能

### 🎯 训练功能
- **LoRA微调**：高效的低秩适应微调
- **QLoRA微调**：量化LoRA，节省显存
- **全参数微调**：完整的模型参数微调
- **多模态微调**：支持视觉-语言模型微调

### 🤖 推理功能
- **单条推理**：快速单问题回答
- **批量推理**：高效批量处理
- **交互式对话**：实时对话体验
- **API服务**：RESTful API接口

### 🚀 部署功能
- **本地部署**：单机部署服务
- **分布式部署**：多GPU并行部署
- **云端部署**：支持各种推理后端
- **多模型部署**：同时部署多个LoRA模型

### 📊 评估功能
- **自动评估**：多种评估指标
- **基准测试**：标准数据集测试
- **对比分析**：模型性能对比
- **报告生成**：详细评估报告

### 🎯 智能体功能
- **RAG智能体**：检索增强生成
- **偏好学习智能体**：基于反馈学习
- **医疗智能体**：专业医疗问答
- **自定义智能体**：灵活扩展能力

## 🚀 快速开始

### 环境要求
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+
- 16GB+ GPU内存（推荐）

### 安装依赖
```bash
# 安装MS-Swift
pip install ms-swift

# 安装项目依赖
pip install -r requirements.txt

# 检查环境
python 4_scripts/main.py tools --check_env
```

### 基础使用

#### 1. 训练模型
```bash
# 使用LoRA预设训练
python 4_scripts/main.py train --preset lora --model Qwen/Qwen2.5-7B-Instruct

# 自定义训练参数
python 4_scripts/main.py train \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset medical_qa_dataset \
    --train_type lora \
    --num_train_epochs 3 \
    --output_dir ./6_output/my_model
```

#### 2. 推理对话
```bash
# 交互式对话
python 4_scripts/main.py infer \
    --checkpoint ./6_output/my_model \
    --mode interactive

# 单条推理
python 4_scripts/main.py infer \
    --checkpoint ./6_output/my_model \
    --mode single \
    --question "什么是人工智能？"
```

#### 3. 部署服务
```bash
# 本地部署
python 4_scripts/main.py deploy \
    --checkpoint ./6_output/my_model \
    --preset local

# 服务器部署
python 4_scripts/main.py deploy \
    --checkpoint ./6_output/my_model \
    --preset server \
    --port 8000
```

#### 4. 评估模型
```bash
# 标准评估
python 4_scripts/main.py evaluate \
    --checkpoint ./6_output/my_model \
    --preset standard \
    --test_data ./test_data.json

# 基准测试
python 4_scripts/main.py evaluate \
    --checkpoint ./6_output/my_model \
    --benchmark
```

## 📋 配置管理

### 配置文件结构
```
0_configs/
├── 0_train_config.json     # 训练配置
├── 1_model_config.json     # 模型配置
├── 2_deploy_config.json    # 部署配置
└── agents/                 # 智能体配置
    ├── rag_config.json
    ├── preference_config.json
    └── medical_config.json
```

### 配置使用方法
```bash
# 使用配置文件
python 4_scripts/train.py --config 0_configs/0_train_config.json

# 查看配置信息
python 4_scripts/main.py info --config
```

## 🛠️ 工具层说明

### 通用工具
- **配置加载器**：统一的配置管理
- **日志系统**：结构化日志记录
- **GPU管理器**：GPU资源管理
- **文件I/O工具**：高效的文件操作
- **评估指标**：丰富的评估指标
- **装饰器库**：常用的功能装饰器

### 工具使用示例
```python
from logger import Logger
from config_loader import ConfigLoader
from gpu_manager import GPUManager

# 创建日志器
logger = Logger("MyApp").get_logger()

# 加载配置
config_loader = ConfigLoader()
config = config_loader.load_config("config.json")

# 管理GPU
gpu_manager = GPUManager()
gpu_info = gpu_manager.get_gpu_info()
```

## 🏢 企业级特性

### 高可用性
- **容错机制**：自动错误恢复
- **监控告警**：实时状态监控
- **负载均衡**：智能负载分配
- **备份恢复**：完整的备份策略

### 安全性
- **访问控制**：基于角色的权限管理
- **数据加密**：敏感数据加密存储
- **审计日志**：完整的操作审计
- **合规性**：符合数据保护规范

### 扩展性
- **模块化设计**：松耦合的模块设计
- **插件系统**：灵活的功能扩展
- **API接口**：完整的REST API
- **多语言支持**：支持多种编程语言

## 📊 性能优化

### 内存优化
- **梯度检查点**：减少内存占用
- **混合精度训练**：提升训练效率
- **模型并行**：大模型分布式训练
- **动态批处理**：自适应批次大小

### 计算优化
- **算子融合**：减少计算开销
- **缓存机制**：智能结果缓存
- **异步处理**：提升并发性能
- **硬件加速**：充分利用GPU资源

## 🔧 开发指南

### 代码规范
- **PEP8标准**：Python代码规范
- **类型注解**：完整的类型标注
- **文档字符串**：详细的API文档
- **单元测试**：完整的测试覆盖

### 贡献指南
1. Fork本项目
2. 创建特性分支
3. 提交代码变更
4. 创建Pull Request
5. 代码审查

### 架构扩展
- **新增智能体**：继承BaseAgent类
- **新增工具**：添加到1_utils目录
- **新增评估指标**：扩展metrics模块
- **新增部署后端**：扩展deployment模块

## 🐛 故障排除

### 常见问题
1. **GPU内存不足**
   - 减小batch_size
   - 启用梯度检查点
   - 使用QLoRA训练

2. **训练速度慢**
   - 检查数据加载效率
   - 优化模型并行策略
   - 使用混合精度训练

3. **推理延迟高**
   - 使用量化模型
   - 启用KV缓存
   - 选择合适的推理后端

### 调试工具
```bash
# 检查环境
python 4_scripts/main.py tools --check_env

# 验证配置
python 4_scripts/train.py --dry_run

# 查看日志
tail -f 6_output/training/logs/training.log
```

## 📚 参考文档

- [MS-Swift官方文档](https://github.com/modelscope/swift)
- [PyTorch文档](https://pytorch.org/docs/)
- [Transformers文档](https://huggingface.co/docs/transformers/)
- [LoRA论文](https://arxiv.org/abs/2106.09685)
- [QLoRA论文](https://arxiv.org/abs/2305.14314)

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 🤝 社区支持

- **GitHub Issues**：问题反馈和功能请求
- **讨论区**：技术交流和经验分享
- **Wiki**：详细的使用教程
- **示例代码**：丰富的应用示例

## 🎯 路线图

### v1.0.0 (当前版本)
- ✅ 7层架构设计
- ✅ 基础训练功能
- ✅ 推理和部署
- ✅ 评估系统

### v1.1.0 (计划中)
- 🔄 智能体系统完善
- 🔄 API接口优化
- 🔄 性能监控
- 🔄 可视化界面

### v1.2.0 (未来)
- 📅 多模态支持增强
- 📅 分布式训练优化
- 📅 云原生部署
- 📅 企业级安全

---

**联系方式**
- 项目地址：[GitHub Repository]
- 技术支持：[support@example.com]
- 官方网站：[https://example.com]