# 架构设计文档

## 设计原则

本项目采用分层架构设计，遵循以下核心原则：

### 1. 分离关注点 (Separation of Concerns)
每一层都有明确的职责边界，避免功能耦合：
- **配置层**：专注于配置管理
- **工具层**：提供通用工具函数
- **核心层**：实现业务逻辑
- **智能体层**：专业AI功能
- **脚本层**：用户交互接口
- **模型层**：模型资源管理
- **输出层**：结果数据管理
- **文档层**：知识和说明

### 2. 单一职责原则 (Single Responsibility Principle)
每个模块只负责一个明确的功能领域，便于维护和扩展。

### 3. 依赖倒置原则 (Dependency Inversion Principle)
高层模块不依赖低层模块，都依赖于抽象接口。

### 4. 开闭原则 (Open/Closed Principle)
对扩展开放，对修改封闭，支持插件化扩展。

## 层次结构详解

### 第0层：配置层 (0_configs/)
**职责**：统一的配置管理和参数控制

```
0_configs/
├── 0_train_config.json      # 训练配置
├── 1_model_config.json      # 模型配置
├── 2_deploy_config.json     # 部署配置
└── agents/                  # 智能体配置
    ├── rag_config.json
    ├── preference_config.json
    └── medical_config.json
```

**设计特点**：
- 集中式配置管理
- 分类配置文件
- 支持环境变量覆盖
- 配置验证和默认值

### 第1层：工具层 (1_utils/)
**职责**：提供通用的工具函数和基础设施

```
1_utils/
├── 0_config_loader.py       # 配置加载器
├── 1_logger.py             # 日志系统
├── 2_gpu_manager.py        # GPU管理
├── 3_io_utils.py           # 文件I/O工具
├── 4_metrics.py            # 评估指标
└── 5_decorators.py         # 装饰器库
```

**设计特点**：
- 无状态函数设计
- 高度可复用
- 完整的错误处理
- 性能优化

### 第2层：核心层 (2_core/)
**职责**：实现核心业务逻辑和算法

```
2_core/
├── __init__.py             # 核心API接口
├── training/               # 训练模块
│   └── trainer.py
├── inference/              # 推理模块
│   └── inference.py
├── deployment/             # 部署模块
│   └── deploy.py
└── evaluation/             # 评估模块
    └── evaluator.py
```

**设计特点**：
- 模块化设计
- 统一的API接口
- 可插拔的组件
- 完整的生命周期管理

### 第3层：智能体层 (3_agents/)
**职责**：实现专业的AI智能体功能

```
3_agents/
├── base_agent.py           # 基础智能体框架
├── rag_agent.py           # RAG智能体
├── preference_agent.py     # 偏好学习智能体
└── medical_agent.py       # 医疗智能体
```

**设计特点**：
- 基于继承的扩展机制
- 状态管理
- 事件驱动架构
- 插件化能力

### 第4层：脚本层 (4_scripts/)
**职责**：提供命令行接口和用户交互

```
4_scripts/
├── main.py                 # 主入口脚本
├── train.py               # 训练脚本
├── infer.py               # 推理脚本
├── deploy.py              # 部署脚本
└── evaluate.py            # 评估脚本
```

**设计特点**：
- 统一的命令行接口
- 参数验证
- 错误处理
- 用户友好的提示

### 第5层：模型层 (5_models/)
**职责**：模型文件的存储和管理

```
5_models/
├── pretrained/             # 预训练模型
├── finetuned/             # 微调模型
├── checkpoints/           # 训练检查点
└── exports/               # 导出模型
```

**设计特点**：
- 分类存储
- 版本管理
- 元数据记录
- 自动清理

### 第6层：输出层 (6_output/)
**职责**：管理所有的输出结果和日志

```
6_output/
├── training/              # 训练输出
├── inference/             # 推理输出
├── evaluation/            # 评估输出
├── deployment/            # 部署输出
├── agents/               # 智能体输出
└── exports/              # 导出数据
```

**设计特点**：
- 结构化存储
- 时间戳命名
- 自动归档
- 空间管理

### 第7层：文档层 (7_docs/)
**职责**：项目文档和知识管理

```
7_docs/
├── README.md              # 项目说明
├── architecture.md        # 架构文档
├── api_reference.md       # API参考
├── tutorials/             # 教程
└── examples/              # 示例代码
```

**设计特点**：
- 完整的文档体系
- 多层次说明
- 示例驱动
- 持续更新

## 数据流设计

### 训练流程
```
用户输入 → 4_scripts/train.py → 2_core/training/ → 1_utils/ → 6_output/training/
   ↑                                ↓
0_configs/ ←→ 配置加载 ←→ 模型保存 → 5_models/
```

### 推理流程
```
用户输入 → 4_scripts/infer.py → 2_core/inference/ → 5_models/ → 结果输出 → 6_output/inference/
                                     ↓
                               1_utils/logger → 日志记录
```

### 智能体流程
```
用户交互 → 3_agents/base_agent → 2_core/inference/ → 响应生成
    ↓                ↓                    ↓
   会话管理 ←→ 偏好学习 ←→ 知识检索 → 6_output/agents/
```

## 错误处理策略

### 分层错误处理
1. **工具层**：基础异常处理和日志记录
2. **核心层**：业务逻辑异常处理和恢复
3. **智能体层**：状态恢复和优雅降级
4. **脚本层**：用户友好的错误信息

### 错误传播机制
```python
try:
    # 业务逻辑
    result = core_function()
except CoreException as e:
    # 记录详细错误信息
    logger.error(f"Core error: {e}")
    # 转换为用户友好的错误
    raise UserFriendlyException("操作失败，请检查配置")
```

## 扩展机制

### 插件架构
每一层都支持插件式扩展：

```python
# 注册新的训练类型
@register_trainer("custom_lora")
class CustomLoRATrainer(BaseTrainer):
    def train(self, config):
        # 自定义训练逻辑
        pass

# 注册新的智能体
@register_agent("custom_agent")
class CustomAgent(BaseAgent):
    def process_input(self, input_text):
        # 自定义处理逻辑
        pass
```

### 配置驱动扩展
通过配置文件控制功能启用：

```json
{
  "enabled_features": ["lora", "qlora", "custom_lora"],
  "agent_types": ["rag", "preference", "custom"],
  "extensions": {
    "custom_lora": {
      "module": "extensions.custom_lora",
      "config": {...}
    }
  }
}
```

## 性能考虑

### 缓存策略
- **配置缓存**：避免重复加载配置文件
- **模型缓存**：智能模型加载和卸载
- **结果缓存**：缓存计算结果避免重复计算

### 内存管理
- **懒加载**：按需加载大型资源
- **资源池**：复用昂贵的资源对象
- **垃圾回收**：主动释放不需要的资源

### 并发处理
- **异步I/O**：非阻塞的文件和网络操作
- **线程池**：管理并发任务
- **进程隔离**：避免单点故障

## 安全设计

### 访问控制
- **配置文件权限**：限制敏感配置的访问
- **模型文件保护**：防止模型被恶意修改
- **日志脱敏**：避免敏感信息泄露

### 输入验证
- **参数校验**：严格的输入参数验证
- **类型检查**：运行时类型检查
- **边界检查**：防止缓冲区溢出

### 审计跟踪
- **操作日志**：记录所有关键操作
- **访问日志**：跟踪资源访问
- **错误日志**：记录异常和错误

## 测试策略

### 单元测试
每个工具函数都有对应的单元测试：

```python
def test_config_loader():
    loader = ConfigLoader()
    config = loader.load_config("test_config.json")
    assert config["model"] == "test_model"
```

### 集成测试
测试各层之间的集成：

```python
def test_training_pipeline():
    # 测试完整的训练流程
    trainer = TrainingManager("test_config.json")
    result = trainer.train_lora(test_params)
    assert result.success
```

### 端到端测试
测试完整的用户场景：

```bash
# 测试训练到部署的完整流程
python test_e2e_training.py
python test_e2e_inference.py
python test_e2e_deployment.py
```

## 部署架构

### 开发环境
```
Developer Machine
├── 代码编辑器 (VS Code)
├── Python环境 (venv/conda)
├── GPU开发卡
└── 本地存储
```

### 测试环境
```
Test Server
├── CI/CD Pipeline
├── 自动化测试
├── 性能基准测试
└── 安全扫描
```

### 生产环境
```
Production Cluster
├── 负载均衡器
├── 应用服务器集群
├── GPU计算节点
├── 分布式存储
└── 监控系统
```

## 监控和运维

### 关键指标
- **训练指标**：loss、准确率、训练时间
- **推理指标**：延迟、吞吐量、成功率
- **系统指标**：CPU、内存、GPU使用率
- **业务指标**：用户满意度、模型质量

### 告警规则
- **资源告警**：GPU内存使用超过90%
- **性能告警**：推理延迟超过阈值
- **错误告警**：错误率超过5%
- **业务告警**：模型质量下降

### 日志聚合
```
Application Logs → Log Collector → Log Storage → Analysis Dashboard
                                      ↓
                                  Alert System
```

## 未来演进

### 架构演进方向
1. **微服务化**：将各层拆分为独立的微服务
2. **云原生**：支持Kubernetes部署
3. **边缘计算**：支持边缘设备部署
4. **联邦学习**：支持分布式训练

### 技术栈演进
1. **容器化**：Docker和Kubernetes
2. **服务网格**：Istio服务治理
3. **流式处理**：支持实时数据流
4. **GraphQL API**：更灵活的API设计

这个架构设计为项目提供了坚实的技术基础，支持快速开发、灵活扩展和稳定运行。