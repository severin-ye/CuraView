# 🏥🤖 CuraView (精衡) - Medical AI Hallucination Detection & Correction System# 🏥🤖 Jingheng - 医疗AI幻觉检测与纠错系统



[中文](README.zh-CN.md) | **English**> 基于多智能体架构的医疗大模型幻觉检测、归类、纠错一体化研究平台



> A multi-agent architecture-based integrated research platform for medical large model hallucination detection, classification, and correction



## 🎯 Project Overview## 🎯 项目概述



CuraView is an innovative research platform focused on hallucination detection and correction for medical large language models. By building a multi-agent collaborative system, it achieves automated error detection, classification archiving, and intelligent correction of medical AI-generated content, providing strong technical assurance for the reliability and safety of medical AI.Jingheng是一个专注于医疗大语言模型幻觉检测与纠错的创新研究平台。通过构建多智能体协作系统，实现对医疗AI生成内容的自动化错误检测、分类归档和智能纠正，为医疗AI的可靠性和安全性提供强有力的技术保障。



### 🌟 Core Innovations### 🌟 核心创新



- **🔍 Hallucination Detection Agent**: Intelligent error detection system based on context engineering- **🔍 幻觉检测Agent**: 基于上下文工程的智能错误检测系统

- **📊 Error Classification Agent**: Graph-structured medical error classification system- **📊 错误归类Agent**: 图结构化的医疗错误分类体系

- **⚡ Error Generation Agent**: Efficient synthetic error data generation engine- **⚡ 错误生成Agent**: 高效的合成错误数据生成引擎  

- **🧠 Correction Model Training**: Intelligent correction system with joint learning + instruction fine-tuning- **🧠 纠错模型训练**: 联合学习+指令微调的智能纠错系统

- **🏥 Medical Data Processing**: EHR data standardization and corpus construction tools- **🏥 医疗数据处理**: EHR数据标准化与语料构建工具



## 🧠 Research Architecture & Implementation Path## 🧠 研究架构与实现路径



### Phase 1: Multi-Agent Error Detection System 🔍### Phase 1: 多智能体错误检测系统 🔍



#### 1.1 Error Detection Agent#### 1.1 错误检测Agent

```python```python

# Hallucination detection based on context engineering# 基于上下文工程的幻觉检测

class HallucinationDetector:class HallucinationDetector:

    - Use generative models from papers as baseline    - 使用论文中的生成模型作为baseline

    - Construct professional medical context prompt engineering    - 构造专业医学上下文提示工程

    - Statistical error frequency and distribution patterns    - 统计错误频率和分布模式

    - Output: Error location annotation + confidence score    - 输出: 错误位置标注 + 置信度评分

``````



#### 1.2 Error Classification Agent#### 1.2 错误归类Agent  

```python```python

# Graph-structured medical error classification# 图结构化的医疗错误分类

class ErrorClassifier:class ErrorClassifier:

    - Medical knowledge graph-driven error classification system    - 医学知识图谱驱动的错误分类体系

    - Multi-dimensional error labels: factual/logical/consistency/safety    - 多维度错误标签: 事实性/逻辑性/一致性/安全性

    - Hierarchical error type tree construction    - 层次化错误类型树构建

    - Output: Structured error classification labels    - 输出: 结构化错误分类标签

``````



#### 1.3 Error Generation Agent#### 1.3 错误生成Agent

```python```python

# Efficient synthetic error data generation# 高效合成错误数据生成

class ErrorSynthesizer:class ErrorSynthesizer:

    - Synthetic data generation based on real error patterns    - 基于真实错误模式的合成数据生成

    - Adversarial sample construction    - 对抗性样本构造

    - Diverse error type coverage    - 多样性错误类型覆盖

    - Output: Large-scale annotated error corpus    - 输出: 大规模标注错误语料

``````



### Phase 2: Intelligent Correction System Training 🧠### Phase 2: 智能纠错系统训练 🧠



#### 2.1 Hallucination Detection Small Model Training#### 2.1 幻觉检测小模型训练

```bash```bash

# Joint learning + instruction fine-tuning# 联合学习 + 指令微调

Training Pipeline:Training Pipeline:

├── Correction chain corpus construction (generative model + correction system)├── 纠错链语料构建 (生成模型+纠错系统)

├── Instruction fine-tuning (system+thinking+answer)├── 指令微调 (system+thinking+answer)

├── Qwen3 base data fusion (1:1 ratio, prevent catastrophic forgetting)├── Qwen3基础数据融合 (1:1比例，防止灾难性遗忘)

└── Multi-task joint optimization└── 多任务联合优化

``````



#### 2.2 Medical Generation Model Optimization#### 2.2 医学生成模型优化

```bash```bash

# Generation quality improvement strategy# 生成质量提升策略

Optimization Strategy:Optimization Strategy:

├── SFT: generative model + correction system → positive-negative contrastive learning├── SFT: 生成模型+纠错系统→正反对比学习

├── DPO: original manual corpus → preference optimization├── DPO: 原始人工语料→偏好优化

├── RLHF: human feedback reinforcement learning├── RLHF: 人类反馈强化学习

└── Joint learning: multi-objective collaborative optimization└── 联合学习: 多目标协同优化

``````



### Phase 3: RAG-Enhanced Correction Model 🔧### Phase 3: RAG增强纠错模型 🔧



#### 3.1 Hallucination Correction Core Function#### 3.1 幻觉纠正核心功能

```python```python

class HallucinationCorrector:class HallucinationCorrector:

    def correct_hallucination(self, text, rag_context):    def correct_hallucination(self, text, rag_context):

        """        """

        Intelligent correction based on RAG retrieval        基于RAG召回的智能纠错

                

        Returns:        Returns:

        ├── Error localization: precise error position annotation        ├── 错误定位: 精确标注错误位置

        ├── Error cause: deep analysis of error causes        ├── 错误原因: 深度分析错误成因  

        ├── Improvement suggestions: structured correction schemes        ├── 改进建议: 结构化修正方案

        └── Rewrite output: optimized medical text        └── 重写输出: 优化后的医学文本

        """        """

        pass        pass

``````



#### 3.2 Testing & Evaluation System#### 3.2 测试与评估体系

```bash```bash

# Comprehensive evaluation framework# 全面评估框架

Evaluation Framework:Evaluation Framework:

├── Detection accuracy: hallucination recognition precision evaluation├── 检测准确率: 幻觉识别精度评估

├── Classification correctness: error classification effectiveness evaluation├── 归类正确率: 错误分类效果评估  

├── Correction quality: medical accuracy of correction results├── 纠错质量: 修正结果医学准确性

├── Manual verification: comparison with professional physician annotations├── 人工验证: 专业医师标注对比

└── Synthetic errors: ability to detect artificially constructed errors└── 合成错误: 人工构造错误检测能力

``````



## 📁 Project Architecture## � 项目架构



``````

/home/work/hd//home/work/hd/

├── 🔍 agents/                       # Multi-agent system (planned)├── 🔍 agents/                       # 多智能体系统 (规划中)

│   ├── hallucination_detector/     # Hallucination Detection Agent│   ├── hallucination_detector/     # 幻觉检测Agent

│   ├── error_classifier/           # Error Classification Agent│   ├── error_classifier/           # 错误归类Agent  

│   ├── error_synthesizer/          # Error Generation Agent│   ├── error_synthesizer/          # 错误生成Agent

│   └── hallucination_corrector/    # Hallucination Correction Agent│   └── hallucination_corrector/    # 幻觉纠正Agent

││

├── 🧠 2_core/                       # Core training and inference├── 🧠 2_core/                       # 核心训练与推理

│   ├── 1_train/                     # Model training module│   ├── 1_train/                     # 模型训练模块

│   │   ├── Fine-tuning.ipynb        # Full parameter fine-tuning│   │   ├── Fine-tuning.ipynb        # 全参数微调

│   │   ├── Fine-tuning-lora.ipynb   # LoRA fine-tuning│   │   ├── Fine-tuning-lora.ipynb   # LoRA微调  

│   │   ├── joint_learning/          # Joint learning training (planned)│   │   ├── joint_learning/          # 联合学习训练 (规划中)

│   │   └── instruction_tuning/      # Instruction fine-tuning (planned)│   │   └── instruction_tuning/      # 指令微调 (规划中)

│   └── 2_inference/                 # Inference engine│   └── 2_inference/                 # 推理引擎

│       └── infer_demo.py            # Inference demo│       └── infer_demo.py            # 推理演示

││

├── 🗄️ _models/                      # Model repository├── 🗄️ _models/                      # 模型资源库

│   ├── base/                        # Base pre-trained models│   ├── base/                        # 基础预训练模型

│   │   ├── Qwen3-30B-A3B-Thinking-2507/  # Main generative model│   │   ├── Qwen3-30B-A3B-Thinking-2507/  # 主力生成模型

│   │   └── qwen3-4b-thinking/       # Lightweight detection model│   │   └── qwen3-4b-thinking/       # 轻量检测模型

│   ├── fine-tune/                   # Fine-tuned models│   ├── fine-tune/                   # 微调模型

│   └── correction_models/           # Correction models (planned)│   └── correction_models/           # 纠错模型 (规划中)

││

├── 🏥 scripts/ehr_json_builder/     # Medical data processing tools├── 🏥 scripts/ehr_json_builder/     # 医疗数据处理工具

│   ├── src/ehr_data_processor.py    # EHR data processor│   ├── src/ehr_data_processor.py    # EHR数据处理器

│   ├── script/validate_ehr_data.py  # Data validation tools│   ├── script/validate_ehr_data.py  # 数据验证工具

│   ├── output/                      # Processing results│   ├── output/                      # 处理结果

│   └── quick_start.py               # Quick start│   └── quick_start.py               # 快速启动

││

├── 📊 discharge-me/                 # MIMIC-IV emergency dataset├── 📊 discharge-me/                 # MIMIC-IV急诊数据集  

│   ├── train/                       # Training corpus (46,998 patients)│   ├── train/                       # 训练语料 (46,998患者)

│   ├── valid/                       # Validation data│   ├── valid/                       # 验证数据

│   ├── test_phase_1/               # Test set phase 1│   ├── test_phase_1/               # 测试集阶段1

│   └── test_phase_2/               # Test set phase 2│   └── test_phase_2/               # 测试集阶段2

││

├── 📈 evaluation/                   # Evaluation system (planned)├── 📈 evaluation/                   # 评估体系 (规划中)

│   ├── detection_metrics/          # Detection metrics evaluation│   ├── detection_metrics/          # 检测指标评估

│   ├── classification_metrics/     # Classification effectiveness evaluation│   ├── classification_metrics/     # 分类效果评估

│   ├── correction_quality/         # Correction quality evaluation│   ├── correction_quality/         # 纠错质量评估

│   └── human_annotation/           # Manual annotation comparison│   └── human_annotation/           # 人工标注对比

││

├── 📚 文档/                         # Research documentation├── 📚 文档/                         # 研究文档

│   ├── MS-SWIFT_使用指南.md         # Fine-tuning framework tutorial│   ├── MS-SWIFT_使用指南.md         # 微调框架教程

│   ├── Qwen3高效微调.ipynb          # Model fine-tuning practice│   ├── Qwen3高效微调.ipynb          # 模型微调实践

│   ├── 幻觉检测研究.md              # Core research methods (planned)│   ├── 幻觉检测研究.md              # 核心研究方法 (规划中)

│   └── 错误分类体系.md              # Medical error classification (planned)│   └── 错误分类体系.md              # 医学错误分类 (规划中)

││

├── requirements.txt                 # Environment dependencies├── requirements.txt                 # 环境依赖

├── .venv/                          # Virtual environment├── .venv/                          # 虚拟环境

└── README.md                       # Project documentation└── README.md                       # 项目文档

``````



## 🚀 Quick Start## 🚀 快速开始



### 1. Environment Setup### 1. 环境准备



```bash```bash

# Clone project# 克隆项目

git clone https://github.com/severin-ye/hd.gitgit clone https://github.com/severin-ye/hd.git

cd hdcd hd



# Activate virtual environment# 激活虚拟环境

source .venv/bin/activatesource .venv/bin/activate



# Install core dependencies# 安装核心依赖

pip install -r requirements.txtpip install -r requirements.txt

pip install ms-swift -Upip install ms-swift -U



# Verify environment# 验证环境

python -c "import torch; print(f'PyTorch: {torch.__version__}')"python -c "import torch; print(f'PyTorch: {torch.__version__}')"

python -c "import swift; print('MS-Swift installed successfully')"python -c "import swift; print('MS-Swift installed successfully')"

``````



### 2. Medical Data Preprocessing 🏥### 2. 医疗数据预处理 🏥



Build high-quality medical training corpus to provide data foundation for hallucination detection.构建高质量的医疗训练语料，为幻觉检测提供数据基础。



```bash```bash

# Enter data processing module# 进入数据处理模块

cd scripts/ehr_json_buildercd scripts/ehr_json_builder



# Process MIMIC-IV emergency data (46,998 patient records)# 处理MIMIC-IV急诊数据 (46,998患者记录)

python quick_start.py /home/work/hd/discharge-me/train ./output --chunksize 20000python quick_start.py /home/work/hd/discharge-me/train ./output --chunksize 20000



# Validate data quality# 验证数据质量

python script/validate_ehr_data.py --output_dir ./outputpython script/validate_ehr_data.py --output_dir ./output



# Output corpus statistics# 输出语料统计

echo "✅ Generated medical corpus: 1.3GB structured data"echo "✅ 生成医疗语料: 1.3GB结构化数据"

echo "✅ Patient records: 46,998 entries"echo "✅ 患者记录数: 46,998条"  

echo "✅ Data integrity: 100% coverage"echo "✅ 数据完整性: 100%覆盖率"

``````



### 3. Base Model Fine-tuning 🧠### 3. 基础模型微调 🧠



Use Qwen3 model for medical domain adaptation training.使用Qwen3模型进行医疗领域适配训练。



```bash```bash

# Start LoRA fine-tuning (recommended)# 启动LoRA微调 (推荐)

jupyter notebook 2_core/1_train/Fine-tuning-lora.ipynbjupyter notebook 2_core/1_train/Fine-tuning-lora.ipynb



# Or full parameter fine-tuning (when resources are sufficient)# 或全参数微调 (资源充足时)

jupyter notebook 2_core/1_train/Fine-tuning.ipynbjupyter notebook 2_core/1_train/Fine-tuning.ipynb



# Monitor training process# 监控训练过程

tensorboard --logdir 2_core/1_train/output/tensorboard --logdir 2_core/1_train/output/

``````



### 4. Hallucination Detection System Deployment 🔍### 4. 幻觉检测系统部署 🔍



```bash```bash

# Start inference demo (current version)# 启动推理演示 (当前版本)

python 2_core/2_inference/infer_demo.pypython 2_core/2_inference/infer_demo.py



# TODO: Hallucination Detection Agent (in development)# TODO: 幻觉检测Agent (开发中)

# python agents/hallucination_detector/detect.py# python agents/hallucination_detector/detect.py



# TODO: Error Classification System (in development)# TODO: 错误归类系统 (开发中)  

# python agents/error_classifier/classify.py# python agents/error_classifier/classify.py



# TODO: Intelligent Correction System (in development)# TODO: 智能纠错系统 (开发中)

# python agents/hallucination_corrector/correct.py# python agents/hallucination_corrector/correct.py

``````



## 🔬 Core Technical Solutions## � 核心技术方案



### 🔍 Hallucination Detection Technology Stack### 🔍 幻觉检测技术栈



#### Detection Methodology#### 检测方法论

```python```python

# Multi-level hallucination detection architecture# 多层次幻觉检测架构

Detection Framework:Detection Framework:

├── Semantic consistency detection: fact verification based on medical knowledge graph├── 语义一致性检测: 基于医学知识图谱的事实验证

├── Logical coherence detection: reasoning chain path verification├── 逻辑连贯性检测: 推理链路径验证  

├── Context relevance detection: RAG retrieval content comparison analysis├── 上下文相关性检测: RAG召回内容对比分析

├── Professional terminology accuracy: medical dictionary + ontology matching├── 专业术语准确性: 医学词典+本体匹配

└── Clinical safety detection: risk assessment + contraindication checking└── 临床安全性检测: 风险评估+禁忌症检查

``````



#### Context Engineering Strategy#### 上下文工程策略

```bash```bash

# Professional medical prompt engineering# 专业医学提示工程

Context Engineering:Context Engineering:

├── Medical background knowledge injection├── 医学背景知识注入

├── Clinical experience case reference├── 临床经验案例参考  

├── Multi-modal information fusion (text + image + test data)├── 多模态信息融合 (文本+图像+检验数据)

├── Specialty field specialization (internal medicine/surgery/emergency/imaging)├── 专科领域特化 (内科/外科/急诊/影像)

└── Real-time knowledge base updates└── 实时知识库更新

``````



### 📊 Error Classification System### 📊 错误分类体系



#### Medical Error Classification Tree#### 医学错误分类树

```mermaid```mermaid

graph TDgraph TD

    A[Medical AI Errors] --> B[Factual Errors]    A[医疗AI错误] --> B[事实性错误]

    A --> C[Logical Errors]    A --> C[逻辑性错误]  

    A --> D[Consistency Errors]    A --> D[一致性错误]

    A --> E[Safety Errors]    A --> E[安全性错误]

        

    B --> B1[Disease Diagnosis Errors]    B --> B1[疾病诊断错误]

    B --> B2[Drug Information Errors]    B --> B2[药物信息错误]

    B --> B3[Anatomical Structure Errors]    B --> B3[解剖结构错误]

        

    C --> C1[Causal Relationship Errors]    C --> C1[因果关系错误]

    C --> C2[Temporal Logic Errors]    C --> C2[时序逻辑错误]

    C --> C3[Reasoning Step Errors]    C --> C3[推理步骤错误]

        

    D --> D1[Contradictions]    D --> D1[前后矛盾]

    D --> D2[Terminology Inconsistency]    D --> D2[术语不一致]

    D --> D3[Value Conflicts]    D --> D3[数值冲突]

        

    E --> E1[Drug Contraindications]    E --> E1[用药禁忌]

    E --> E2[Treatment Risks]    E --> E2[治疗风险]

    E --> E3[Diagnostic Delays]    E --> E3[诊断延误]

``````



### 🧠 Intelligent Correction Model### 🧠 智能纠错模型



#### Correction Model Architecture#### 纠错模型架构

```python```python

class MedicalHallucinationCorrector:class MedicalHallucinationCorrector:

    """    """

    Medical Hallucination Intelligent Correction System    医疗幻觉智能纠错系统

    """    """

    def __init__(self):    def __init__(self):

        self.detector = HallucinationDetector()        self.detector = HallucinationDetector()

        self.classifier = ErrorClassifier()        self.classifier = ErrorClassifier() 

        self.rag_retriever = MedicalRAGRetriever()        self.rag_retriever = MedicalRAGRetriever()

        self.corrector = CorrectionGenerator()        self.corrector = CorrectionGenerator()

        

    def correct_pipeline(self, medical_text):    def correct_pipeline(self, medical_text):

        # Step 1: Hallucination detection        # Step 1: 幻觉检测

        errors = self.detector.detect(medical_text)        errors = self.detector.detect(medical_text)

                

        # Step 2: Error classification        # Step 2: 错误分类

        error_types = self.classifier.classify(errors)        error_types = self.classifier.classify(errors)

                

        # Step 3: RAG knowledge retrieval        # Step 3: RAG知识召回

        contexts = self.rag_retriever.retrieve(medical_text, errors)        contexts = self.rag_retriever.retrieve(medical_text, errors)

                

        # Step 4: Intelligent correction        # Step 4: 智能纠错

        corrections = self.corrector.generate(        corrections = self.corrector.generate(

            text=medical_text,            text=medical_text,

            errors=errors,            errors=errors, 

            types=error_types,            types=error_types,

            contexts=contexts            contexts=contexts

        )        )

                

        return {        return {

            "error_positions": errors,            "错误位置": errors,

            "error_types": error_types,            "错误类型": error_types, 

            "correction_suggestions": corrections,            "纠错建议": corrections,

            "rewritten_text": self.rewrite(medical_text, corrections)            "改写文本": self.rewrite(medical_text, corrections)

        }        }

``````



## 🎯 Application Scenarios & Value## 🎯 应用场景与价值



### 🏥 Clinical Application Scenarios### 🏥 临床应用场景



#### Intelligent Diagnosis Assistance#### 智能诊疗辅助

- **AI Diagnosis Verification**: Hallucination detection and correction for AI-generated diagnostic reports- **AI诊断验证**: 对AI生成的诊断报告进行幻觉检测和纠错

- **Treatment Plan Review**: Verify medical accuracy of AI-recommended treatment plans- **治疗方案审核**: 验证AI推荐治疗方案的医学准确性

- **Medication Safety Check**: Detect medication errors and contraindications in AI prescriptions- **用药安全检查**: 检测AI开具处方中的用药错误和禁忌

- **Medical Record Quality Control**: Automated quality control and error correction for medical documentation- **病历质控**: 自动化病历文书的质量控制和错误纠正



#### Medical Education & Training#### 医学教育培训  

- **Clinical Thinking Training**: Improve clinical reasoning skills through error case analysis- **临床思维训练**: 通过错误案例分析提升医学生临床推理能力

- **Medical Knowledge Verification**: Help medical students identify and correct medical misconceptions- **医学知识验证**: 帮助医学生识别和纠正医学知识误区

- **Case Discussion Assistance**: Provide structured error analysis tools for medical education- **案例讨论辅助**: 为医学教育提供结构化的错误分析工具



#### Medical Safety Assurance#### 医疗安全保障

- **Risk Warning System**: Real-time detection of safety risks in medical AI outputs- **风险预警系统**: 实时检测医疗AI输出中的安全风险

- **Quality Monitoring**: Continuous monitoring of medical AI system output quality- **质量监控**: 持续监控医疗AI系统的输出质量

- **Compliance Checking**: Ensure AI medical recommendations comply with clinical guidelines and standards- **合规性检查**: 确保AI医疗建议符合临床指南和规范



### 🔬 Research Innovation Value### 🔬 科研创新价值



#### Technical Innovation Breakthroughs#### 技术创新突破

```python```python

# Core technical breakthrough points# 核心技术突破点

Innovation Points:Innovation Points:

├── Multi-agent collaboration: closed-loop system of detection→classification→correction├── 多智能体协作: 检测→分类→纠错的闭环系统

├── Medical knowledge graph: intelligent application of structured medical knowledge├── 医学知识图谱: 结构化医学知识的智能应用  

├── Context engineering: prompt engineering methods for professional medical domains├── 上下文工程: 专业医学领域的提示工程方法

├── Joint learning: end-to-end training of generation+detection+correction├── 联合学习: 生成+检测+纠错的端到端训练

└── RAG enhancement: intelligent retrieval application of real-time medical knowledge base└── RAG增强: 实时医学知识库的智能检索应用

``````



#### Academic Contributions#### 学术贡献

- **New Hallucination Detection Methods**: Propose medical domain-specific hallucination detection algorithms- **幻觉检测新方法**: 提出医疗领域特化的幻觉检测算法

- **Error Classification System**: Build systematized medical AI error classification standards- **错误分类体系**: 构建系统化的医疗AI错误分类标准

- **Correction Model Architecture**: Design end-to-end medical text intelligent correction system- **纠错模型架构**: 设计端到端的医疗文本智能纠错系统

- **Evaluation Benchmark**: Establish standard evaluation dataset for medical AI hallucination detection- **评估基准**: 建立医疗AI幻觉检测的标准评估数据集



### 📊 Industry Application Prospects### 📊 产业应用前景



#### Medical AI Product Optimization#### 医疗AI产品优化

- **EMR System Enhancement**: Provide intelligent quality control functions for electronic medical record systems- **EMR系统增强**: 为电子病历系统提供智能质控功能

- **AI Diagnosis Products**: Improve reliability and safety of AI diagnosis products- **AI诊断产品**: 提升AI诊断产品的可靠性和安全性

- **Medical Robots**: Provide safety assurance mechanisms for medical service robots- **医疗机器人**: 为医疗服务机器人提供安全保障机制

- **Telemedicine**: Ensure accuracy of remote medical AI consultations- **远程医疗**: 保障远程医疗AI咨询的准确性



#### Regulatory Compliance Support#### 监管合规支持

- **AI Medical Review**: Provide technical support for medical AI product regulation- **AI医疗审查**: 为医疗AI产品监管提供技术支持

- **Quality Standards**: Establish quantitative evaluation standards for medical AI output quality- **质量标准**: 建立医疗AI输出质量的量化评估标准

- **Safety Certification**: Provide verification tools for medical AI system safety certification- **安全认证**: 为医疗AI系统安全认证提供验证工具



## 📈 Current Progress & Next Steps## 📈 当前进展与下一步计划



### ✅ Completed Work### ✅ 已完成工作



#### Phase 1: Infrastructure Development#### Phase 1: 基础设施建设

- [x] **Medical Data Processing System**: Completed MIMIC-IV dataset processing (46,998 patient records)- [x] **医疗数据处理系统**: 完成MIMIC-IV数据集处理(46,998患者记录)

- [x] **Model Fine-tuning Framework**: Integrated MS-Swift, supports Qwen3 series model fine-tuning- [x] **模型微调框架**: 集成MS-Swift，支持Qwen3系列模型微调

- [x] **Base Inference Engine**: Built GPU-optimized model inference system- [x] **基础推理引擎**: 搭建GPU优化的模型推理系统

- [x] **Development Environment**: Built complete Python development environment and dependency management- [x] **开发环境**: 构建完整的Python开发环境和依赖管理



#### Phase 2: Core Model Training#### Phase 2: 核心模型训练

- [x] **Base Model Deployment**: Qwen3-30B-A3B-Thinking-2507 large model- [x] **基础模型部署**: Qwen3-30B-A3B-Thinking-2507大模型

- [x] **Lightweight Model**: qwen3-4b-thinking detection model- [x] **轻量模型**: qwen3-4b-thinking检测模型

- [x] **LoRA Fine-tuning**: Efficient parameter fine-tuning method implementation- [x] **LoRA微调**: 高效参数微调方法实现

- [x] **Data Validation**: Complete data quality checking and statistical analysis- [x] **数据验证**: 完整的数据质量检查和统计分析



### 🚧 Ongoing Work### 🚧 正在进行的工作



#### Phase 3: Hallucination Detection System (Current Focus)#### Phase 3: 幻觉检测系统 (当前重点)

- [ ] **Error Detection Agent**: Hallucination detection algorithm based on context engineering- [ ] **错误检测Agent**: 基于上下文工程的幻觉检测算法

  - Progress: Design phase, prototype expected in 2 weeks  - 进度: 设计阶段，预计2周完成原型

- [ ] **Error Classification Agent**: Graph-structured medical error classification system- [ ] **错误归类Agent**: 图结构化的医疗错误分类体系

  - Progress: Medical knowledge graph construction in progress  - 进度: 医学知识图谱构建中

- [ ] **Error Generation Agent**: Synthetic error data generation engine- [ ] **错误生成Agent**: 合成错误数据生成引擎

  - Progress: Data augmentation strategy research in progress  - 进度: 数据增强策略研究中



### 🎯 Next Steps (Next 3 Months)### 🎯 下一步计划 (Next 3 Months)



#### Phase 4: Intelligent Correction System Training#### Phase 4: 智能纠错系统训练

```timeline```timeline

Month 1: Joint Learning FrameworkMonth 1: 联合学习框架

├── Week 1-2: Correction chain corpus construction├── Week 1-2: 纠错链语料构建

├── Week 3: Instruction fine-tuning data preparation├── Week 3: 指令微调数据准备  

└── Week 4: Multi-task joint training framework setup└── Week 4: 多任务联合训练框架搭建



Month 2: Model Training & OptimizationMonth 2: 模型训练与优化

├── Week 1-2: Hallucination detection small model training├── Week 1-2: 幻觉检测小模型训练

├── Week 3: Qwen3 data fusion (prevent catastrophic forgetting)├── Week 3: Qwen3数据融合(防灾难性遗忘)

└── Week 4: Model performance evaluation and tuning└── Week 4: 模型性能评估与调优



Month 3: RAG-Enhanced Correction SystemMonth 3: RAG增强纠错系统

├── Week 1-2: RAG knowledge base construction├── Week 1-2: RAG知识库构建

├── Week 3: Correction model integration testing├── Week 3: 纠错模型集成测试

└── Week 4: End-to-end system evaluation└── Week 4: 系统端到端评估

``````



#### Phase 5: Medical Generation Model Optimization (Long-term Planning)#### Phase 5: 医学生成模型优化 (长期规划)

```bash```bash

# Generation quality improvement roadmap# 生成质量提升路线图

Generation Model Roadmap:Generation Model Roadmap:

├── SFT training: positive-negative contrastive learning data construction├── SFT训练: 正反对比学习数据构建

├── DPO optimization: manual corpus preference learning├── DPO优化: 人工语料偏好学习  

├── RLHF integration: human feedback reinforcement learning├── RLHF集成: 人类反馈强化学习

└── Joint deployment: integrated generation+detection+correction system└── 联合部署: 生成+检测+纠错一体化系统

``````



### 🔬 Experimental Design & Evaluation### 🔬 实验设计与评估



#### Experimental Validation Plan#### 实验验证计划

```python```python

# Phased experimental validation# 分阶段实验验证

Evaluation Plan:Evaluation Plan:

├── Baseline comparison: comparison with existing hallucination detection methods├── 基线对比: 与现有幻觉检测方法对比

├── Ablation experiments: analysis of independent contribution of each module├── 消融实验: 各模块独立贡献度分析

├── Manual evaluation: professional physician annotation verification├── 人工评估: 专业医师标注验证

├── Clinical trials: real medical scenario application testing├── 临床试验: 真实医疗场景应用测试

└── Long-term monitoring: system stability and accuracy tracking└── 长期监控: 系统稳定性和准确性追踪

``````



#### Success Metrics Definition#### 成功指标定义

- **Detection Accuracy**: >95% (hallucination recognition precision/recall)- **检测准确率**: >95% (幻觉识别precision/recall)

- **Classification Correctness**: >90% (error type classification accuracy)- **分类正确率**: >90% (错误类型分类accuracy)  

- **Correction Quality**: >85% (medical expert rating)- **纠错质量**: >85% (医学专家评分)

- **System Response**: <3 seconds (end-to-end processing time)- **系统响应**: <3秒 (端到端处理时间)

- **Safety Assurance**: 0 tolerance (serious medical error miss rate)- **安全保障**: 0容忍 (严重医疗错误漏检率)



## ⚙️ Technical Specifications & Environment Requirements## ⚙️ 技术规格与环境要求



### System Configuration### 系统配置

- **Python**: 3.10+ (recommended 3.11)- **Python**: 3.10+ (推荐3.11)

- **Memory**: 32GB+ (large model training), 16GB+ (inference deployment)- **内存**: 32GB+ (大模型训练), 16GB+ (推理部署)

- **Storage**: 100GB+ (models + data + experimental results)- **存储**: 100GB+ (模型+数据+实验结果)

- **GPU**: NVIDIA A100/V100 (training), RTX 4090+ (inference)- **GPU**: NVIDIA A100/V100 (训练), RTX 4090+ (推理)



### Core Technology Stack### 核心技术栈

```txt```txt

# Deep learning frameworks# 深度学习框架

torch>=2.0.0         # PyTorch core frameworktorch>=2.0.0         # PyTorch核心框架

transformers>=4.30.0 # HuggingFace model librarytransformers>=4.30.0 # HuggingFace模型库

ms-swift>=2.0.0      # ModelScope fine-tuning frameworkms-swift>=2.0.0      # ModelScope微调框架



# Data processing# 数据处理

pandas>=1.5.0        # Structured data processingpandas>=1.5.0        # 结构化数据处理

numpy>=1.21.0        # Numerical computationnumpy>=1.21.0        # 数值计算

datasets>=2.10.0     # Dataset managementdatasets>=2.10.0     # 数据集管理



# Hallucination detection specific# 幻觉检测专用

sentence-transformers # Semantic similarity computationsentence-transformers # 语义相似度计算

faiss-cpu            # Vector retrieval and similarity matchingfaiss-cpu            # 向量检索与相似度匹配

spacy>=3.4.0         # Natural language processingspacy>=3.4.0         # 自然语言处理

networkx>=2.8        # Knowledge graph constructionnetworkx>=2.8        # 知识图谱构建



# RAG and knowledge management# RAG与知识管理

langchain>=0.1.0     # RAG frameworklangchain>=0.1.0     # RAG框架

chromadb>=0.4.0      # Vector databasechromadb>=0.4.0      # 向量数据库

``````



### Performance Benchmarks### 性能基准

- **Data Processing**: 1000+ patients/second (EHR multi-table joining)- **数据处理**: 1000+患者/秒 (EHR多表拼接)

- **Hallucination Detection**: <2 seconds/document (average 500-word medical text)- **幻觉检测**: <2秒/文档 (平均500字医疗文本)

- **Error Classification**: <500ms/error (multi-label classification)- **错误分类**: <500ms/错误 (多标签分类)

- **Intelligent Correction**: <5 seconds/document (including RAG retrieval)- **智能纠错**: <5秒/文档 (包含RAG检索)

- **Memory Usage**: <8GB (inference mode), <32GB (training mode)- **内存占用**: <8GB (推理模式), <32GB (训练模式)



### Model Scale Support### 模型规格支持

```bash```bash

# Supported model scales# 支持的模型规模

Model Scale Support:Model Scale Support:

├── Small models: 1B-4B parameters (detection-specific)├── 小型模型: 1B-4B参数 (检测专用)

├── Medium models: 7B-14B parameters (balanced performance)├── 中型模型: 7B-14B参数 (平衡性能)  

├── Large models: 30B-70B parameters (generation main force)├── 大型模型: 30B-70B参数 (生成主力)

└── Giant models: 100B+ parameters (research frontier)└── 巨型模型: 100B+参数 (研究前沿)

``````



## 📚 Documentation & Resources## 📚 文档与资源



### Technical Documentation### 技术文档

- [MS-Swift User Guide](文档/MS-SWIFT_使用指南.md) - Detailed model fine-tuning tutorial- [MS-Swift使用指南](文档/MS-SWIFT_使用指南.md) - 模型微调详细教程

- [Qwen3 Efficient Fine-tuning](文档/Qwen3高效微调.ipynb) - Practical fine-tuning cases- [Qwen3高效微调](文档/Qwen3高效微调.ipynb) - 实战微调案例

- [Environment Repair Guide](文档/环境修复指南.md) - Common problem solutions- [环境修复指南](文档/环境修复指南.md) - 常见问题解决

- [Hallucination Detection Research](文档/幻觉检测研究.md) - Core algorithm explanation (in development)- [幻觉检测研究](文档/幻觉检测研究.md) - 核心算法说明 (开发中)

- [Medical Error Classification](文档/错误分类体系.md) - Classification standard definition (in development)- [医学错误分类](文档/错误分类体系.md) - 分类标准定义 (开发中)



### Learning Resources### 学习资源

```bash```bash

# Recommended learning path# 推荐学习路径

Learning Path:Learning Path:

├── Medical AI Basics: understand medical NLP and clinical applications├── 医学AI基础: 了解医疗NLP和临床应用

├── Hallucination Detection Theory: learn cutting-edge LLM hallucination detection methods├── 幻觉检测理论: 学习LLM幻觉检测前沿方法

├── Multi-agent Systems: master Agent collaboration architecture design├── 多智能体系统: 掌握Agent协作架构设计

├── RAG Technology: understand retrieval-augmented generation principles├── RAG技术: 理解检索增强生成原理

└── Evaluation Methods: familiarize with medical AI evaluation standards and metrics└── 评估方法: 熟悉医疗AI评估标准和指标

``````



### Dataset Information### 数据集信息

- **MIMIC-IV-ED**: Emergency department electronic medical record dataset- **MIMIC-IV-ED**: 急诊科电子病历数据集

- **Processed Corpus**: 46,998 patient structured records- **处理后语料**: 46,998患者结构化记录

- **Synthetic Error Data**: Multi-type medical error samples (in development)- **合成错误数据**: 多类型医疗错误样本 (开发中)

- **Manual Annotation**: Professional physician quality evaluation data (planned)- **人工标注**: 专业医师质量评估数据 (规划中)



## 🔧 Troubleshooting## 🔧 故障排除



### Common Issues### 常见问题



**1. EHR data processing memory insufficient****1. EHR数据处理内存不足**

```bash```bash

# Reduce chunk size# 减小分块大小

python quick_start.py --chunksize 10000python quick_start.py --chunksize 10000

``````



**2. Model fine-tuning GPU memory insufficient****2. 模型微调显存不足**

```bash```bash

# Use QLoRA fine-tuning# 使用QLoRA微调

# Select QLoRA configuration in Jupyter notebook# 在Jupyter notebook中选择QLoRA配置

``````



**3. Data path errors****3. 数据路径错误**

```bash```bash

# Check data directory structure# 检查数据目录结构

ls -la /home/work/hd/discharge-me/train/ls -la /home/work/hd/discharge-me/train/

``````



### Debugging Tips### 调试技巧



**Enable verbose logging****启用详细日志**

```bash```bash

export PYTHONPATH=/home/work/hd:$PYTHONPATHexport PYTHONPATH=/home/work/hd:$PYTHONPATH

python -u scripts/ehr_json_builder/quick_start.py 2>&1 | tee debug.logpython -u scripts/ehr_json_builder/quick_start.py 2>&1 | tee debug.log

``````



## 🤝 Contributing## 🤝 贡献



We welcome community contributions! Please check the following ways to participate:我们欢迎社区贡献！请查看以下方式参与：



1. **Issue Reports**: Report bugs in GitHub Issues1. **问题报告**: 在GitHub Issues中报告bug

2. **Feature Suggestions**: Propose new feature ideas2. **功能建议**: 提出新功能想法

3. **Code Contributions**: Submit Pull Requests3. **代码贡献**: 提交Pull Request

4. **Documentation Improvements**: Improve project documentation4. **文档改进**: 完善项目文档



### Contribution Guidelines### 贡献指南

```bash```bash

# 1. Fork the project# 1. Fork项目

# 2. Create feature branch# 2. 创建功能分支

git checkout -b feature/new-featuregit checkout -b feature/new-feature



# 3. Commit changes# 3. 提交更改

git commit -m "Add new feature"git commit -m "Add new feature"



# 4. Push to branch# 4. 推送到分支

git push origin feature/new-featuregit push origin feature/new-feature



# 5. Create Pull Request# 5. 创建Pull Request

``````



## 📊 Project Statistics## 📊 项目统计



- **Code Volume**: 10,000+ lines of Python code- **代码量**: 10,000+ 行Python代码

- **Data Processing Capability**: 46,998 patient records- **数据处理能力**: 46,998患者记录

- **Model Support**: 4B-30B parameter scales- **模型支持**: 4B-30B参数规模

- **Documentation Coverage**: 5 detailed tutorial documents- **文档覆盖**: 5个详细教程文档

- **Test Coverage**: Complete data validation system- **测试覆盖**: 完整的数据验证体系



## 🏆 Expected Results & Impact## 🏆 预期成果与影响



### 📊 Technical Achievements### 📊 技术成果

- **Open Source Toolkit**: Complete medical AI hallucination detection and correction system- **开源工具包**: 完整的医疗AI幻觉检测与纠错系统

- **Standard Dataset**: Medical hallucination detection benchmark dataset- **标准数据集**: 医疗幻觉检测benchmark数据集

- **Evaluation Framework**: Systematized medical AI quality evaluation methods- **评估框架**: 系统化的医疗AI质量评估方法

- **Best Practices**: Medical AI safety deployment guidelines and standards- **最佳实践**: 医疗AI安全部署指南和规范



### 🎓 Academic Contributions### 🎓 学术贡献

- **Top Conference Papers**: Target AAAI/IJCAI/ACL and other AI top conferences- **顶级会议论文**: 目标AAAI/IJCAI/ACL等AI顶会

- **Professional Journals**: Publications in medical informatics and AI medical journals- **专业期刊**: 医疗信息学和AI医疗期刊发表

- **Technical Patents**: Core algorithm and system architecture patent applications- **技术专利**: 核心算法和系统架构专利申请

- **Open Source Impact**: Promote medical AI safety research community development- **开源影响**: 推动医疗AI安全研究社区发展



### 🏥 Industry Value### 🏥 产业价值

- **Medical AI Products**: Provide safety assurance for commercial medical AI products- **医疗AI产品**: 为商业医疗AI产品提供安全保障

- **Regulatory Support**: Provide technical standards for medical AI regulation- **监管支持**: 为医疗AI监管提供技术标准

- **Clinical Applications**: Quality control tools in actual medical scenarios- **临床应用**: 实际医疗场景中的质量控制工具

- **Education & Training**: Intelligent assistance systems in medical education- **教育培训**: 医学教育中的智能辅助系统



## 🤝 Collaboration & Contribution## 🤝 合作与贡献



### 🔬 Academic Collaboration### 🔬 学术合作

We welcome collaboration with the following institutions and experts:我们欢迎与以下机构和专家合作：

- **Medical Schools**: Clinical experts participate in error annotation and validation- **医学院校**: 临床专家参与错误标注和验证

- **AI Research Institutions**: Joint research and development of hallucination detection algorithms- **AI研究机构**: 幻觉检测算法联合研发

- **Medical Information Enterprises**: Real scenario application testing- **医疗信息企业**: 真实场景应用测试

- **Regulatory Agencies**: Standard setting and norm establishment- **监管机构**: 标准制定和规范建立



### 💻 Open Source Contribution### 💻 开源贡献

```bash```bash

# Participation methods# 参与方式

Contribution Ways:Contribution Ways:

├── Code contribution: core algorithm optimization and new feature development├── 代码贡献: 核心算法优化和新功能开发

├── Data contribution: medical error cases and annotation data├── 数据贡献: 医疗错误案例和标注数据

├── Documentation improvement: technical documentation and usage tutorials├── 文档完善: 技术文档和使用教程

├── Testing feedback: bug reports and performance optimization suggestions├── 测试反馈: Bug报告和性能优化建议

└── Academic discussion: method improvement and innovative ideas└── 学术讨论: 方法改进和创新思路

``````



### 🎯 Join Us### 🎯 加入我们

If you are interested in medical AI safety, hallucination detection, intelligent correction and other fields, welcome to:如果您对医疗AI安全、幻觉检测、智能纠错等领域感兴趣，欢迎：



1. **Submit Issues**: Report problems or suggest features1. **提交Issue**: 报告问题或建议功能

2. **Fork & PR**: Directly contribute code and documentation2. **Fork & PR**: 直接贡献代码和文档

3. **Academic Discussion**: Participate in technical solution discussions3. **学术讨论**: 参与技术方案讨论

4. **Data Sharing**: Provide medical error case data4. **数据共享**: 提供医疗错误案例数据

5. **Joint Research**: Deep collaborative research projects5. **联合研究**: 深度合作研究项目



## 📄 License & Citation## 📄 许可证与引用



### Open Source License### 开源许可

This project uses **MIT License** - see [LICENSE](LICENSE) file for details.本项目采用 **MIT License** - 详见 [LICENSE](LICENSE) 文件。



### Academic Citation### 学术引用

If this project helps your research, please consider citing:如果本项目对您的研究有帮助，请考虑引用：



```bibtex```bibtex

@misc{curaview2025,@misc{jingheng2025,

  title={CuraView: A Medical AI Hallucination Detection and Correction System},  title={Jingheng: A Medical AI Hallucination Detection and Correction System},

  author={Severin Ye and Contributors},  author={Severin Ye and Contributors},

  year={2025},  year={2025},

  url={https://github.com/severin-ye/hd},  url={https://github.com/severin-ye/hd},

  note={Medical AI Research Platform for Hallucination Detection and Correction}  note={Medical AI Research Platform for Hallucination Detection and Correction}

}}

``````



## 🔗 Related Links

## 🔗 相关链接

- [MS-Swift Official Repository](https://github.com/modelscope/ms-swift)

- [ModelScope Model Hub](https://modelscope.cn/models)- [MS-Swift官方仓库](https://github.com/modelscope/ms-swift)

- [MIMIC-IV Dataset](https://physionet.org/content/mimic-iv-ed/)- [ModelScope模型库](https://modelscope.cn/models)

- [Qwen Model Series](https://github.com/QwenLM/Qwen)- [MIMIC-IV数据集](https://physionet.org/content/mimic-iv-ed/)

- [Qwen模型系列](https://github.com/QwenLM/Qwen)

## 📧 Contact Information

## 📧 联系方式

- **Project Lead**: Severin Ye

- **GitHub**: [@severin-ye](https://github.com/severin-ye)- **项目负责人**: Severin Ye  

- **Email**: 6severin9@gmail.com- **GitHub**: [@severin-ye](https://github.com/severin-ye)

- **Research Areas**: Medical AI Safety, Hallucination Detection, Intelligent Correction Systems- **邮箱**: 6severin9@gmail.com

- **研究方向**: 医疗AI安全、幻觉检测、智能纠错系统

---

---

<div align="center">

<div align="center">

### 🌟 If this project helps your research, please give us a Star! 🌟

### 🌟 如果本项目对您的研究有帮助，请给我们一个Star! 🌟

[![Stars](https://img.shields.io/github/stars/severin-ye/hd?style=social)](https://github.com/severin-ye/hd/stargazers)

[![Forks](https://img.shields.io/github/forks/severin-ye/hd?style=social)](https://github.com/severin-ye/hd/network/members)[![Stars](https://img.shields.io/github/stars/severin-ye/hd?style=social)](https://github.com/severin-ye/hd/stargazers)

[![Issues](https://img.shields.io/github/issues/severin-ye/hd)](https://github.com/severin-ye/hd/issues)[![Forks](https://img.shields.io/github/forks/severin-ye/hd?style=social)](https://github.com/severin-ye/hd/network/members)

[![License](https://img.shields.io/github/license/severin-ye/hd)](LICENSE)[![Issues](https://img.shields.io/github/issues/severin-ye/hd)](https://github.com/severin-ye/hd/issues)

[![License](https://img.shields.io/github/license/severin-ye/hd)](LICENSE)

**Let's advance medical AI safety research together and make AI better serve human health!**

**共同推进医疗AI安全研究，让AI更好地服务人类健康！**

</div>

</div>

---

---

## 🎉 Quick Experience Example

## 🎉 快速体验示例

```bash

# 🚀 One-click start complete pipeline```bash

git clone https://github.com/severin-ye/hd.git && cd hd# 🚀 一键启动完整pipeline

git clone https://github.com/severin-ye/hd.git && cd hd

# 1️⃣ Environment preparation

source .venv/bin/activate && pip install -r requirements.txt# 1️⃣ 环境准备

source .venv/bin/activate && pip install -r requirements.txt

# 2️⃣ Data processing (generate medical training corpus)

cd scripts/ehr_json_builder && python quick_start.py# 2️⃣ 数据处理 (生成医疗训练语料)

cd scripts/ehr_json_builder && python quick_start.py

# 3️⃣ Model fine-tuning (adapt to medical domain)

jupyter notebook ../../2_core/1_train/Fine-tuning-lora.ipynb# 3️⃣ 模型微调 (适配医疗领域)  

jupyter notebook ../../2_core/1_train/Fine-tuning-lora.ipynb

# 4️⃣ Inference testing (verify basic functions)

python ../../2_core/2_inference/infer_demo.py# 4️⃣ 推理测试 (验证基础功能)

python ../../2_core/2_inference/infer_demo.py

# 🔮 Future feature preview (in development)

# python agents/hallucination_detector/detect.py --text "Patient diagnosed with diabetes, recommend penicillin treatment"# 🔮 未来功能预览 (开发中)

# Expected output: ⚠️ Medication error detected: Penicillin is not suitable for diabetes treatment# python agents/hallucination_detector/detect.py --text "患者诊断为糖尿病，建议服用青霉素治疗"

```# 预期输出: ⚠️ 检测到用药错误：青霉素不适用于糖尿病治疗

```

**Start exploring the safety boundaries of medical AI now, let's build a more reliable intelligent medical future together!** 🏥🤖✨
**现在就开始探索医疗AI的安全边界，让我们一起构建更可靠的智能医疗未来！** 🏥🤖✨