🩺 EHR 多表拼接 → 单患者 JSON 构建工具 (EHR JSON Builder)

高效、模块化的电子病历（EHR）多表数据处理与整合工具
支持临床摘要生成、幻觉检测、结构化建模等医学 AI 任务

🧭 目录

项目概述

主要功能

输入数据结构

安装与使用

输出文件说明

数据验证与质量检查

性能优化

核心架构设计

项目目录结构

应用场景

自定义扩展

故障排除与支持

版本与版权信息

📋 项目概述

该工具旨在将电子病历（EHR）系统中多个异构 CSV 数据表整合为以患者为中心的统一 JSON 文件。
输出结果支持模型训练（如临床摘要生成、幻觉检测）、数据分析及质量评估等多种场景。

🎯 主要功能

✅ 多表数据整合：自动读取和处理 6 个 EHR 数据表
✅ 患者级聚合：以 subject_id 为中心拼接完整医疗轨迹
✅ 结构化输出：生成标准 JSON 与 JSONL 格式
✅ 内存优化：支持分块处理、低内存模式
✅ 数据验证：自动字段完整性与覆盖率分析
✅ 扩展性强：易于添加新表与自定义字段映射

📁 输入数据结构
文件名	含义	示例字段
diagnosis.csv	诊断信息	subject_id, stay_id, icd_code, icd_title
discharge.csv	出院记录（文本）	note_id, subject_id, text
discharge_target.csv	出院指导/预测目标	note_id, discharge_instructions, brief_hospital_course
edstays.csv	急诊留观记录	subject_id, hadm_id, stay_id, intime, outtime, gender
radiology.csv	放射影像报告	note_id, subject_id, hadm_id, text
triage.csv	分诊信息	subject_id, temperature, heartrate, o2sat, pain

示例：

subject_id,stay_id,temperature,heartrate,resprate,o2sat,sbp,dbp,pain,acuity,chiefcomplaint
10000032,38112554,98.9,88.0,18.0,97.0,116.0,88.0,10,3.0,Abdominal distention

🔧 安装与使用
环境要求

Python 3.8+

pandas >= 1.5.0

numpy >= 1.21.0

安装依赖
pip install -r requirements.txt

🚀 快速开始
方法一：使用快速启动脚本（推荐）
python quick_start.py

方法二：标准处理流程
python run_ehr_processor.py --data_dir /path/to/data --output_dir ./output --chunksize 20000

方法三：模块化调用
from src.ehr_data_processor import EHRDataProcessor

processor = EHRDataProcessor(data_dir="/path/to/data", output_dir="./output")
processor.process_data(chunksize=10000)

参数说明
参数	默认值	说明
--data_dir	./train	原始 CSV 文件目录
--output_dir	./output	输出保存目录
--chunksize	10000	分块读取大小，0 表示一次性读取
--validate	False	是否在处理完成后运行验证脚本
📄 输出文件说明
文件名	格式	内容	用途
ehr_dataset_full.json	JSON	含完整元数据的全量数据集	离线分析 / 调试
ehr_patients.jsonl	JSONL	每行一个患者对象	模型训练 / 流式处理
processing_report.txt	TXT	处理统计与覆盖率报告	数据质量监控
patient_summary_stats.csv	CSV	患者信息汇总	统计分析
JSON 示例
{
  "name": "Patient 1",
  "universal": {
    "subject_id": 10000032,
    "hadm_id": 22841357,
    "stay_id": 38112554
  },
  "diagnosis": {"icd_title": "OTHER ASCITES"},
  "discharge": {"text": "出院记录文本"},
  "triage": {"temperature": 98.9, "heartrate": 88.0}
}

🔍 数据验证与质量检查

验证脚本：

python script/validate_ehr_data.py --output_dir ./output


验证功能：

✅ JSON 格式检查

✅ 字段完整性验证

✅ 数据覆盖率分析

✅ 样本展示与错误日志输出

🚀 性能优化
优化项	说明
分块读取	减少内存峰值，适用于百万级记录
O(1) 索引查找	预构建患者索引加速匹配
缓存机制	避免重复文件加载
流式写出	支持 .jsonl 实时输出

性能指标（基于 46,998 患者）：

处理时间：约 2 分钟

内存占用：< 2GB

数据完整性：99.9%

输出大小：

ehr_dataset_full.json: 646.3MB

ehr_patients.jsonl: 628.6MB

🧩 核心架构设计
数据处理流程
1. 文件读取 → 2. 清洗与标准化 → 3. 患者索引构建 → 4. JSON 拼接 → 5. 输出保存

主要模块
模块	功能
read_csv_file()	分块读取单个 CSV
clean_dataframe()	数据清洗与字段重命名
build_patient_index()	构建患者 ID 索引
build_patient_json()	生成单患者 JSON
generate_metadata()	输出元数据结构
save_results()	输出 JSON / JSONL 文件
📁 项目目录结构
ehr_json_builder/
├── README.md                    # 总说明
├── quick_start.py               # 快速启动入口
├── run/                         # 运行脚本
│   └── run_ehr_processor.py
├── src/                         # 核心逻辑
│   └── ehr_data_processor.py
├── script/                      # 工具脚本
│   └── validate_ehr_data.py
└── output/                      # 输出结果
    ├── ehr_dataset_full.json
    ├── ehr_patients.jsonl
    ├── processing_report.txt
    └── patient_summary_stats.csv

📈 应用场景
医疗 AI 模型训练

临床摘要生成（Clinical Summarization）

幻觉检测（Hallucination Detection）

再入院风险预测（Readmission Prediction）

数据分析与治理

患者路径追踪（Patient Journey）

医疗质量评估（Quality Monitoring）

资源利用优化（Resource Allocation）

教学与科研

临床语言建模（Clinical NLP）

医学数据标准化研究

医院信息系统结构化实验

🔧 自定义扩展
添加新表
self.file_config['custom_table'] = {
    'filename': 'custom.csv',
    'key_fields': ['subject_id'],
    'unique_fields': ['custom_field']
}

自定义字段映射

修改 metadata 或 clean_dataframe() 实现新的数据类型支持。

新输出格式

在 save_results() 中扩展 .parquet、.avro 等格式。

🧠 故障排除与支持
问题	解决方案
内存不足	减小 chunksize 参数
文件路径错误	检查 --data_dir 是否包含目标 CSV
输出不完整	查看日志 processing_report.txt
无法写入输出目录	检查文件夹权限

调试运行：

python quick_start.py --chunksize 1000 --validate

📜 版本与版权信息
项目属性	信息
版本号	v1.0
发布日期	2025-10-18
作者	Severin Ye / 叶博韬
测试状态	✅ 已验证（46,998 患者记录）
许可证	MIT License
🎉 总结

EHR JSON Builder 是一个高性能、可扩展、面向 AI 研究的 EHR 数据整合工具。
通过多表拼接与结构化输出，它为临床自然语言处理、医疗大模型训练及医疗数据治理提供了坚实的技术基础。

使用命令：

python quick_start.py