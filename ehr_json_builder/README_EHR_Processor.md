# 🩺 EHR 多表拼接 → 单患者 JSON 构建工具

## 📋 项目概述

这个项目实现了一个高效的数据管道，用于将电子病历（EHR）系统中的多个 CSV 表格数据整合成结构化的单患者 JSON 文件，专门用于**临床摘要生成**和**幻觉检测**模型训练。

## 🎯 主要功能

✅ **多表数据整合**: 自动读取和处理 6 个 EHR 数据表  
✅ **患者为中心**: 按 `subject_id` 聚合每个患者的完整医疗记录  
✅ **结构化输出**: 生成标准化的 JSON 格式，包含元数据和患者数据  
✅ **内存优化**: 支持大数据文件的分块处理，防止内存溢出  
✅ **数据验证**: 内置数据质量检查和覆盖率分析  
✅ **双格式输出**: 同时生成 `.json` 和 `.jsonl` 格式文件  

## 📁 输入数据结构

项目处理以下 6 个 CSV 数据表：

### 1. `diagnosis.csv` - 诊断信息
```csv
subject_id,stay_id,seq_num,icd_code,icd_version,icd_title
10000032,38112554,1,78959,9,OTHER ASCITES
```

### 2. `discharge.csv` - 出院记录（实际为出院报告文本）
```csv
note_id,subject_id,hadm_id,note_type,note_seq,charttime,storetime,text
10000032-DS-22,10000032,22841357,DS,22,2180-06-27 00:00:00,2180-07-01 10:15:00,"详细出院报告..."
```

### 3. `discharge_target.csv` - 出院预测目标（实际为出院指导）
```csv
note_id,hadm_id,discharge_instructions,brief_hospital_course,discharge_instructions_word_count,brief_hospital_course_word_count
15373895-DS-19,28448473,"出院指导内容...","住院过程摘要...",760,398
```

### 4. `edstays.csv` - 急诊留观记录
```csv
subject_id,hadm_id,stay_id,intime,outtime,gender,race,arrival_transport,disposition
10000032,22841357,38112554,2180-06-26 15:54:00,2180-06-26 21:31:00,F,WHITE,AMBULANCE,ADMITTED
```

### 5. `radiology.csv` - 放射影像报告
```csv
note_id,subject_id,hadm_id,note_type,note_seq,charttime,storetime,text
10000032-RR-22,10000032,22841357,RR,22,2180-06-26 17:15:00,2180-06-26 19:28:00,"影像检查报告..."
```

### 6. `triage.csv` - 分诊信息
```csv
subject_id,stay_id,temperature,heartrate,resprate,o2sat,sbp,dbp,pain,acuity,chiefcomplaint
10000032,38112554,98.9,88.0,18.0,97.0,116.0,88.0,10,3.0,Abdominal distention
```

## 🔧 安装和使用

### 环境要求
- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.21.0

### 安装依赖
```bash
pip install -r requirements_ehr.txt
```

### 快速开始

#### 1. 使用示例脚本（推荐）
```bash
python run_ehr_processor.py
```

#### 2. 命令行使用
```bash
python ehr_data_processor.py --data_dir /path/to/csv/files --output_dir ./output
```

#### 3. 自定义参数
```bash
python ehr_data_processor.py \
    --data_dir /home/work/hd/discharge-me/train \
    --output_dir ./custom_output \
    --chunksize 20000
```

### 参数说明
- `--data_dir`: CSV 数据文件所在目录
- `--output_dir`: 输出文件保存目录（默认：`./output`）
- `--chunksize`: 分块读取大小（默认：10000，设为 0 表示一次性读取）

## 📄 输出文件

### 1. `ehr_dataset_full.json` - 完整数据集
包含完整的元数据和所有患者记录的结构化 JSON 文件：

```json
{
  "description": {
    "dataset_name": "EHR Dataset Field Map",
    "purpose": "Provide structured per-patient data for clinical summarization and hallucination detection tasks.",
    "structure": "Each record contains universal keys shared across all files and unique fields from six EHR data sources.",
    "generated_at": "2025-10-18 19:59:46",
    "version": "1.0"
  },
  "inverse_map": {
    "subject_id": ["diagnosis", "discharge", "edstays", "radiology", "triage"],
    "hadm_id": ["discharge", "discharge_target", "edstays", "radiology"],
    ...
  },
  "metadata": {
    "subject_id": "identifier",
    "hadm_id": "identifier",
    "temperature": "vital_sign",
    "text": "clinical_text",
    ...
  },
  "patients": [
    {
      "name": "Patient 1",
      "universal": {
        "subject_id": 10000032,
        "hadm_id": 22841357,
        "stay_id": 38112554,
        "note_id": "10000032-DS-22",
        "note_type": "DS",
        "charttime": "2180-06-27 00:00:00",
        "storetime": "2180-07-01 10:15:00"
      },
      "diagnosis": {
        "seq_num": 1,
        "icd_code": "78959",
        "icd_version": 9,
        "icd_title": "OTHER ASCITES"
      },
      "discharge": {
        "note_type": "DS",
        "note_seq": 22,
        "text": "完整的出院记录文本..."
      },
      "edstays": {
        "intime": "2180-06-26 15:54:00",
        "outtime": "2180-06-26 21:31:00",
        "gender": "F",
        "race": "WHITE",
        "arrival_transport": "AMBULANCE",
        "disposition": "ADMITTED"
      },
      "radiology": {
        "text": "影像检查报告文本..."
      },
      "triage": {
        "temperature": 98.9,
        "heartrate": 88.0,
        "resprate": 18.0,
        "o2sat": 97.0,
        "sbp": 116.0,
        "dbp": 88.0,
        "pain": 10,
        "acuity": 3.0,
        "chiefcomplaint": "Abdominal distention"
      }
    }
  ]
}
```

### 2. `ehr_patients.jsonl` - 流式格式
每行一个患者 JSON 对象，适合流式处理和模型训练：

```jsonl
{"name": "Patient 1", "universal": {...}, "diagnosis": {...}, ...}
{"name": "Patient 2", "universal": {...}, "diagnosis": {...}, ...}
{"name": "Patient 3", "universal": {...}, "diagnosis": {...}, ...}
```

### 3. `processing_report.txt` - 处理报告
详细的数据处理统计信息：

```
EHR 数据处理报告
==================================================

处理时间: 2025-10-18 20:00:08
患者总数: 46998

各表数据覆盖情况:
------------------------------
diagnosis: 46998/46998 (100.0%)
discharge: 46998/46998 (100.0%)
edstays: 46998/46998 (100.0%)
radiology: 46998/46998 (100.0%)
triage: 46998/46998 (100.0%)
```

### 4. `patient_summary_stats.csv` - 患者统计摘要
可用于快速分析的 CSV 格式统计文件，包含每个患者的基本信息和数据完整性指标。

## 🔍 数据验证

使用内置的验证工具检查数据质量：

```bash
python validate_ehr_data.py
```

验证功能包括：
- ✅ JSON 格式验证
- ✅ 数据结构完整性检查
- ✅ 字段覆盖率分析
- ✅ 数据质量问题检测
- ✅ 样本患者展示

## 🚀 性能优化特性

### 内存管理
- **分块读取**: 支持大文件的分块处理，防止内存溢出
- **索引优化**: 预构建患者索引，实现 O(1) 查找效率
- **增量处理**: 逐患者构建 JSON，降低内存峰值

### 处理效率
- **并行友好**: 架构支持未来的多进程并行化
- **缓存机制**: 智能的数据缓存减少重复计算
- **流式输出**: 支持大规模数据的流式处理

### 扩展性
- **模块化设计**: 清晰的类结构便于功能扩展
- **配置化**: 表结构和字段映射完全可配置
- **多格式支持**: 可轻松扩展支持其他输出格式

## 📊 数据统计（示例运行结果）

基于 discharge-me 训练数据集的处理结果：

- **总患者数**: 46,998 人
- **数据文件**: 6 个（总计 ~1.4 GB）
- **处理时间**: ~2 分钟
- **输出大小**: 
  - `ehr_dataset_full.json`: 646.3 MB
  - `ehr_patients.jsonl`: 628.6 MB
- **数据覆盖率**: 
  - diagnosis: 100%
  - discharge: 100%
  - edstays: 100%
  - radiology: 100%
  - triage: 100%
  - discharge_target: 0% (表结构不匹配)

## 🛠️ 核心技术架构

### 数据处理流程
```
1. 文件读取 → 2. 数据清洗 → 3. 患者索引构建 → 4. JSON 拼装 → 5. 输出生成
```

### 关键算法
1. **"先分表聚合 → 再按 ID 拼装"** 的设计理念
2. **基于 subject_id 的 groupby 聚合**，避免复杂的多表 join
3. **O(1) 索引查找**，提高患者数据检索效率
4. **增量 JSON 构建**，逐患者处理降低内存使用

### 代码结构
```
EHRDataProcessor/
├── __init__()              # 初始化配置
├── read_csv_file()         # 文件读取
├── load_all_data()         # 批量数据加载
├── clean_dataframe()       # 数据清洗
├── build_patient_index()   # 索引构建
├── build_patient_json()    # 单患者 JSON 构建
├── generate_metadata()     # 元数据生成
├── process_data()          # 主处理流程
└── save_results()          # 结果保存
```

## 💡 使用建议

### 模型训练场景
1. **临床摘要生成**: 使用 `ehr_patients.jsonl` 进行流式训练
2. **幻觉检测**: 利用结构化的 `universal` 字段进行事实验证
3. **多模态学习**: 结合文本（`text` 字段）和结构化数据（生命体征等）

### 数据分析场景
1. **统计分析**: 使用 `patient_summary_stats.csv` 进行快速分析
2. **数据探索**: 使用完整的 JSON 文件进行深度挖掘
3. **质量监控**: 定期运行验证工具确保数据质量

### 生产部署
1. **批处理**: 适合大规模离线数据处理
2. **增量更新**: 支持新数据的增量处理
3. **监控告警**: 集成处理报告进行状态监控

## 🔧 自定义扩展

### 添加新的数据表
1. 在 `file_config` 中添加新表配置
2. 定义 `key_fields` 和 `unique_fields`
3. 更新 `universal_fields` 集合（如需要）

### 自定义字段映射
1. 修改 `metadata` 字典添加新字段类型
2. 在 `clean_dataframe()` 中添加特定的数据处理逻辑
3. 更新 `extract_table_specific_fields()` 方法

### 输出格式扩展
1. 在 `save_results()` 中添加新的导出逻辑
2. 支持 Parquet、Avro 等其他格式
3. 集成云存储 API 进行直接上传

## 📞 支持和维护

### 常见问题
1. **内存不足**: 减小 `chunksize` 参数
2. **处理时间长**: 增大 `chunksize` 或使用 SSD 存储
3. **数据缺失**: 检查原始 CSV 文件的字段名和格式

### 日志和调试
- 所有处理步骤都有详细的日志输出
- 使用 `logging` 模块可调整日志级别
- 异常处理确保错误信息的完整性

### 版本历史
- **v1.0** (2025-10-18): 初始版本，支持 6 表拼接和双格式输出

---

## 🎉 总结

这个 EHR 数据处理工具实现了高效、可扩展的多表数据整合方案，完美满足了临床 AI 模型训练的数据需求。通过结构化的 JSON 输出和完善的元数据支持，为后续的模型开发提供了坚实的数据基础。

**立即开始使用**：
```bash
python run_ehr_processor.py
```