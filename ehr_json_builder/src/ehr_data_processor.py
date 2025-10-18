#!/usr/bin/env python3
"""
EHR 多表拼接 → 单患者 JSON 构建工具

功能：
1. 从六个 CSV 文件读取 EHR 数据
2. 按 subject_id 聚合患者数据
3. 生成结构化的患者 JSON 文件
4. 支持大规模数据的内存友好处理

"""

import pandas as pd
import json
import gzip
import os
from typing import Dict, List, Any, Optional
from collections import defaultdict
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EHRDataProcessor:
    """EHR 数据处理器"""
    
    def __init__(self, data_dir: str, output_dir: str = "./output"):
        """
        初始化数据处理器
        
        Args:
            data_dir: 数据文件目录路径
            output_dir: 输出文件目录路径
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 定义文件映射（根据实际数据结构）
        self.file_config = {
            'diagnosis': {
                'filename': 'diagnosis.csv',
                'key_fields': ['subject_id', 'stay_id'],
                'unique_fields': ['seq_num', 'icd_code', 'icd_version', 'icd_title']
            },
            'discharge': {  # 实际是 discharge_target 的数据结构
                'filename': 'discharge.csv', 
                'key_fields': ['note_id', 'subject_id', 'hadm_id'],
                'unique_fields': ['note_type', 'note_seq', 'charttime', 'storetime', 'text']
            },
            'discharge_target': {  # 实际是 discharge 的数据结构
                'filename': 'discharge_target.csv',
                'key_fields': ['note_id', 'hadm_id'],
                'unique_fields': ['discharge_instructions', 'brief_hospital_course', 
                                'discharge_instructions_word_count', 'brief_hospital_course_word_count']
            },
            'edstays': {
                'filename': 'edstays.csv',
                'key_fields': ['subject_id', 'hadm_id', 'stay_id'],
                'unique_fields': ['intime', 'outtime', 'gender', 'race', 'arrival_transport', 'disposition']
            },
            'radiology': {
                'filename': 'radiology.csv',
                'key_fields': ['note_id', 'subject_id', 'hadm_id'],
                'unique_fields': ['note_type', 'note_seq', 'charttime', 'storetime', 'text']
            },
            'triage': {
                'filename': 'triage.csv',
                'key_fields': ['subject_id', 'stay_id'],
                'unique_fields': ['temperature', 'heartrate', 'resprate', 'o2sat', 
                                'sbp', 'dbp', 'pain', 'acuity', 'chiefcomplaint']
            }
        }
        
        # 定义通用字段（跨表共享）
        self.universal_fields = {
            'subject_id', 'hadm_id', 'stay_id', 'note_id', 
            'note_type', 'charttime', 'storetime'
        }
        
    def read_csv_file(self, filename: str, chunksize: Optional[int] = 10000) -> pd.DataFrame:
        """
        读取CSV文件（支持gzip压缩）
        
        Args:
            filename: 文件名
            chunksize: 分块大小，None表示一次性读取
            
        Returns:
            DataFrame
        """
        filepath = os.path.join(self.data_dir, filename)
        
        # 检查是否为gzip文件
        if filename.endswith('.gz'):
            compression = 'gzip'
        else:
            compression = None
            
        try:
            if chunksize is None:
                # 一次性读取
                df = pd.read_csv(filepath, compression=compression)
            else:
                # 分块读取并合并
                chunks = []
                for chunk in pd.read_csv(filepath, compression=compression, chunksize=chunksize):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
                
            logger.info(f"成功读取文件 {filename}, 行数: {len(df)}")
            return df
            
        except Exception as e:
            logger.error(f"读取文件 {filename} 失败: {e}")
            raise
    
    def load_all_data(self, chunksize: Optional[int] = 10000) -> Dict[str, pd.DataFrame]:
        """
        加载所有数据文件
        
        Args:
            chunksize: 分块大小，None表示一次性读取
            
        Returns:
            包含所有表的字典
        """
        data = {}
        
        for table_name, config in self.file_config.items():
            logger.info(f"正在加载 {table_name} 数据...")
            
            # 尝试多种文件扩展名
            possible_files = [
                config['filename'],
                config['filename'] + '.gz',
                config['filename'].replace('.csv', '.csv.gz')
            ]
            
            df = None
            for filename in possible_files:
                filepath = os.path.join(self.data_dir, filename)
                if os.path.exists(filepath):
                    df = self.read_csv_file(filename, chunksize)
                    break
                    
            if df is None:
                logger.warning(f"找不到 {table_name} 的数据文件，跳过...")
                continue
                
            # 数据清洗：处理空值和数据类型
            df = self.clean_dataframe(df, table_name)
            data[table_name] = df
            
        return data
    
    def clean_dataframe(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """
        清洗数据框
        
        Args:
            df: 原始数据框
            table_name: 表名
            
        Returns:
            清洗后的数据框
        """
        # 处理数值列
        numeric_columns = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 
                          'pain', 'acuity', 'seq_num', 'icd_version', 'note_seq',
                          'discharge_instructions_word_count', 'brief_hospital_course_word_count']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 处理时间列
        time_columns = ['charttime', 'storetime', 'intime', 'outtime']
        for col in time_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # 处理文本列：去除首尾空格
        text_columns = ['text', 'discharge_instructions', 'brief_hospital_course', 
                       'chiefcomplaint', 'icd_title']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        logger.info(f"{table_name} 数据清洗完成")
        return df
    
    def build_patient_index(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        构建患者索引以提高查询效率
        
        Args:
            data: 所有表的数据
            
        Returns:
            按subject_id分组的索引
        """
        logger.info("正在构建患者索引...")
        
        grouped_data = {}
        
        for table_name, df in data.items():
            if 'subject_id' in df.columns:
                # 按subject_id分组，转换为字典格式
                grouped = df.groupby('subject_id').apply(
                    lambda x: x.to_dict(orient='records')
                ).to_dict()
                grouped_data[table_name] = grouped
            else:
                logger.warning(f"表 {table_name} 没有 subject_id 字段，跳过分组")
                grouped_data[table_name] = {}
        
        # 获取所有患者ID
        all_subject_ids = set()
        for table_name, grouped in grouped_data.items():
            all_subject_ids.update(grouped.keys())
        
        logger.info(f"索引构建完成，共找到 {len(all_subject_ids)} 个患者")
        
        return grouped_data, sorted(all_subject_ids)
    
    def extract_universal_fields(self, patient_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        提取患者的通用字段
        
        Args:
            patient_data: 单个患者在所有表中的数据
            
        Returns:
            通用字段字典
        """
        universal = {}
        
        # 从edstays表开始（通常包含最完整的ID信息）
        if 'edstays' in patient_data and patient_data['edstays']:
            ed_record = patient_data['edstays'][0]
            for field in self.universal_fields:
                if field in ed_record:
                    universal[field] = ed_record[field]
        
        # 从其他表补充缺失的字段
        for table_name, records in patient_data.items():
            if records:
                record = records[0]
                for field in self.universal_fields:
                    if field not in universal and field in record:
                        universal[field] = record[field]
        
        # 格式化时间字段
        for field in ['charttime', 'storetime', 'intime', 'outtime']:
            if field in universal and pd.notna(universal[field]):
                if isinstance(universal[field], pd.Timestamp):
                    universal[field] = universal[field].strftime('%Y-%m-%d %H:%M:%S')
        
        return universal
    
    def extract_table_specific_fields(self, records: List[Dict], table_name: str) -> Dict[str, Any]:
        """
        提取表特定字段
        
        Args:
            records: 表中的记录列表
            table_name: 表名
            
        Returns:
            表特定字段字典
        """
        if not records:
            return {}
        
        # 取第一条记录（可以根据需要调整逻辑）
        record = records[0]
        specific_fields = {}
        
        # 获取该表的独有字段
        unique_fields = self.file_config[table_name]['unique_fields']
        
        for field in unique_fields:
            if field in record:
                value = record[field]
                
                # 处理时间格式
                if isinstance(value, pd.Timestamp):
                    value = value.strftime('%Y-%m-%d %H:%M:%S')
                # 处理NaN值
                elif pd.isna(value):
                    value = None
                # 处理数值类型
                elif isinstance(value, (int, float)) and pd.notna(value):
                    # 保持数值类型
                    pass
                else:
                    # 转换为字符串并处理
                    value = str(value) if value is not None else None
                
                specific_fields[field] = value
        
        return specific_fields
    
    def build_patient_json(self, patient_id: str, patient_data: Dict[str, List[Dict]], 
                          patient_index: int) -> Dict[str, Any]:
        """
        构建单个患者的JSON记录
        
        Args:
            patient_id: 患者ID
            patient_data: 患者在所有表中的数据
            patient_index: 患者序号
            
        Returns:
            患者JSON记录
        """
        # 提取通用字段
        universal = self.extract_universal_fields(patient_data)
        
        # 构建患者记录
        patient_record = {
            "name": f"Patient {patient_index}",
            "universal": universal
        }
        
        # 添加各表的特定字段
        for table_name in self.file_config.keys():
            records = patient_data.get(table_name, [])
            specific_fields = self.extract_table_specific_fields(records, table_name)
            patient_record[table_name] = specific_fields
        
        return patient_record
    
    def generate_metadata(self) -> Dict[str, Dict]:
        """
        生成元数据信息
        
        Returns:
            包含description、inverse_map、metadata的字典
        """
        description = {
            "dataset_name": "EHR Dataset Field Map",
            "purpose": "Provide structured per-patient data for clinical summarization and hallucination detection tasks.",
            "structure": "Each record contains universal keys shared across all files and unique fields from six EHR data sources (diagnosis, discharge, discharge_target, edstays, radiology, triage).",
            "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "version": "1.0"
        }
        
        # 构建字段反向映射
        inverse_map = defaultdict(list)
        for table_name, config in self.file_config.items():
            all_fields = config['key_fields'] + config['unique_fields']
            for field in all_fields:
                inverse_map[field].append(table_name)
        
        # 字段元数据
        metadata = {
            # 标识符
            "subject_id": "identifier",
            "hadm_id": "identifier", 
            "stay_id": "identifier",
            "note_id": "identifier",
            
            # 文本标签
            "note_type": "text_label",
            
            # 时间戳
            "charttime": "timestamp",
            "storetime": "timestamp",
            "intime": "timestamp", 
            "outtime": "timestamp",
            
            # 诊断相关
            "seq_num": "sequence_index",
            "icd_code": "diagnosis_code",
            "icd_version": "diagnosis_code_version", 
            "icd_title": "diagnosis_title",
            
            # 临床文本
            "discharge_instructions": "clinical_text",
            "brief_hospital_course": "clinical_text", 
            "text": "clinical_text",
            "chiefcomplaint": "clinical_text",
            
            # 数值统计
            "discharge_instructions_word_count": "numeric_statistic",
            "brief_hospital_course_word_count": "numeric_statistic", 
            "note_seq": "sequence_index",
            
            # 人口统计学
            "gender": "demographic",
            "race": "demographic",
            
            # 就诊信息
            "arrival_transport": "encounter_info",
            "disposition": "encounter_outcome",
            
            # 生命体征
            "temperature": "vital_sign",
            "heartrate": "vital_sign", 
            "resprate": "vital_sign",
            "o2sat": "vital_sign",
            "sbp": "vital_sign",
            "dbp": "vital_sign", 
            "pain": "vital_sign",
            
            # 临床评分
            "acuity": "clinical_score"
        }
        
        return {
            "description": description,
            "inverse_map": dict(inverse_map), 
            "metadata": metadata
        }
    
    def process_data(self, chunksize: Optional[int] = 10000) -> Dict[str, Any]:
        """
        处理所有数据并生成JSON结构
        
        Args:
            chunksize: 读取数据的分块大小
            
        Returns:
            完整的EHR数据集JSON
        """
        logger.info("开始数据处理流程...")
        
        # 1. 加载所有数据
        data = self.load_all_data(chunksize)
        
        if not data:
            raise ValueError("没有找到任何有效的数据文件")
        
        # 2. 构建患者索引
        grouped_data, all_subject_ids = self.build_patient_index(data)
        
        # 3. 构建患者记录
        patients = []
        logger.info("正在构建患者记录...")
        
        for i, subject_id in enumerate(all_subject_ids, start=1):
            if i % 1000 == 0:
                logger.info(f"已处理 {i}/{len(all_subject_ids)} 个患者")
            
            # 收集该患者在所有表中的数据
            patient_data = {}
            for table_name in self.file_config.keys():
                patient_data[table_name] = grouped_data[table_name].get(subject_id, [])
            
            # 构建患者JSON记录
            patient_record = self.build_patient_json(subject_id, patient_data, i)
            patients.append(patient_record)
        
        # 4. 生成元数据
        metadata_info = self.generate_metadata()
        
        # 5. 组装最终结果
        result = {
            **metadata_info,
            "patients": patients
        }
        
        logger.info(f"数据处理完成！共处理 {len(patients)} 个患者")
        return result
    
    def save_results(self, data: Dict[str, Any]):
        """
        保存处理结果
        
        Args:
            data: 处理后的数据
        """
        logger.info("正在保存结果...")
        
        # 保存完整JSON文件
        full_json_path = os.path.join(self.output_dir, "ehr_dataset_full.json")
        with open(full_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"完整JSON文件已保存: {full_json_path}")
        
        # 保存JSONL文件（每行一个患者）
        jsonl_path = os.path.join(self.output_dir, "ehr_patients.jsonl")
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for patient in data['patients']:
                json.dump(patient, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"JSONL文件已保存: {jsonl_path}")
        
        # 生成统计报告
        self.generate_report(data)
    
    def generate_report(self, data: Dict[str, Any]):
        """
        生成处理报告
        
        Args:
            data: 处理后的数据
        """
        report_path = os.path.join(self.output_dir, "processing_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("EHR 数据处理报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"患者总数: {len(data['patients'])}\n\n")
            
            # 统计各表的覆盖率
            f.write("各表数据覆盖情况:\n")
            f.write("-" * 30 + "\n")
            
            for table_name in self.file_config.keys():
                count = sum(1 for p in data['patients'] if p[table_name])
                coverage = count / len(data['patients']) * 100
                f.write(f"{table_name}: {count}/{len(data['patients'])} ({coverage:.1f}%)\n")
            
            f.write("\n字段统计:\n")
            f.write("-" * 30 + "\n")
            for field, field_type in data['metadata'].items():
                f.write(f"{field}: {field_type}\n")
        
        logger.info(f"处理报告已保存: {report_path}")
    
    def run(self, chunksize: Optional[int] = 10000):
        """
        运行完整的数据处理流程
        
        Args:
            chunksize: 读取数据的分块大小
        """
        try:
            # 处理数据
            result = self.process_data(chunksize)
            
            # 保存结果
            self.save_results(result)
            
            logger.info("数据处理流程完成！")
            
        except Exception as e:
            logger.error(f"数据处理失败: {e}")
            raise


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='EHR 多表拼接数据处理工具')
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='数据文件目录路径')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='输出文件目录路径')
    parser.add_argument('--chunksize', type=int, default=10000,
                       help='分块读取大小（设为0表示一次性读取）')
    
    args = parser.parse_args()
    
    # 处理chunksize参数
    chunksize = args.chunksize if args.chunksize > 0 else None
    
    # 创建处理器并运行
    processor = EHRDataProcessor(args.data_dir, args.output_dir)
    processor.run(chunksize)


if __name__ == "__main__":
    main()