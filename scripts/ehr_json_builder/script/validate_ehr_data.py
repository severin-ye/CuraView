#!/usr/bin/env python3
"""
EHR 数据验证和查看工具

功能：
1. 验证生成的 JSON 文件的完整性
2. 查看数据统计和示例
3. 执行基本的数据质量检查
"""

import json
import os
from typing import Dict, List, Any
import pandas as pd

def load_and_validate_json(filepath: str) -> Dict[str, Any]:
    """加载并验证 JSON 文件"""
    print(f"📄 加载文件: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath}")
        return None
    
    file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
    print(f"📊 文件大小: {file_size:.1f} MB")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("✅ JSON 格式验证通过")
        return data
    except json.JSONDecodeError as e:
        print(f"❌ JSON 格式错误: {e}")
        return None
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None

def validate_data_structure(data: Dict[str, Any]) -> bool:
    """验证数据结构"""
    print("\n🔍 验证数据结构...")
    
    required_keys = ['description', 'inverse_map', 'metadata', 'patients']
    
    for key in required_keys:
        if key not in data:
            print(f"❌ 缺少必需的键: {key}")
            return False
        print(f"✅ 包含键: {key}")
    
    # 验证 patients 是列表
    if not isinstance(data['patients'], list):
        print("❌ patients 应该是列表")
        return False
    
    print(f"✅ 患者记录数量: {len(data['patients'])}")
    
    # 验证第一个患者的结构
    if data['patients']:
        patient = data['patients'][0]
        expected_patient_keys = ['name', 'universal', 'diagnosis', 'discharge', 
                               'discharge_target', 'edstays', 'radiology', 'triage']
        
        for key in expected_patient_keys:
            if key not in patient:
                print(f"❌ 患者记录缺少键: {key}")
                return False
        print("✅ 患者记录结构正确")
    
    return True

def analyze_data_coverage(data: Dict[str, Any]) -> Dict[str, Any]:
    """分析数据覆盖情况"""
    print("\n📊 分析数据覆盖情况...")
    
    patients = data['patients']
    total_patients = len(patients)
    
    coverage_stats = {}
    
    # 分析各表的数据覆盖率
    tables = ['diagnosis', 'discharge', 'discharge_target', 'edstays', 'radiology', 'triage']
    
    for table in tables:
        count = sum(1 for p in patients if p[table])
        coverage = count / total_patients * 100 if total_patients > 0 else 0
        coverage_stats[table] = {
            'count': count,
            'total': total_patients,
            'coverage': coverage
        }
        print(f"   {table}: {count}/{total_patients} ({coverage:.1f}%)")
    
    # 分析通用字段的覆盖率
    print("\n🔑 通用字段覆盖情况:")
    universal_fields = ['subject_id', 'hadm_id', 'stay_id', 'note_id', 'note_type', 'charttime', 'storetime']
    
    for field in universal_fields:
        count = sum(1 for p in patients if field in p['universal'] and p['universal'][field] is not None)
        coverage = count / total_patients * 100 if total_patients > 0 else 0
        print(f"   {field}: {count}/{total_patients} ({coverage:.1f}%)")
    
    return coverage_stats

def show_sample_patients(data: Dict[str, Any], num_samples: int = 3):
    """显示样本患者"""
    print(f"\n👥 显示前 {num_samples} 个患者样本:")
    
    patients = data['patients']
    
    for i in range(min(num_samples, len(patients))):
        patient = patients[i]
        print(f"\n--- {patient['name']} ---")
        print(f"Subject ID: {patient['universal'].get('subject_id', 'N/A')}")
        print(f"Gender: {patient['edstays'].get('gender', 'N/A')}")
        print(f"Race: {patient['edstays'].get('race', 'N/A')}")
        
        # 显示诊断信息
        if patient['diagnosis']:
            print(f"诊断: {patient['diagnosis'].get('icd_title', 'N/A')}")
        
        # 显示生命体征
        if patient['triage']:
            triage = patient['triage']
            print(f"生命体征: 体温{triage.get('temperature', 'N/A')}°F, " +
                  f"心率{triage.get('heartrate', 'N/A')}, " +
                  f"血压{triage.get('sbp', 'N/A')}/{triage.get('dbp', 'N/A')}")
        
        # 显示主诉
        if patient['triage'] and 'chiefcomplaint' in patient['triage']:
            print(f"主诉: {patient['triage']['chiefcomplaint']}")

def check_data_quality(data: Dict[str, Any]) -> Dict[str, Any]:
    """检查数据质量"""
    print("\n🔬 数据质量检查...")
    
    patients = data['patients']
    total_patients = len(patients)
    
    quality_issues = {
        'missing_subject_id': 0,
        'missing_basic_info': 0,
        'empty_records': 0,
        'duplicate_names': 0
    }
    
    seen_names = set()
    
    for patient in patients:
        # 检查缺失的 subject_id
        if not patient['universal'].get('subject_id'):
            quality_issues['missing_subject_id'] += 1
        
        # 检查缺失的基本信息
        if not patient['edstays'].get('gender') or not patient['edstays'].get('race'):
            quality_issues['missing_basic_info'] += 1
        
        # 检查空记录
        has_data = any(patient[table] for table in ['diagnosis', 'discharge', 'edstays', 'radiology', 'triage'])
        if not has_data:
            quality_issues['empty_records'] += 1
        
        # 检查重复的患者名称
        name = patient['name']
        if name in seen_names:
            quality_issues['duplicate_names'] += 1
        seen_names.add(name)
    
    # 报告质量问题
    for issue, count in quality_issues.items():
        if count > 0:
            percentage = count / total_patients * 100
            print(f"⚠️  {issue}: {count} ({percentage:.1f}%)")
        else:
            print(f"✅ {issue}: 无问题")
    
    return quality_issues

def export_summary_stats(data: Dict[str, Any], output_path: str):
    """导出汇总统计"""
    print(f"\n📊 导出统计摘要到: {output_path}")
    
    patients = data['patients']
    
    # 创建统计数据框
    stats_data = []
    
    for patient in patients:
        row = {
            'patient_name': patient['name'],
            'subject_id': patient['universal'].get('subject_id'),
            'gender': patient['edstays'].get('gender'),
            'race': patient['edstays'].get('race'),
            'has_diagnosis': bool(patient['diagnosis']),
            'has_discharge': bool(patient['discharge']),
            'has_radiology': bool(patient['radiology']),
            'has_triage': bool(patient['triage']),
            'chief_complaint': patient['triage'].get('chiefcomplaint', ''),
            'icd_code': patient['diagnosis'].get('icd_code', ''),
            'icd_title': patient['diagnosis'].get('icd_title', '')
        }
        stats_data.append(row)
    
    # 保存为 CSV
    df = pd.DataFrame(stats_data)
    df.to_csv(output_path, index=False)
    print(f"✅ 已保存 {len(df)} 条患者统计记录")

def main(output_dir=None):
    """主函数"""
    import argparse
    
    # 如果没有传入 output_dir，则使用命令行参数
    if output_dir is None:
        parser = argparse.ArgumentParser(description='EHR 数据验证工具')
        parser.add_argument('--output_dir', type=str, 
                           default='./output',
                           help='输出文件目录路径')
        args = parser.parse_args()
        output_dir = args.output_dir
    
    print("=" * 60)
    print("🔍 EHR 数据验证和查看工具")
    print("=" * 60)
    
    json_file = os.path.join(output_dir, "ehr_dataset_full.json")
    
    # 检查文件是否存在
    if not os.path.exists(json_file):
        print(f"❌ 错误: 找不到数据文件 - {json_file}")
        print("💡 请确认数据处理已完成，且输出目录正确")
        return 1
    
    # 1. 加载和验证 JSON 文件
    data = load_and_validate_json(json_file)
    if not data:
        return 1
    
    # 2. 验证数据结构
    if not validate_data_structure(data):
        print("❌ 数据结构验证失败")
        return 1
    
    # 3. 分析数据覆盖情况
    coverage_stats = analyze_data_coverage(data)
    
    # 4. 显示样本患者
    show_sample_patients(data, num_samples=3)
    
    # 5. 检查数据质量
    quality_issues = check_data_quality(data)
    
    # 6. 导出统计摘要
    summary_path = os.path.join(output_dir, "patient_summary_stats.csv")
    export_summary_stats(data, summary_path)
    
    print("\n🎉 数据验证完成!")
    print(f"\n📋 快速统计:")
    print(f"   总患者数: {len(data['patients'])}")
    print(f"   数据字段数: {len(data['metadata'])}")
    print(f"   支持的表: {len(coverage_stats)}")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)