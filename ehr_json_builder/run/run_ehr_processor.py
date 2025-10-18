#!/usr/bin/env python3
"""
EHR 数据处理示例脚本

使用示例：
python run_ehr_processor.py

这个脚本将处理 discharge-me/train 目录中的数据并生成JSON文件
"""

import os
import sys
from ehr_json_builder.src.ehr_data_processor import EHRDataProcessor

def main():
    """运行 EHR 数据处理示例"""
    
    # 设置数据目录和输出目录
    data_dir = "/home/work/hd/discharge-me/train"
    output_dir = "/home/work/hd/output"
    
    print("=" * 60)
    print("🏥 EHR 多表拼接 → 单患者 JSON 构建工具")
    print("=" * 60)
    print()
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"❌ 错误：数据目录不存在: {data_dir}")
        return 1
    
    print(f"📁 数据目录: {data_dir}")
    print(f"📄 输出目录: {output_dir}")
    print()
    
    # 检查数据文件
    print("📋 检查数据文件...")
    files_found = []
    expected_files = [
        'diagnosis.csv', 'discharge.csv', 'discharge_target.csv',
        'edstays.csv', 'radiology.csv', 'triage.csv'
    ]
    
    for filename in expected_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            files_found.append(filename)
            print(f"   ✅ {filename} ({file_size:.1f} MB)")
        else:
            print(f"   ❌ {filename} (未找到)")
    
    if not files_found:
        print("❌ 没有找到任何数据文件！")
        return 1
    
    print(f"\n📊 找到 {len(files_found)} 个数据文件")
    print()
    
    try:
        # 创建数据处理器
        print("🚀 初始化数据处理器...")
        processor = EHRDataProcessor(data_dir, output_dir)
        
        # 运行处理流程（使用较大的chunksize以提高性能）
        print("🔄 开始数据处理...")
        print("   注意：大文件处理可能需要几分钟时间...")
        print()
        
        processor.run(chunksize=50000)  # 使用较大的块大小
        
        print()
        print("🎉 数据处理完成！")
        print()
        print("📁 输出文件:")
        
        # 检查输出文件
        output_files = [
            "ehr_dataset_full.json",
            "ehr_patients.jsonl", 
            "processing_report.txt"
        ]
        
        for filename in output_files:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                print(f"   📄 {filename} ({file_size:.1f} MB)")
            else:
                print(f"   ❌ {filename} (生成失败)")
        
        print()
        print("💡 使用建议:")
        print("   - ehr_dataset_full.json: 包含完整元数据的结构化数据集")
        print("   - ehr_patients.jsonl: 每行一个患者，适合流式处理和训练")
        print("   - processing_report.txt: 数据处理统计报告")
        print()
        
        return 0
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        print("\n详细错误信息:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)