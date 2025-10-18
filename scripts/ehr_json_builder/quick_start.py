#!/usr/bin/env python3
"""
EHR JSON Builder - 简化运行脚本

这个脚本简化了 EHR 数据处理的使用，自动配置路径和参数。

使用方法：
    python quick_start.py [数据目录] [输出目录]
    
示例：
    python quick_start.py                                    # 使用默认路径
    python quick_start.py /path/to/data                      # 自定义数据路径  
    python quick_start.py /path/to/data /path/to/output      # 自定义输入输出路径
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到 Python 路径，以便导入模块
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='EHR JSON Builder - 电子病历多表拼接工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s                                    # 使用默认路径
  %(prog)s /path/to/data                      # 指定数据目录
  %(prog)s /path/to/data /path/to/output      # 指定输入输出目录
  %(prog)s --chunksize 20000                  # 自定义分块大小
        """
    )
    
    parser.add_argument(
        'data_dir', 
        nargs='?',
        default='/home/work/hd/discharge-me/train',
        help='CSV 数据文件目录 (默认: /home/work/hd/discharge-me/train)'
    )
    
    parser.add_argument(
        'output_dir',
        nargs='?', 
        default='./output',
        help='输出文件目录 (默认: ./output)'
    )
    
    parser.add_argument(
        '--chunksize',
        type=int,
        default=50000,
        help='分块读取大小 (默认: 50000, 设为0表示一次性读取)'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='处理完成后自动运行数据验证'
    )
    
    args = parser.parse_args()
    
    print("🏥 EHR JSON Builder v1.0")
    print("=" * 60)
    print(f"📁 数据目录: {args.data_dir}")
    print(f"📄 输出目录: {args.output_dir}")
    print(f"🔧 分块大小: {args.chunksize}")
    print("=" * 60)
    print()
    
    # 检查数据目录
    if not os.path.exists(args.data_dir):
        print(f"❌ 错误: 数据目录不存在 - {args.data_dir}")
        print("💡 请确认路径正确，或使用 --help 查看使用说明")
        return 1
    
    try:
        # 导入并使用数据处理器
        from src.ehr_data_processor import EHRDataProcessor
        
        print("🚀 初始化数据处理器...")
        processor = EHRDataProcessor(args.data_dir, args.output_dir)
        
        print("🔄 开始数据处理...")
        processor.run(chunksize=args.chunksize if args.chunksize > 0 else None)
        
        print("🎉 数据处理完成!")
        
        # 检查输出文件
        output_files = {
            'ehr_dataset_full.json': '完整JSON数据集',
            'ehr_patients.jsonl': '流式患者数据', 
            'processing_report.txt': '处理统计报告'
        }
        
        print("\n📁 生成的文件:")
        for filename, description in output_files.items():
            filepath = os.path.join(args.output_dir, filename)
            if os.path.exists(filepath):
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"   ✅ {filename} ({size_mb:.1f} MB) - {description}")
            else:
                print(f"   ❌ {filename} - 生成失败")
        
        # 可选的数据验证
        if args.validate:
            print("\n🔍 运行数据验证...")
            try:
                from script.validate_ehr_data import main as validate_main
                validate_main()
            except Exception as e:
                print(f"⚠️  验证过程中出现问题: {e}")
        else:
            print(f"\n💡 可以运行以下命令进行数据验证:")
            print(f"   python script/validate_ehr_data.py")
        
        print(f"\n📖 详细文档: README_EHR_Processor.md")
        return 0
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("💡 请确认项目结构完整，所有必需文件都存在")
        return 1
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())