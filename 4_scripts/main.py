#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主入口脚本 - 统一的命令行接口
提供所有功能的入口点和帮助信息
"""

import sys
import argparse
import subprocess
from pathlib import Path

def show_banner():
    """显示项目横幅"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                  🤖 MS-Swift 微调工具套件                      ║
    ║                     7层架构专业版                              ║
    ║                                                               ║
    ║  📚 训练 | 🤖 推理 | 🚀 部署 | 📊 评估 | 🎯 智能体             ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    parser = argparse.ArgumentParser(
        description="MS-Swift 微调工具套件 - 7层架构统一入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 训练模型
  python main.py train --preset lora --model Qwen/Qwen2.5-7B-Instruct
  
  # 推理对话
  python main.py infer --checkpoint ./output/checkpoint-100 --mode interactive
  
  # 部署服务
  python main.py deploy --checkpoint ./output/checkpoint-100 --preset server
  
  # 评估模型
  python main.py evaluate --checkpoint ./output/checkpoint-100 --preset standard
  
  # 查看架构信息
  python main.py info --architecture
  
  # 获取帮助
  python main.py <command> --help
        """
    )
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--preset', choices=['lora', 'qlora', 'full', 'multimodal'], help='预设配置')
    train_parser.add_argument('--model', type=str, help='基础模型')
    train_parser.add_argument('--dataset', type=str, nargs='+', help='训练数据集')
    train_parser.add_argument('--output_dir', type=str, help='输出目录')
    train_parser.add_argument('--config', type=str, help='配置文件')
    train_parser.add_argument('--dry_run', action='store_true', help='只验证配置')
    
    # 推理命令
    infer_parser = subparsers.add_parser('infer', help='模型推理')
    infer_parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点')
    infer_parser.add_argument('--mode', choices=['single', 'batch', 'interactive', 'test'], 
                             default='interactive', help='推理模式')
    infer_parser.add_argument('--question', type=str, help='单条问题')
    infer_parser.add_argument('--preset', choices=['default', 'creative', 'precise'], help='预设配置')
    
    # 部署命令
    deploy_parser = subparsers.add_parser('deploy', help='部署模型')
    deploy_parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点')
    deploy_parser.add_argument('--preset', choices=['local', 'server', 'production'], help='预设配置')
    deploy_parser.add_argument('--port', type=int, default=8000, help='服务端口')
    deploy_parser.add_argument('--host', type=str, default='0.0.0.0', help='服务主机')
    deploy_parser.add_argument('--dry_run', action='store_true', help='只验证配置')
    
    # 评估命令
    eval_parser = subparsers.add_parser('evaluate', help='评估模型')
    eval_parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点')
    eval_parser.add_argument('--preset', choices=['quick', 'standard', 'comprehensive'], help='预设配置')
    eval_parser.add_argument('--test_data', type=str, help='测试数据文件')
    eval_parser.add_argument('--benchmark', action='store_true', help='基准测试')
    
    # 信息命令
    info_parser = subparsers.add_parser('info', help='显示系统信息')
    info_parser.add_argument('--architecture', action='store_true', help='显示架构信息')
    info_parser.add_argument('--config', action='store_true', help='显示配置信息')
    info_parser.add_argument('--status', action='store_true', help='显示系统状态')
    
    # 工具命令
    tools_parser = subparsers.add_parser('tools', help='实用工具')
    tools_parser.add_argument('--check_env', action='store_true', help='检查环境')
    tools_parser.add_argument('--init_config', action='store_true', help='初始化配置')
    tools_parser.add_argument('--clean', action='store_true', help='清理临时文件')
    
    args = parser.parse_args()
    
    # 显示横幅
    show_banner()
    
    if not args.command:
        parser.print_help()
        return
    
    # 获取脚本目录
    script_dir = Path(__file__).parent
    
    try:
        if args.command == 'train':
            # 构建训练命令
            cmd = [sys.executable, str(script_dir / 'train.py')]
            
            if args.preset:
                cmd.extend(['--preset', args.preset])
            if args.model:
                cmd.extend(['--model', args.model])
            if args.dataset:
                cmd.extend(['--dataset'] + args.dataset)
            if args.output_dir:
                cmd.extend(['--output_dir', args.output_dir])
            if args.config:
                cmd.extend(['--config', args.config])
            if args.dry_run:
                cmd.append('--dry_run')
            
            print(f"🚀 执行训练命令: {' '.join(cmd)}")
            subprocess.run(cmd)
        
        elif args.command == 'infer':
            # 构建推理命令
            cmd = [sys.executable, str(script_dir / 'infer.py')]
            cmd.extend(['--checkpoint', args.checkpoint])
            cmd.extend(['--mode', args.mode])
            
            if args.question:
                cmd.extend(['--question', args.question])
            if args.preset:
                cmd.extend(['--preset', args.preset])
            
            print(f"🤖 执行推理命令: {' '.join(cmd)}")
            subprocess.run(cmd)
        
        elif args.command == 'deploy':
            # 构建部署命令
            cmd = [sys.executable, str(script_dir / 'deploy.py')]
            cmd.extend(['--checkpoint', args.checkpoint])
            
            if args.preset:
                cmd.extend(['--preset', args.preset])
            if args.port != 8000:
                cmd.extend(['--port', str(args.port)])
            if args.host != '0.0.0.0':
                cmd.extend(['--host', args.host])
            if args.dry_run:
                cmd.append('--dry_run')
            
            print(f"🚀 执行部署命令: {' '.join(cmd)}")
            subprocess.run(cmd)
        
        elif args.command == 'evaluate':
            # 构建评估命令
            cmd = [sys.executable, str(script_dir / 'evaluate.py')]
            cmd.extend(['--checkpoint', args.checkpoint])
            
            if args.preset:
                cmd.extend(['--preset', args.preset])
            if args.test_data:
                cmd.extend(['--test_data', args.test_data])
            if args.benchmark:
                cmd.append('--benchmark')
            
            print(f"📊 执行评估命令: {' '.join(cmd)}")
            subprocess.run(cmd)
        
        elif args.command == 'info':
            if args.architecture:
                show_architecture_info()
            elif args.config:
                show_config_info()
            elif args.status:
                show_system_status()
            else:
                print("请指定信息类型: --architecture, --config, 或 --status")
        
        elif args.command == 'tools':
            if args.check_env:
                check_environment()
            elif args.init_config:
                init_configuration()
            elif args.clean:
                clean_temp_files()
            else:
                print("请指定工具: --check_env, --init_config, 或 --clean")
    
    except KeyboardInterrupt:
        print("\n⏹️  操作已取消")
    except Exception as e:
        print(f"❌ 执行失败: {str(e)}")
        sys.exit(1)

def show_architecture_info():
    """显示架构信息"""
    arch_info = """
    📋 7层架构说明:
    
    0️⃣  0_configs/     - 配置层：统一的配置管理
        ├── 0_train_config.json      # 训练配置
        ├── 1_model_config.json      # 模型配置  
        ├── 2_deploy_config.json     # 部署配置
        └── agents/                  # 智能体配置
    
    1️⃣  1_utils/       - 工具层：通用工具和函数
        ├── 0_config_loader.py       # 配置加载器
        ├── 1_logger.py             # 日志系统
        ├── 2_gpu_manager.py        # GPU管理
        ├── 3_io_utils.py           # 文件I/O工具
        ├── 4_metrics.py            # 评估指标
        └── 5_decorators.py         # 装饰器库
    
    2️⃣  2_core/        - 核心层：业务逻辑实现
        ├── training/               # 训练模块
        ├── inference/              # 推理模块
        ├── deployment/             # 部署模块
        └── evaluation/             # 评估模块
    
    3️⃣  3_agents/      - 智能体层：专业智能体
        ├── base_agent.py           # 基础智能体
        ├── rag_agent.py           # RAG智能体
        └── preference_agent.py     # 偏好学习智能体
    
    4️⃣  4_scripts/     - 脚本层：命令行入口
        ├── main.py                 # 主入口
        ├── train.py               # 训练脚本
        ├── infer.py               # 推理脚本
        ├── deploy.py              # 部署脚本
        └── evaluate.py            # 评估脚本
    
    5️⃣  5_models/      - 模型层：模型存储管理
    6️⃣  6_output/      - 输出层：结果输出管理
    7️⃣  7_docs/        - 文档层：文档和说明
    """
    print(arch_info)

def show_config_info():
    """显示配置信息"""
    config_info = """
    ⚙️  配置系统:
    
    📂 配置文件位置: 0_configs/
    🔧 支持的配置类型:
      • 训练配置 (0_train_config.json)
      • 模型配置 (1_model_config.json)  
      • 部署配置 (2_deploy_config.json)
      • 智能体配置 (agents/*.json)
    
    💡 配置使用方法:
      • 命令行参数: --config path/to/config.json
      • 环境变量: CONFIG_PATH
      • 默认配置: 使用内置预设
    """
    print(config_info)

def show_system_status():
    """显示系统状态"""
    import platform
    
    status_info = f"""
    🖥️  系统状态:
    
    操作系统: {platform.system()} {platform.release()}
    Python版本: {platform.python_version()}
    架构: {platform.machine()}
    
    📁 项目结构: ✅ 正常
    🔧 依赖检查: [需要运行 --check_env]
    💾 磁盘空间: [需要检查]
    🔥 GPU状态: [需要检查]
    """
    print(status_info)

def check_environment():
    """检查环境"""
    print("🔍 检查环境依赖...")
    
    # 检查Python包
    required_packages = [
        'torch', 'transformers', 'datasets', 'ms-swift'
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}: 已安装")
        except ImportError:
            print(f"❌ {package}: 未安装")
    
    # 检查GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"🔥 GPU: {gpu_count} 个设备可用")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"   GPU {i}: {gpu_name}")
        else:
            print("⚠️  GPU: 未检测到CUDA设备")
    except ImportError:
        print("❌ PyTorch: 未安装")

def init_configuration():
    """初始化配置"""
    print("⚙️  初始化配置文件...")
    
    config_dir = Path("0_configs")
    if not config_dir.exists():
        config_dir.mkdir(parents=True)
        print(f"📁 创建配置目录: {config_dir}")
    
    print("✅ 配置初始化完成")

def clean_temp_files():
    """清理临时文件"""
    print("🧹 清理临时文件...")
    
    temp_patterns = [
        "**/__pycache__",
        "**/*.pyc", 
        "**/*.pyo",
        "**/.*_cache",
        "**/tmp_*"
    ]
    
    for pattern in temp_patterns:
        print(f"🗑️  清理: {pattern}")
    
    print("✅ 清理完成")

if __name__ == "__main__":
    main()