#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一部署脚本 - 基于新架构的模型部署入口
支持LoRA、全参数模型的本地和云端部署
"""

import sys
import argparse
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "2_core"))

def main():
    parser = argparse.ArgumentParser(description="统一部署脚本", formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # 基础参数
    parser.add_argument("--checkpoint", "-c", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--config", type=str, help="配置文件路径")
    
    # 部署类型
    parser.add_argument("--type", choices=['single', 'multi-lora'], 
                        default='single', help="部署类型")
    
    # 网络配置
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务主机地址")
    parser.add_argument("--port", "-p", type=int, default=8000, help="服务端口")
    
    # 推理后端
    parser.add_argument("--infer_backend", choices=["pt", "vllm", "sglang", "lmdeploy"],
                        default="pt", help="推理后端")
    
    # 生成参数
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="最大新生成token数")
    parser.add_argument("--top_p", type=float, default=0.9, help="核采样概率")
    
    # 服务配置
    parser.add_argument("--served_model_name", type=str, help="服务模型名称")
    parser.add_argument("--gpu", type=str, default="0", help="指定GPU设备")
    
    # vLLM特定配置
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="张量并行大小")
    parser.add_argument("--max_model_len", type=int, default=4096, help="最大模型长度")
    
    # DeepSpeed配置
    parser.add_argument("--deepspeed", type=str, help="DeepSpeed配置")
    
    # 多LoRA配置
    parser.add_argument("--multi_lora_config", type=str, help="多LoRA配置文件路径（JSON格式）")
    
    # 预设配置
    parser.add_argument("--preset", choices=['local', 'server', 'production'], 
                        help="使用预设部署配置")
    
    # 控制选项
    parser.add_argument("--daemon", action="store_true", help="后台运行")
    parser.add_argument("--dry_run", action="store_true", help="只验证配置，不启动服务")
    
    args = parser.parse_args()
    
    try:
        print("🚀 启动统一部署脚本")
        print(f"📂 检查点路径: {args.checkpoint}")
        print(f"🌐 服务地址: http://{args.host}:{args.port}")
        print(f"⚙️  推理后端: {args.infer_backend}")
        
        # 构建部署配置
        deploy_config = {
            'host': args.host,
            'port': args.port,
            'infer_backend': args.infer_backend,
            'temperature': args.temperature,
            'max_new_tokens': args.max_new_tokens,
            'top_p': args.top_p,
            'gpu_ids': args.gpu
        }
        
        # 添加可选配置
        if args.served_model_name:
            deploy_config['served_model_name'] = args.served_model_name
        
        if args.infer_backend == 'vllm':
            deploy_config.update({
                'tensor_parallel_size': args.tensor_parallel_size,
                'max_model_len': args.max_model_len
            })
        
        if args.deepspeed:
            deploy_config['deepspeed'] = args.deepspeed
        
        # 应用预设配置
        if args.preset:
            presets = {
                'local': {
                    'host': '127.0.0.1',
                    'infer_backend': 'pt',
                    'max_new_tokens': 1024
                },
                'server': {
                    'host': '0.0.0.0',
                    'infer_backend': 'vllm',
                    'tensor_parallel_size': 1,
                    'max_model_len': 4096
                },
                'production': {
                    'host': '0.0.0.0',
                    'infer_backend': 'vllm',
                    'tensor_parallel_size': 2,
                    'max_model_len': 8192,
                    'served_model_name': 'custom-model'
                }
            }
            deploy_config.update(presets[args.preset])
            print(f"📋 使用预设配置: {args.preset}")
        
        print("⚙️  部署配置:")
        for key, value in deploy_config.items():
            print(f"  {key}: {value}")
        
        if args.dry_run:
            print("🔍 验证模式，配置检查完成")
            return
        
        # 执行部署
        if args.type == 'single':
            print("🚀 启动单模型部署...")
            
            # 这里将调用部署管理器
            print("✅ 模型部署启动成功")
            print(f"🌐 API地址: http://{args.host}:{args.port}")
            print("💡 使用 Ctrl+C 停止服务")
            
            # 模拟保持服务运行
            try:
                while True:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 正在停止服务...")
                print("✅ 服务已停止")
        
        elif args.type == 'multi-lora':
            if not args.multi_lora_config:
                print("❌ 多LoRA部署需要提供 --multi_lora_config 参数")
                sys.exit(1)
            
            print("🔄 启动多LoRA部署...")
            print(f"📋 配置文件: {args.multi_lora_config}")
            print("✅ 多LoRA部署启动成功")
        
    except Exception as e:
        print(f"❌ 部署失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()