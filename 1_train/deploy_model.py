#!/usr/bin/env python3
"""
微调后模型部署脚本
支持LoRA和全参数微调后的模型部署
"""

import os
import argparse
import subprocess
from pathlib import Path


def deploy_lora_model(checkpoint_path: str, port: int = 8000, **kwargs):
    """部署LoRA微调后的模型"""
    cmd = [
        "swift", "deploy",
        "--adapters", checkpoint_path,
        "--port", str(port),
        "--host", "0.0.0.0"
    ]
    
    # 添加可选参数
    if kwargs.get("infer_backend"):
        cmd.extend(["--infer_backend", kwargs["infer_backend"]])
    if kwargs.get("temperature") is not None:
        cmd.extend(["--temperature", str(kwargs["temperature"])])
    if kwargs.get("max_new_tokens"):
        cmd.extend(["--max_new_tokens", str(kwargs["max_new_tokens"])])
    if kwargs.get("served_model_name"):
        cmd.extend(["--served_model_name", kwargs["served_model_name"]])
    
    return cmd


def deploy_full_model(checkpoint_path: str, port: int = 8000, **kwargs):
    """部署全参数微调后的模型"""
    cmd = [
        "swift", "deploy",
        "--model", checkpoint_path,
        "--port", str(port),
        "--host", "0.0.0.0"
    ]
    
    # 添加可选参数
    if kwargs.get("infer_backend"):
        cmd.extend(["--infer_backend", kwargs["infer_backend"]])
    if kwargs.get("temperature") is not None:
        cmd.extend(["--temperature", str(kwargs["temperature"])])
    if kwargs.get("max_new_tokens"):
        cmd.extend(["--max_new_tokens", str(kwargs["max_new_tokens"])])
    if kwargs.get("served_model_name"):
        cmd.extend(["--served_model_name", kwargs["served_model_name"]])
    
    return cmd


def deploy_multi_lora(lora_configs: dict, port: int = 8000, **kwargs):
    """部署多LoRA模型"""
    adapters_str = " ".join([f"{name}={path}" for name, path in lora_configs.items()])
    
    cmd = [
        "swift", "deploy",
        "--adapters", adapters_str,
        "--port", str(port),
        "--host", "0.0.0.0"
    ]
    
    # 添加可选参数
    if kwargs.get("infer_backend"):
        cmd.extend(["--infer_backend", kwargs["infer_backend"]])
    if kwargs.get("temperature") is not None:
        cmd.extend(["--temperature", str(kwargs["temperature"])])
    if kwargs.get("max_new_tokens"):
        cmd.extend(["--max_new_tokens", str(kwargs["max_new_tokens"])])
    
    return cmd


def main():
    parser = argparse.ArgumentParser(description="微调后模型部署")
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                       help="模型checkpoint路径")
    parser.add_argument("--type", "-t", choices=["lora", "full", "multi-lora"], 
                       default="auto", help="部署类型")
    parser.add_argument("--port", "-p", type=int, default=8000,
                       help="服务端口")
    parser.add_argument("--infer-backend", choices=["pt", "vllm", "sglang", "lmdeploy"],
                       default="pt", help="推理后端")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="生成温度")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                       help="最大新生成token数")
    parser.add_argument("--served-model-name", type=str,
                       help="服务模型名称")
    parser.add_argument("--gpu", type=str, default="0",
                       help="指定GPU设备")
    parser.add_argument("--multi-lora-config", type=str,
                       help="多LoRA配置文件路径（JSON格式）")
    
    args = parser.parse_args()
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # 部署参数
    deploy_kwargs = {
        "infer_backend": args.infer_backend,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "served_model_name": args.served_model_name,
    }
    
    # 检测部署类型
    if args.type == "auto":
        checkpoint_path = Path(args.checkpoint)
        adapter_files = ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]
        has_adapter = any((checkpoint_path / f).exists() for f in adapter_files)
        deploy_type = "lora" if has_adapter else "full"
    else:
        deploy_type = args.type
    
    # 生成部署命令
    if deploy_type == "lora":
        cmd = deploy_lora_model(args.checkpoint, args.port, **deploy_kwargs)
        print(f"🚀 部署LoRA模型: {args.checkpoint}")
    elif deploy_type == "full":
        cmd = deploy_full_model(args.checkpoint, args.port, **deploy_kwargs)
        print(f"🚀 部署全参数模型: {args.checkpoint}")
    elif deploy_type == "multi-lora":
        if not args.multi_lora_config:
            print("❌ 多LoRA部署需要指定--multi-lora-config参数")
            return
        
        import json
        with open(args.multi_lora_config, 'r') as f:
            lora_configs = json.load(f)
        
        cmd = deploy_multi_lora(lora_configs, args.port, **deploy_kwargs)
        print(f"🚀 部署多LoRA模型: {list(lora_configs.keys())}")
    
    # 输出部署命令
    print("📋 部署命令:")
    print(" ".join(cmd))
    print()
    
    # 执行部署
    try:
        print("🔄 启动部署服务...")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 部署失败: {e}")
    except KeyboardInterrupt:
        print("\n👋 部署服务已停止")


if __name__ == "__main__":
    main()