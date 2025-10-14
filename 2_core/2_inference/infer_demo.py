import os
import time
import threading
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re



def generate_response(model, tokenizer, prompt: str, max_tokens: int = 10000, temperature: float = 0.7) -> Dict:
    """
    推理函数，负责生成回复
    
    Args:
        model: 已加载的语言模型
        tokenizer: 对应的分词器
        prompt: 用户输入的提示文本
        max_tokens: 最大生成token数量，默认512
        temperature: 采样温度，控制生成随机性，默认0.7
    
    Returns:
        Dict: 包含生成的回复文本和详细统计信息的字典
    """
    # 构建对话格式的消息列表，包含用户角色和内容
    messages = [{"role": "user", "content": prompt}]
    
    # 使用分词器的聊天模板将消息转换为模型可理解的文本格式
    # add_generation_prompt=True 会添加助手的回复提示符
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 将文本转换为token ID张量，并移动到模型所在的设备（GPU/CPU）
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # 记录推理开始时间，用于性能统计
    start_time = time.time()
    
    # 使用torch.no_grad()禁用梯度计算，节省内存并加速推理
    with torch.no_grad():
        # 调用模型的生成方法进行文本生成
        outputs = model.generate(
            **inputs,                           # 解包输入张量（包含input_ids, attention_mask等）
            max_new_tokens=max_tokens,          # 限制新生成的token数量
            temperature=temperature,            # 控制生成的随机性，越高越随机
            do_sample=True,                     # 启用采样而非贪心解码
            pad_token_id=tokenizer.eos_token_id,  # 设置填充token ID
            eos_token_id=tokenizer.eos_token_id,  # 设置结束token ID，明确指定停止条件
            # 以下参数用于提高生成质量，避免乱码和重复
            repetition_penalty=1.1,            # 重复惩罚，降低重复内容的概率
            no_repeat_ngram_size=3              # 防止3-gram及以上的重复序列
        )
    
    # 记录推理结束时间
    end_time = time.time()
    
    # ========== 计算性能统计信息 ==========
    generation_time = end_time - start_time                    # 总生成耗时（秒）
    input_length = inputs['input_ids'].shape[1]               # 输入prompt的token长度
    total_length = outputs[0].shape[0]                        # 输出序列的总token长度
    generated_tokens = total_length - input_length            # 新生成的token数量
    # 计算每秒生成的token数，避免除零错误
    tokens_per_second = generated_tokens / generation_time if generation_time > 0 else 0
    
    # ========== 改进的文本解码和清理 ==========
    try:
        # 只提取新生成的部分token IDs，避免包含输入的prompt
        generated_ids = outputs[0][input_length:]
        
        # 解码生成的token为文本
        # skip_special_tokens=True: 跳过特殊token如<eos>, <pad>等
        # clean_up_tokenization_spaces=True: 清理分词产生的多余空格
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        # 使用专门的清理函数移除乱码和特殊字符
        response_text = clean_text_output(response_text)
        
        # 移除可能残留的结束标记文本（某些模型可能会输出可见的结束符）
        if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token:
            response_text = response_text.replace(tokenizer.eos_token, '')
        
        # ========== 处理可能被截断的句子 ==========
        # 检查文本长度足够且可能存在截断问题
        if response_text and len(response_text) > 10:
            # 获取文本的最后一个字符
            last_char = response_text[-1]
            
            # 如果最后一个字符不是标点符号也不是字母数字，可能是截断的
            if last_char not in '.!?。！？' and not last_char.isalnum():
                # 使用正则表达式按句子分割，保留分隔符
                sentences = re.split(r'([.!?。！？])', response_text)
                
                # 如果有多个句子片段，尝试重组到最后一个完整句子
                if len(sentences) > 2:
                    # 重新组合完整的句子（每两个元素组成一个完整句子）
                    complete_text = ''
                    for i in range(0, len(sentences) - 2, 2):  # 步长为2，跳过最后可能不完整的句子
                        if i + 1 < len(sentences):
                            complete_text += sentences[i] + sentences[i + 1]  # 句子内容 + 标点符号
                    
                    # 如果重组后有内容，使用重组的文本
                    if complete_text:
                        response_text = complete_text
        
    except Exception as e:
        # ========== 异常处理：备用解码方法 ==========
        print(f"⚠️ 文本解码出现问题: {e}")
        
        # 使用传统方法：解码完整输出然后截取
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        response_text = full_response[len(text):].strip()  # 移除输入部分，保留生成部分
        
        # 仍然使用清理函数处理可能的乱码
        response_text = clean_text_output(response_text)
    
    # ========== 返回结果字典 ==========
    return {
        'response': response_text,  # 清理后的生成文本
        'stats': {                  # 详细的性能统计信息
            'generation_time': generation_time,                           # 生成耗时（秒）
            'generated_tokens': generated_tokens,                        # 生成的token数量
            'input_length': input_length,                               # 输入长度
            'total_length': total_length,                               # 总输出长度
            'tokens_per_second': tokens_per_second,                     # 生成速度（tokens/秒）
            'ms_per_token': (generation_time * 1000) / generated_tokens if generated_tokens > 0 else 0  # 每token耗时（毫秒）
        }
    }



def clean_text_output(text: str) -> str:
    """清理文本输出，移除乱码和特殊字符"""
    if not text:
        return text
    
    # 移除替换字符（通常表示乱码）
    text = text.replace('\ufffd', '')  # Unicode 替换字符
    text = text.replace('�', '')       # 常见的乱码字符
    
    # 移除其他控制字符（保留常用的换行、制表符）
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x84\x86-\x9F]', '', text)
    
    # 移除可能的不完整的多字节字符
    try:
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
    except:
        pass
    
    # 清理多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def display_gpu_info():
    """显示GPU配置信息"""
    print("🔧 GPU 配置信息:")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("⚠️  未检测到可用的 GPU")
        return
    
    print(f"检测到 GPU 数量: {torch.cuda.device_count()}")
    print(f"当前设备: {torch.cuda.current_device()}")
    
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '未设置（将使用所有GPU）')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_devices}")


def load_model(model_path: str):
    """加载模型和分词器"""
    print("正在加载模型...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("✅ 模型加载成功！")
    return model, tokenizer


def display_model_info(model):
    """显示模型分布和显存使用信息"""
    print("\n📊 模型设备分布:")
    
    try:
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            print("设备映射:")
            device_map = model.hf_device_map
            if isinstance(device_map, dict):
                for layer, device in device_map.items():
                    print(f"  {layer}: {device}")
            else:
                print(f"  设备映射类型: {type(device_map)}")
        else:
            print(f"模型主设备: {next(model.parameters()).device}")
        
        # 检查模型参数分布
        device_count = {}
        for name, param in model.named_parameters():
            device = str(param.device)
            device_count[device] = device_count.get(device, 0) + 1
        
        print("参数分布统计:")
        for device, count in device_count.items():
            print(f"  {device}: {count} 个参数张量")
            
    except Exception as e:
        print(f"无法获取设备映射信息: {e}")
        print(f"模型设备: {next(model.parameters()).device}")
    
    # 显示每个GPU的显存使用情况
    print("\n💾 GPU 显存使用情况:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: 已分配 {allocated:.2f}GB / 预留 {reserved:.2f}GB / 总计 {total:.1f}GB")

def display_inference_stats(stats: Dict, response_text: str = ""):
    """显示推理统计信息"""
    print(f"\n📈 推理性能统计:")
    print(f"  ⏱️  总生成时间: {stats['generation_time']:.3f} 秒")
    print(f"  🔢 生成token数: {stats['generated_tokens']} 个")
    print(f"  🚀 推理速度: {stats['tokens_per_second']:.2f} tokens/秒")
    print(f"  ⚡ 每token用时: {stats['ms_per_token']:.1f} 毫秒")
    print(f"  📥 输入长度: {stats['input_length']} tokens")
    print(f"  📤 总输出长度: {stats['total_length']} tokens")
    print(f"  📊 输出/输入比: {stats['generated_tokens']/stats['input_length']:.2f}x")
    
    # 文本质量检查
    if response_text:
        # 检查是否包含乱码字符
        has_garbled = '�' in response_text or '\ufffd' in response_text
        # 检查是否有异常的字符比例
        printable_chars = sum(1 for c in response_text if c.isprintable() or c in '\n\t')
        total_chars = len(response_text)
        printable_ratio = printable_chars / total_chars if total_chars > 0 else 1.0
        
        if has_garbled:
            print(f"  ⚠️  文本质量: 检测到乱码字符")
        elif printable_ratio < 0.95:
            print(f"  ⚠️  文本质量: 可打印字符比例 {printable_ratio:.1%}")
        else:
            print(f"  ✅ 文本质量: 正常")
    
    # 性能等级评估
    speed = stats['tokens_per_second']
    if speed >= 50:
        performance_level = "🔥 极快"
    elif speed >= 30:
        performance_level = "🚀 很快"
    elif speed >= 15:
        performance_level = "✅ 良好"
    elif speed >= 8:
        performance_level = "⚡ 一般"
    else:
        performance_level = "🐌 较慢"
    
    print(f"  🎯 性能等级: {performance_level}")


def run_benchmark(model, tokenizer, test_prompts: Optional[List[str]] = None, num_runs: int = 3) -> List[float]:
    """执行性能基准测试"""
    if test_prompts is None:
        test_prompts = [
            "你好，请介绍一下自己",
            "请写一首关于春天的短诗",
            "解释一下什么是人工智能",
            "用Python写一个简单的排序算法",
            "请详细解释一下大语言模型的工作原理"
        ]
    
    print(f"\n🧪 开始性能基准测试 (共 {len(test_prompts)} 个问题，每个运行 {num_runs} 次)")
    print("=" * 60)
    
    all_speeds = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 测试问题 {i}/{len(test_prompts)}: {prompt[:50]}...")
        speeds_for_prompt = []
        
        for run in range(num_runs):
            print(f"\n  🔄 第 {run+1}/{num_runs} 次运行:")
            
            result = generate_response(model, tokenizer, prompt, max_tokens=200)
            speed = result['stats']['tokens_per_second']
            
            speeds_for_prompt.append(speed)
            all_speeds.append(speed)
            
            print(f"    ⚡ 速度: {speed:.2f} tokens/秒 ({result['stats']['generated_tokens']} tokens in {result['stats']['generation_time']:.2f}s)")
        
        avg_speed = sum(speeds_for_prompt) / len(speeds_for_prompt)
        print(f"  📊 平均速度: {avg_speed:.2f} tokens/秒")
    
    # 总体统计
    print(f"\n🏆 总体性能统计:")
    print(f"  📈 平均推理速度: {sum(all_speeds)/len(all_speeds):.2f} tokens/秒")
    print(f"  ⚡ 最快速度: {max(all_speeds):.2f} tokens/秒")
    print(f"  🐌 最慢速度: {min(all_speeds):.2f} tokens/秒")
    
    if len(all_speeds) > 1:
        variance = sum([(x - sum(all_speeds)/len(all_speeds))**2 for x in all_speeds]) / len(all_speeds)
        std_dev = variance ** 0.5
        print(f"  📊 速度标准差: {std_dev:.2f}")
    
    return all_speeds


def main():
    """主函数"""
    # 模型路径选择 - 暂时使用30B模型测试
    # model_path = "/home/work/hd/_models/base/Qwen3-4B-Thinking-2507-FP8"
    model_path = "/home/work/hd/_models/base/Qwen3-30B-A3B-Thinking-2507"
    
    # 1. 显示GPU信息
    display_gpu_info()
    print("\n" + "="*50)
    
    # 2. 加载模型
    model, tokenizer = load_model(model_path)
    
    # 3. 显示模型信息
    display_model_info(model)
    
    # 4. 执行推理
    prompt = "请详细解释一下大语言模型的思考模式（Thinking mode）是什么？"
    print(f"\n用户问题: {prompt}")
    print("=" * 50)
    print("AI回复:")
    
    result = generate_response(model, tokenizer, prompt)
    print(result['response'])
    
    # 5. 显示统计信息（包含文本质量检查）
    display_inference_stats(result['stats'], result['response'])
    
    # 6. 可选：运行性能基准测试
    print(f"\n✅ 推理完成！")
    print(f"💡 如需运行性能基准测试，请调用: run_benchmark(model, tokenizer)")


if __name__ == "__main__":
    main()
