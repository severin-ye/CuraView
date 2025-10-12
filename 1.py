import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 指定GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 注释掉，使用所有可用GPU

# 显示GPU信息
print("🔧 GPU 配置信息:")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"检测到 GPU 数量: {torch.cuda.device_count()}")
    print(f"当前设备: {torch.cuda.current_device()}")
    
    # 显示所有GPU信息
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
        print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置（将使用所有GPU）')}")
else:
    print("⚠️  未检测到可用的 GPU")

print("\n" + "="*50)

# 修复后的模型路径
# model_path = "/home/work/hd/models/Qwen3-4B-Thinking-2507-FP8"
model_path = "/home/work/hd/models/Qwen3-30B-A3B-Thinking-2507"

print("正在加载模型...")

# 使用 transformers 直接加载（避免兼容性问题）
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # 自动分配到多个GPU
    trust_remote_code=True
)

print("✅ 模型加载成功！")

# 显示模型分布信息
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
    allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
    reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
    total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
    print(f"  GPU {i}: 已分配 {allocated:.2f}GB / 预留 {reserved:.2f}GB / 总计 {total:.1f}GB")

# 推理函数
def generate_response(prompt, max_tokens=float('inf'), temperature=0.7):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(text):].strip()

# 执行推理
prompt = "请详细解释一下大语言模型的思考模式（Thinking mode）是什么？"
# 打印用户的问题
print(f"\n用户问题: {prompt}")
# 打印分隔线
print("=" * 50) 
# 显示AI回复标记
print("AI回复:")

response = generate_response(prompt)
print(response)
