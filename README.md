# hd

一个用于大语言模型实验与输出管理的最小仓库骨架。

本仓库当前主要包含：
- `requirements.txt`：实验所需的 Python 依赖清单（以推理/训练 LLM、数据处理与可视化为主）。
- `output/`：模型运行输出目录，目前包含 Qwen3-8B-Base 的一次运行产物。
  - `output/Qwen3-8B-Base/v0-20250929-101225/runs/`：运行日志与输出文件（如 `run.log`）。

> 说明：仓库暂未包含具体的训练/推理脚本；你可以将项目脚本放在根目录或 `src/` 目录，并将输出写到 `output/` 目录。

## 环境与依赖

- Python 3.10+（建议）
- 依赖安装：

```bash
pip install -r requirements.txt
```

依赖涵盖：
- 深度学习与推理：`torch`, `transformers`, `accelerate`, `trl`, `peft`, `safetensors`, `triton`
- 数据与评测：`datasets`, `pandas`, `scikit-learn`, `scipy`, `umap-learn`, `rouge`, `tqdm`
- 服务与接口：`fastapi`, `uvicorn`, `gradio`, `openai`, `httpx`
- 可视化：`matplotlib`, `seaborn`, `tensorboard`
- 其他常用工具：`python-dotenv`, `rich`, `typer`, `ruff`

> 注意：依赖中包含 CUDA 相关包（如 `nvidia-cudnn-cu12` 等），请根据你的硬件环境决定是否保留；CPU 环境可考虑精简这些条目。

## 目录结构

```text
.
├─ requirements.txt          # 依赖清单
├─ models/                   # 模型目录（待下载新模型）
├─ output/                   # 运行输出目录
│  └─ Qwen3-8B-Base/
│     └─ v0-20250929-101225/
│        └─ runs/
│           └─ run.log       # 一次运行的日志
├─ .gitignore                # Git 忽略规则（已存在）
└─ .venv/                    # 本地虚拟环境（建议忽略提交）
```

## 模型准备

本仓库当前未包含预下载的模型。请等待提供新的模型链接以下载稳定的 Qwen 模型。

## 快速开始

模型下载完成后，我们将提供对话脚本和使用示例。

### 在线模型示例（可选）

如果需要测试在线模型，可以使用以下示例代码：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 使用本地下载的 7B 模型
model_path = "./models/qwen3-7b"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

prompt = "你好，给我一句关于秋天的诗意描述。"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=128,
        temperature=0.7,
        do_sample=True
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### 使用本地下载的 Qwen3 80B 模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 使用本地下载的 80B 模型（需要更多显存）
model_path = "./models/qwen3-80b"

tokenizer = AutoTokenizer.from_pretrained(model_path)

# 对于大模型，可能需要启用量化或模型并行
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    # load_in_8bit=True,  # 启用 8bit 量化节省显存
)

prompt = "请详细分析人工智能在教育领域的应用前景。"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

以下示例展示如何用 `transformers` 加载在线 Qwen 模型进行简单推理（需根据实际可用模型调整）：

### 在线模型示例（可选）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "Qwen/Qwen2.5-0.5B-Instruct"  # 示例模型，可按需替换

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

prompt = "你好，给我一句关于秋天的诗意描述。"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

运行完成后，你可以将输出与日志保存到 `output/` 目录，例如：

```python
from pathlib import Path
out_dir = Path("output/Qwen3-8B-Base/v0-20250929-101225/runs")
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "run.log").write_text("example output\n", encoding="utf-8")
```

## 开发建议

- 将核心脚本放入 `src/` 目录，并添加 `__init__.py` 以便作为包导入。
- 新建 `scripts/` 存放一键运行脚本（如推理、评测、数据预处理）。
- 使用 `.env` 管理密钥与配置，结合 `python-dotenv` 加载。
- 使用 `ruff` 保持代码风格一致：

```bash
ruff check .
```

## 常见问题

### 模型相关
- **模型太大无法加载**：尝试使用量化（8bit/4bit）或模型并行，在上述代码示例中取消 `load_in_8bit=True` 的注释。
- **80B 模型显存需求**：建议至少 80GB+ 显存，或使用多卡推理，或考虑使用量化版本。
- **重新下载模型**：删除对应的 `models/qwen3-*` 目录，重新运行下载脚本。

### 通用问题
- 推理显存不足：尝试更小的模型、开启 8bit/4bit 量化（需额外依赖），或使用 `accelerate` 分布式。
- CUDA 不匹配：根据你的 CUDA 版本选择对应的 PyTorch 与 NVIDIA 依赖，或在纯 CPU 环境下卸载/注释这些依赖。

### 模型管理

#### 检查模型大小
```bash
du -sh models/*/
```

#### 继续下载 80B 模型（如需要）
```python
# 如果 80B 模型下载中断，可以重新运行下载（会自动断点续传）
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-Next-80B-A3B-Instruct",
    local_dir="./models/qwen3-80b"
)
```

## 许可证

默认未声明许可证。如需开源，请添加 `LICENSE` 并在 README 顶部标注。
