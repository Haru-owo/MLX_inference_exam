
---

# 🧠 MLX LLM Inference Example

This repository is a practice project for running LLM (Large Language Model) inference locally using Apple’s [MLX](https://github.com/ml-explore/mlx) framework.  
It allows you to easily convert pretrained Hugging Face models into MLX format with optional 4-bit quantization.

---

## ⚠️ Disclaimer

This project was created for the purpose of experimenting with MLX and **is not guaranteed to work perfectly**.  
Due to my limited coding skills (plus help from GPT), the code structure may be quite complex and difficult to follow.  
**I do not recommend using this code as-is for production purposes.**  
Instead, I suggest using it as a reference to understand how MLX works and how it differs from traditional CUDA-based systems.

*Since the generation_config file is not always included when downloading a model, it is necessary to check its presence.*

---

## 📦 Key Features

- ✅ Convert Hugging Face models to MLX format
- ✅ Optional 4-bit quantization (`--quantize`)
- ✅ Real-time token generation (streaming)
- ✅ Clean modular design (inference, UI, and sampling utilities separated)
- ✅ Optimized for Apple Silicon (M1/M2/M3)

---

## 📁 Project Structure

```
LMX_inference_exam/
├── mlx_models/                          # Converted MLX models (auto-generated)
│   ├── DeepSeek-R1-Distill-Qwen-1.5B/   # Example (after running install.py)
│   └── EXAONE-Deep-7.8B-int4/           # Quantized example (after running install.py)
├── install.py                           # Main script for downloading and converting models
├── inference.py                         # Core inference logic
├── start.py                             # Simple chat interface (entry point)
├── sampler_utils.py                     # Token sampling utilities
├── token.json.example                   # Hugging Face token example (see setup below)
├── requirements.txt                     # Python dependencies
└── .gitignore                           
```

---

## ⚙️ Requirements

- macOS + Apple Silicon (M1, M2, M3)
- Python 3.10 or later (3.13 recommended)
- [MLX installed](https://github.com/ml-explore/mlx)
- Hugging Face access token (`token.json` required)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔑 Setting Up Hugging Face Token

Rename the included `token.json.example` file to `token.json`, and add your Hugging Face access token:

```json
{
  "hf_token": "your_huggingface_access_token"
}
```

---

## 🚀 How to Use

### 1. Convert model without quantization

```bash
python install.py <model-name>
```

Example:

```bash
python install.py deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

Model will be saved to:

```
mlx_models/DeepSeek-R1-Distill-Qwen-1.5B/
```

---

### 2. Convert model with 4-bit quantization

```bash
python install.py <model-name> --quantize
```

Example:

```bash
python install.py LGAI-EXAONE/EXAONE-Deep-7.8B --quantize
```

> MLX does not currently support selecting the number of quantization bits.  
> Enabling `--quantize` applies the default 4-bit quantization.

Model will be saved to:

```
mlx_models/EXAONE-Deep-7.8B-int4/
```

---

## 💬 Run Chat UI (Inference)

```bash
python start.py
```

---

## 📄 License

This project is licensed under the **GNU General Public License v3.0**.  
See the [LICENSE](LICENSE) file for details.

---

## 🛠 Troubleshooting

If the `python` command doesn’t work on your system, try using `python3` instead:

```bash
python3 install.py <model-name>
```

---
