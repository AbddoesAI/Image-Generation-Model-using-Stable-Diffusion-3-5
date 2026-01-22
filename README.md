# Stable Diffusion 3.5 Fine-tuning on Google Colab

A complete, production-ready Google Colab notebook for fine-tuning Stable Diffusion 3.5 using LoRA (Low-Rank Adaptation). Optimized for consumer GPUs with automatic VRAM detection and memory-efficient training strategies.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/sd35-colab-training/blob/main/SD35_Training.ipynb)

---

## 🌟 Features

- **🚀 One-Click Training**: Complete pipeline from data loading to inference
- **💾 VRAM-Optimized**: Auto-detects GPU tier (T4/L4/A100) and adjusts settings
- **🎯 LoRA Training**: Train on 10-16GB VRAM with minimal quality loss
- **📊 Resolution Bucketing**: Preserves aspect ratios during training
- **🔄 Multiple Schedulers**: Compare Euler, DPM++, and DDIM samplers
- **💿 Auto-Checkpointing**: Saves to Google Drive during training
- **🎨 Validation Images**: Generate samples during training to monitor progress
- **⚡ Memory Efficient**: Gradient checkpointing, 8-bit optimizers, xFormers

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Requirements](#-requirements)
- [Notebook Overview](#-notebook-overview)
- [Training Your Own Model](#-training-your-own-model)
- [License & Usage](#-license--usage)
- [Troubleshooting](#-troubleshooting)
- [Advanced Usage](#-advanced-usage)
- [Citation](#-citation)

---

## 🚀 Quick Start

### 1. Prerequisites

- **Google Account** (for Colab)
- **Hugging Face Account** ([Sign up free](https://huggingface.co/join))
- **Hugging Face Token** with read permissions ([Create here](https://huggingface.co/settings/tokens))
- **Google Colab Pro** (recommended for A100/L4 access, but free T4 works)

### 2. Setup (5 minutes)

```bash
# 1. Click the "Open in Colab" badge above
# 2. Go to Runtime → Change runtime type → Select GPU (T4/L4/A100)
# 3. Run the first cell to detect your GPU
# 4. Replace YOUR_HUGGINGFACE_TOKEN_HERE with your actual token
# 5. Run all cells (Runtime → Run all)
```

### 3. Default Configuration

The notebook comes pre-configured with:
- **Dataset**: `lambdalabs/naruto-blip-captions` (example)
- **Model**: Stable Diffusion 3.5 Large
- **Training**: 10 epochs, LoRA rank 16
- **Resolution**: 1024x1024 (auto-bucketed)
- **Batch Size**: Auto-adjusted based on VRAM

---

## 📦 Requirements

### Minimum Hardware
- **GPU**: NVIDIA T4 (16GB VRAM) or better
- **RAM**: 12GB system RAM
- **Storage**: 20GB free space in Google Drive

### Recommended Hardware
- **GPU**: L4 (24GB) or A100 (40GB)
- **RAM**: 32GB+ system RAM
- **Storage**: 50GB+ for checkpoints and datasets

### Software Dependencies
All dependencies are auto-installed in the notebook:
```
torch>=2.1.0
diffusers>=0.30.3
transformers>=4.44.2
accelerate>=0.34.2
xformers>=0.0.22
bitsandbytes>=0.43.3
peft>=0.12.0
datasets>=2.20.0
```

---

## 📚 Notebook Overview

### Section 1: Environment Setup
- GPU detection and VRAM analysis
- Dependency installation with exact versions
- Hugging Face authentication
- Google Drive mounting for checkpoint storage

### Section 2: SD3.5 Architecture Deep Dive
- **Multimodal Diffusion Transformer (MMDiT)** architecture
- **Three text encoders**: CLIP-L, CLIP-G, T5-XXL
- **Rectified Flow** training framework
- Comparison with SDXL and previous versions

### Section 3: Data Pipeline
- **Multiple data sources**: Hugging Face datasets, Google Drive, local upload
- **Resolution bucketing**: Maintains aspect ratios (1:1, 3:2, 9:16, etc.)
- **Auto-captioning**: Loads `.txt` files alongside images
- **Data visualization**: Preview your dataset before training

### Section 4: Training Strategy
- **LoRA configuration**: Rank, alpha, target modules
- **VRAM-aware optimization**: Automatic settings based on GPU
- **Scheduler options**: Cosine, linear, constant with warmup
- **Mixed precision**: FP16/BF16 support

### Section 5: Training Loop
- **Rectified flow loss**: Native SD3.5 training objective
- **Gradient accumulation**: Effective batch sizes on limited VRAM
- **Accelerate integration**: Multi-GPU ready
- **Checkpoint management**: Auto-save every N steps
- **Validation generation**: Monitor training progress visually

### Section 6: Inference & Generation
- **LoRA weight loading**: Merge trained adapters
- **Scheduler comparison**: Test different samplers
- **CFG tuning**: Guidance scale experimentation
- **Batch generation**: Create multiple variations

### Section 7: Evaluation (Coming Soon)
- Qualitative assessment
- CLIP score calculation
- FID metrics
- Common failure modes

### Section 8: Export & Deployment (Coming Soon)
- Save to Hugging Face Hub
- Export for local use
- ONNX conversion
- Inference optimization

---

## 🎨 Training Your Own Model

### Option 1: Using Hugging Face Datasets

```python
# In Cell 3, modify:
DATASET_NAME = "your-username/your-dataset"
```

Your dataset should have:
- `image` column (PIL Images)
- `text` or `caption` column (strings)

### Option 2: Using Google Drive

```python
# 1. Upload images to Drive: /content/drive/MyDrive/my_training_images/
# 2. Add .txt files with same name as images for captions
# 3. In Cell 4, modify:
train_dataset = SD35Dataset(
    "/content/drive/MyDrive/my_training_images",
    resolution=1024
)
```

Structure:
```
my_training_images/
├── image001.jpg
├── image001.txt  # "a beautiful sunset over mountains"
├── image002.png
├── image002.txt  # "cyberpunk city street at night"
└── ...
```

### Option 3: Direct Upload (Small Datasets)

```python
from google.colab import files
uploaded = files.upload()

# Then point dataset loader to uploaded files
```

### Recommended Training Settings by Use Case

| Use Case | Rank | Epochs | Learning Rate | Dataset Size |
|----------|------|--------|---------------|--------------|
| **Style Transfer** | 8-16 | 5-10 | 1e-4 | 50-200 |
| **Character/Subject** | 16-32 | 10-20 | 5e-5 | 100-500 |
| **Concept Learning** | 32-64 | 15-30 | 3e-5 | 500-2000 |
| **Domain Adaptation** | 64-128 | 20-50 | 1e-5 | 2000+ |

---

## ⚖️ License & Usage

### 📜 Stable Diffusion 3.5 Model License

This notebook uses **Stable Diffusion 3.5** by Stability AI, licensed under the [Stability AI Community License](https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/LICENSE.md).

**Key Terms:**
- ✅ **Free for research and non-commercial use**
- ✅ **Free for commercial use** if your organization's annual revenue < $1,000,000 USD
- ⚠️ **Enterprise License required** for organizations with revenue ≥ $1,000,000 USD
- ✅ **Generated outputs are owned by you**, subject to applicable law
- ⚠️ Must comply with [Acceptable Use Policy](https://stability.ai/use-policy)

**Commercial Use Examples:**
- Freelancer earning $50k/year: ✅ Free
- Startup with $800k revenue: ✅ Free  
- Corporation with $5M revenue: ❌ Needs Enterprise License
- Selling generated art: ✅ Allowed (if under revenue threshold)



### ⚠️ Important Disclaimers

1. **No Model Weights Included**: This repository contains only training code. You must download model weights separately from Hugging Face.

2. **User Responsibility**: You are responsible for:
   - Complying with SD3.5 license terms
   - Following Stability AI's Acceptable Use Policy
   - Ensuring your use case is permitted
   - Verifying commercial licensing requirements

3. **Generated Content**: You own your generated outputs, but must:
   - Not use them to violate others' rights
   - Follow applicable laws in your jurisdiction
   - Comply with platform-specific terms if sharing

4. **No Warranty**: This software is provided "as is" without warranty of any kind.

### 🔗 Additional Resources

- [SD3.5 Model Card](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)
- [Stability AI Licensing FAQ](https://stability.ai/license)
- [Enterprise License Contact](https://stability.ai/enterprise)
- [Acceptable Use Policy](https://stability.ai/use-policy)

---

## 🐛 Troubleshooting

### Common Issues

#### "CUDA out of memory"
```python
# Solutions (in order of preference):
# 1. Reduce batch size
config.train_batch_size = 1

# 2. Increase gradient accumulation
config.gradient_accumulation_steps = 8

# 3. Lower resolution
config.resolution = 768

# 4. Reduce LoRA rank
config.lora_rank = 8

# 5. Clear cache between runs
torch.cuda.empty_cache()
```

#### "Model download failed"
```python
# Verify:
# 1. Hugging Face token is valid
# 2. You've accepted the model license at:
#    https://huggingface.co/stabilityai/stable-diffusion-3.5-large
# 3. Token has "read" permissions
```

#### "Training loss not decreasing"
```python
# Check:
# 1. Learning rate might be too low/high
config.learning_rate = 5e-5  # Try different values: 1e-5 to 1e-4

# 2. Dataset quality
# - Ensure captions are descriptive
# - Verify images aren't corrupted
# - Check aspect ratios aren't too extreme

# 3. Increase training time
config.num_epochs = 20  # More epochs for complex concepts
```

#### "Generated images look identical to training"
```python
# This is overfitting. Solutions:
# 1. Use more training data (100+ images minimum)
# 2. Reduce LoRA rank
config.lora_rank = 8

# 3. Fewer epochs
config.num_epochs = 5

# 4. Add regularization
config.lora_dropout = 0.1
```

#### "Colab disconnected during training"
```python
# Prevention:
# 1. Checkpoint frequently
config.checkpointing_steps = 250  # Save every 250 steps

# 2. Keep tab active (prevents idle timeout)
# 3. Use Colab Pro for longer runtimes
# 4. Run during off-peak hours

# Recovery:
# Resume from last checkpoint saved to Google Drive
```

### Performance Optimization

#### Speed up training:
```python
# 1. Use lower precision
config.mixed_precision = "bf16"  # On A100

# 2. Enable CPU offloading (if desperate)
pipe.enable_model_cpu_offload()

# 3. Reduce validation frequency
config.validation_steps = 500

# 4. Use fewer dataloader workers
config.dataloader_num_workers = 0
```

#### Improve quality:
```python
# 1. Better captions
# Use descriptive, detailed captions (20-50 words)

# 2. Consistent image quality
# Preprocess images: resize, crop, color correct

# 3. Longer training
config.num_epochs = 30

# 4. Higher LoRA rank (if VRAM allows)
config.lora_rank = 32
```

### GPU-Specific Tips

**T4 (16GB VRAM):**
```python
config.train_batch_size = 1
config.gradient_accumulation_steps = 8
config.lora_rank = 8
config.resolution = 768
```

**L4 (24GB VRAM):**
```python
config.train_batch_size = 1
config.gradient_accumulation_steps = 4
config.lora_rank = 16
config.resolution = 1024
```

**A100 (40GB VRAM):**
```python
config.train_batch_size = 2
config.gradient_accumulation_steps = 2
config.lora_rank = 32
config.resolution = 1024
config.mixed_precision = "bf16"  # Better than fp16
```

---

## 🚀 Advanced Usage

### Custom Loss Functions

Add perceptual loss for better quality:

```python
import torch.nn.functional as F

# In training loop, after computing MSE loss:
# Add LPIPS or CLIP-based loss
perceptual_loss = lpips_model(model_pred, target)
total_loss = mse_loss + 0.1 * perceptual_loss
```

### Multi-LoRA Training

Train multiple LoRAs for different concepts:

```python
# Train LoRA 1 on style dataset
# Train LoRA 2 on subject dataset
# Merge at inference:

pipe.load_lora_weights("path/to/style_lora", adapter_name="style")
pipe.load_lora_weights("path/to/subject_lora", adapter_name="subject")
pipe.set_adapters(["style", "subject"], adapter_weights=[0.7, 0.8])
```

### Export to ONNX

```python
from optimum.onnxruntime import ORTStableDiffusionPipeline

# Convert to ONNX for faster inference
ort_pipe = ORTStableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    export=True,
)
ort_pipe.save_pretrained("sd35_onnx")
```

### Weights & Biases Integration

```python
# Enable W&B logging
config.report_to = "wandb"

# Login before training
import wandb
wandb.login()
wandb.init(project="sd35-finetuning", name="my-experiment")
```

### Custom Schedulers

```python
from diffusers import DDPMScheduler

# Use DDPM for training (alternative to rectified flow)
noise_scheduler = DDPMScheduler.from_pretrained(
    config.model_id,
    subfolder="scheduler"
)
```

---

## 📊 Example Results

### Before Fine-tuning
> Prompt: "a portrait in the style of [your style]"
- Generic anime style, not matching training data

### After Fine-tuning (10 epochs, 200 images)
> Same prompt
- Distinctive style matching your dataset
- Character features preserved
- Consistent aesthetic

*Note: Add your own example images to the `/examples` folder*

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add DreamBooth training option
- [ ] Implement textual inversion
- [ ] Add CLIP-based evaluation metrics
- [ ] Support for video generation (SD3.5-turbo)
- [ ] AutoML for hyperparameter tuning
- [ ] Multi-GPU distributed training
- [ ] ControlNet integration
- [ ] WebUI for non-coders

**How to contribute:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📖 Citation

If you use this notebook in your research or project, please cite:

```bibtex
@software{sd35_colab_training,
  author = {Your Name},
  title = {Stable Diffusion 3.5 Fine-tuning on Google Colab},
  year = {2025},
  url = {https://github.com/YOUR_USERNAME/sd35-colab-training}
}

@misc{esser2024sd3,
  title={Scaling Rectified Flow Transformers for High-Resolution Image Synthesis},
  author={Esser, Patrick and Kulal, Sumith and Blattmann, Andreas and others},
  year={2024},
  publisher={Stability AI}
}
```

---

## 🙏 Acknowledgments

- **Stability AI** for Stable Diffusion 3.5
- **Hugging Face** for the Diffusers library
- **Google Colab** for free GPU access
- **Community contributors** who improve this notebook

---


## 📜 Changelog

### v1.0.0 (2025-01-22)
- Initial release
- LoRA training support
- Auto VRAM detection
- Resolution bucketing
- Multi-scheduler support

### Roadmap
- v1.1.0: DreamBooth integration
- v1.2.0: Textual inversion
- v1.3.0: ControlNet support
- v2.0.0: WebUI interface

---

## ⭐ Star History

If you find this useful, please star the repository!

---

**Made with ❤️ for the AI art community**

*Remember: With great generative power comes great responsibility. Use AI ethically and respect others' rights.*
