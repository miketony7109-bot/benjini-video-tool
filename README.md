# 🎬 Benjini Video Tool

An AI-powered text-to-video generation web application built with **Streamlit** and **Wan 2.2** model.

Generate stunning 30-second videos from simple text prompts - completely free and open-source!

---

## ✨ Features

- 🎥 **Text-to-Video Generation**: Convert text prompts into 30-second videos
- 🚀 **Fast & Efficient**: Uses Wan 2.2 model (optimized for consumer GPUs)
- 💻 **Web Interface**: Simple, intuitive Streamlit UI
- 📥 **Video Download**: Download generated videos directly
- 🔄 **Multiple Generations**: Generate multiple videos with different settings
- 💾 **No Watermarks**: Completely free and open-source

---

## 🛠️ System Requirements

| Component | Requirement |
|-----------|-------------|
| **GPU** | NVIDIA GPU with 8GB+ VRAM (RTX 3060/4060 or better) |
| **CPU** | Modern multi-core processor |
| **RAM** | 16GB+ system RAM recommended |
| **Storage** | 50GB+ for model downloads |
| **OS** | Windows 10+, macOS 12+, or Linux |

---

## 📦 Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/miketony7109-bot/benjini-video-tool.git
cd benjini-video-tool
```

### Step 2: Create a Virtual Environment (Optional but Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: First installation may take 10-15 minutes as models are downloaded (~20GB).

### Step 4: Install PyTorch with CUDA Support
If you have an NVIDIA GPU, install the CUDA version:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CPU-only (much slower):
```bash
pip install torch torchvision torchaudio
```

---

## 🚀 Running Locally

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

### Quick Start Guide:
1. Enter your text prompt (e.g., "A serene mountain landscape at sunset")
2. Adjust video settings (duration, resolution, etc.)
3. Click "Generate Video"
4. Wait for generation (2-10 minutes depending on GPU)
5. Download your video!

---

## 📝 Usage Examples

### Example Prompts:

**Cinematic**: 
```
A cinematic shot of a futuristic city at night with flying cars and neon lights, 4K quality
```

**Nature**: 
```
A peaceful forest waterfall with sunlight filtering through the trees, birds chirping
```

**Abstract**: 
```
Colorful liquid paint swirling and mixing in water, creating abstract patterns
```

**Action**: 
```
A dragon flying through a mountain valley, breathing fire, epic fantasy scene
```

---

## 🌍 Deployment

### Option 1: Hugging Face Spaces (Recommended - FREE)

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Select "Streamlit" as the runtime
4. Link your GitHub repository: `https://github.com/miketony7109-bot/benjini-video-tool`
5. Add `requirements.txt` and `app.py`
6. Space will auto-deploy!

**Note**: Free tier has GPU limitations. For faster generation, upgrade to a paid GPU space.

### Option 2: Streamlit Cloud (FREE)

1. Sign up at [Streamlit Cloud](https://streamlit.io/cloud)
2. Connect your GitHub account
3. Deploy the repository
4. App goes live in minutes!

**Note**: Free tier has compute limitations.

### Option 3: Docker (Self-Hosted)

Create a `Dockerfile`:
```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y python3-pip python3-dev
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t benjini-video-tool .
docker run --gpus all -p 8501:8501 benjini-video-tool
```

---

## ⚙️ Configuration

### Environment Variables (Optional)

Create a `.env` file to customize settings:
```env
# Model selection
MODEL_ID=wanx-ai/Wan-2.2-T2V-A14B

# GPU settings
DEVICE=cuda  # or 'cpu'
DTYPE=float16  # or 'float32'

# Generation defaults
DEFAULT_STEPS=30
DEFAULT_HEIGHT=720
DEFAULT_WIDTH=1280
MAX_VIDEO_LENGTH=30
```

---

## 🎯 Model Information

**Wan 2.2 (Alibaba)**
- **Quality**: Excellent for 30-second videos
- **Speed**: 2-10 minutes per video on RTX 3080
- **VRAM**: 8-10GB optimal
- **Resolution**: Up to 1080p
- **License**: Open Source (Free)
- **GitHub**: [wanx-ai/Wan](https://github.com/wanx-ai/Wan)

---

## 🐛 Troubleshooting

### "Out of Memory" Error
```
Solution: Reduce resolution, decrease inference steps, or upgrade GPU
```

### "Model not found" Error
```
Solution: Check internet connection, ensure HuggingFace Hub access, clear cache:
rm -rf ~/.cache/huggingface
```

### "CUDA not available"
```
Solution: Reinstall PyTorch with CUDA support (see Installation Step 4)
```

### Slow Generation
```
Solutions:
- Reduce height/width
- Reduce num_inference_steps
- Enable mixed precision (float16)
- Upgrade to better GPU
```

---

## 📊 Performance Benchmarks

| GPU | Resolution | Steps | Time |
|-----|-----------|-------|------|
| RTX 3060 | 720p | 30 | 8-10 min |
| RTX 4080 | 1080p | 30 | 4-5 min |
| RTX 4090 | 1080p | 50 | 5-7 min |

---

## 📚 Additional Resources

- [Wan 2.2 GitHub](https://github.com/wanx-ai/Wan)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Hugging Face Models](https://huggingface.co)
- [PyTorch CUDA Setup](https://pytorch.org/get-started/locally/)

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

---

## 📄 License

This project is open-source and free to use. See the model licenses for terms and conditions.

---

## ⭐ Support

If you find this project helpful, please:
- ⭐ Star this repository
- 🔄 Share with friends
- 📢 Give feedback

---

## 📞 Contact & Support

For issues, questions, or suggestions:
- Open an [Issue](https://github.com/miketony7109-bot/benjini-video-tool/issues)
- Check [Discussions](https://github.com/miketony7109-bot/benjini-video-tool/discussions)

---

## 🙏 Acknowledgments

- **Wan 2.2**: Alibaba DAMO Academy
- **Streamlit**: For the amazing framework
- **Hugging Face**: For model hosting and tools
- **PyTorch**: For the deep learning backbone

---

**Happy Video Generating! 🎬✨**

Last Updated: May 8, 2026
