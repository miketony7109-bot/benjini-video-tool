"""
Benjini Video Tool - AI Text-to-Video Generator
A Streamlit app for generating videos from text prompts using Wan 2.2 model
"""

import streamlit as st
import torch
import os
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Benjini Video Tool",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton > button {
        width: 100%;
        padding: 0.75rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        cursor: pointer;
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        transform: scale(1.02);
    }
    .header-title {
        text-align: center;
        color: #667eea;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .header-subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Create output directory
OUTPUT_DIR = Path("generated_videos")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_model():
    """Load the Wan 2.2 model"""
    try:
        from diffusers import WanPipeline
        
        st.info("🔄 Loading Wan 2.2 model... This may take a moment on first run.")
        
        model_id = "wanx-ai/Wan-2.2-T2V-A14B"
        
        # Load pipeline
        pipe = WanPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            st.success("✅ Model loaded on GPU!")
        else:
            st.warning("⚠️ GPU not detected. Using CPU (slower). For better performance, use a GPU.")
            pipe = pipe.to("cpu")
        
        return pipe
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        logger.error(f"Model loading error: {str(e)}")
        return None

def generate_video(pipe, prompt, negative_prompt, num_steps, guidance_scale, height, width, num_frames):
    """Generate video from text prompt"""
    try:
        logger.info(f"Generating video with prompt: {prompt}")
        
        with st.spinner("🎬 Generating your video... This may take 2-10 minutes depending on your hardware."):
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=int(num_steps),
                guidance_scale=float(guidance_scale),
                height=int(height),
                width=int(width),
                num_frames=int(num_frames),
                generator=torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
            )
            
            videos = output.videos
            return videos[0] if videos else None
    except Exception as e:
        st.error(f"❌ Error during generation: {str(e)}")
        logger.error(f"Generation error: {str(e)}")
        return None

def save_video(video_frames, filename):
    """Save video frames to file"""
    try:
        import cv2
        import numpy as np
        
        filepath = OUTPUT_DIR / filename
        
        # Assuming video_frames is a PIL Image or numpy array
        # Convert to numpy array if needed
        if hasattr(video_frames, 'numpy'):
            frames = video_frames.numpy()
        else:
            frames = np.array(video_frames)
        
        # Define codec and create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 15
        
        if frames.ndim == 4:  # Multiple frames
            height, width = frames[0].shape[:2]
            out = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))
            
            for frame in frames:
                if frame.dtype == np.float32 or frame.dtype == np.float64:
                    frame = (frame * 255).astype(np.uint8)
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()
        
        return filepath
    except Exception as e:
        logger.error(f"Error saving video: {str(e)}")
        return None

# Main UI
st.markdown('<div class="header-title">🎬 Benjini Video Tool</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Generate 30-Second Videos from Text Prompts using AI</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.subheader("Video Parameters")
    duration_seconds = st.slider("Video Duration (seconds):", 5, 30, 10)
    num_frames = int((duration_seconds / 5) * 25)  # Approximate frames
    
    height = st.selectbox("Video Height:", [480, 576, 720], index=2)
    width = st.selectbox("Video Width:", [848, 1024, 1280], index=2)
    
    num_inference_steps = st.slider("Quality Steps (more = better quality, slower):", 20, 50, 30)
    guidance_scale = st.slider("Prompt Adherence:", 1.0, 10.0, 5.0)
    
    st.divider()
    
    st.subheader("About")
    st.info("""
    **Benjini Video Tool** uses the Wan 2.2 model to generate 
    high-quality videos from text descriptions.
    
    - 📹 Supports up to 30-second videos
    - 🎨 High-quality output
    - ⚡ Optimized for consumer GPUs
    - 🆓 100% free and open-source
    
    **Requirements:**
    - GPU with 8GB+ VRAM recommended
    - Stable internet connection
    - Patience (first run downloads ~30GB model)
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Prompt")
    prompt = st.text_area(
        "Enter your video prompt:",
        placeholder="e.g., A serene landscape with mountains and sunset, cinematic quality",
        height=100
    )
    
    negative_prompt = st.text_area(
        "Negative prompt (optional):",
        placeholder="e.g., blurry, low quality, distorted",
        height=80
    )

with col2:
    st.subheader("ℹ️ Information")
    st.info(f"""
    **Generation Settings:**
    - Duration: ~{duration_seconds} seconds
    - Resolution: {width}x{height}
    - Steps: {num_inference_steps}
    - Guidance Scale: {guidance_scale}
    - Estimated Frames: {num_frames}
    
    **Expected Time:** 2-10 minutes depending on your hardware
    
    **Tips:**
    - Be descriptive in your prompt
    - Include style (cinematic, cartoon, etc.)
    - Specify camera movement if desired
    """)

st.divider()

# Generate button
if st.button("🚀 Generate Video", use_container_width=True):
    if not prompt.strip():
        st.error("❌ Please enter a prompt!")
    else:
        # Load model
        pipe = load_model()
        
        if pipe:
            # Generate video
            video = generate_video(
                pipe,
                prompt,
                negative_prompt,
                num_inference_steps,
                guidance_scale,
                height,
                width,
                num_frames
            )
            
            if video is not None:
                st.success("✅ Video generated successfully!")
                
                # Save video
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"video_{timestamp}.mp4"
                
                filepath = save_video(video, filename)
                
                if filepath:
                    st.info(f"✅ Video saved to: {filepath}")
                
                # Display video
                st.video(str(filepath)) if filepath else st.warning("Could not display video")

st.divider()

# Footer
st.markdown("""
---
**Benjini Video Tool** | Powered by Wan 2.2 & Streamlit | 🚀 Open Source

For issues and updates, visit: https://github.com/miketony7109-bot/benjini-video-tool
""")
