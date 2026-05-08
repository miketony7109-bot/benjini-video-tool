"""
Benjini Video Tool - AI Text-to-Video Generator (Streamlit Cloud Edition)
A Streamlit app for generating videos from text prompts using optimized models
"""

import streamlit as st
import torch
import os
from pathlib import Path
from datetime import datetime
import logging
import numpy as np

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

# Custom CSS - Mobile optimized
st.markdown("""
    <style>
    .main {
        padding: 1rem;
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
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .header-subtitle {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# Create output directory
OUTPUT_DIR = Path("generated_videos")
OUTPUT_DIR.mkdir(exist_ok=True)

# Cache the model load to avoid reloading
@st.cache_resource
def load_model(model_choice):
    """Load the selected video generation model with caching"""
    try:
        st.info("🔄 Loading model... This may take a moment on first run.")
        
        if model_choice == "Stable Video Diffusion (Fast ⚡)":
            from diffusers import StableVideoDiffusionPipeline
            from diffusers.utils import load_image
            
            model_id = "stabilityai/stable-video-diffusion-img2vid-xt"
            
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                variant="fp16" if torch.cuda.is_available() else None
            )
            
        elif model_choice == "Text2Video-Zero (Lightweight)":
            from diffusers import TextToVideoSDPipeline
            
            model_id = "damo-viton-xl/Text2Video-Zero"
            
            pipe = TextToVideoSDPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        
        else:  # Default lightweight option
            from diffusers import AnimateDiffPipeline
            from diffusers.models import MotionAdapter
            
            model_id = "guoyww/animatediff-motion-adapter-v1-5-2"
            
            motion_adapter = MotionAdapter.from_pretrained(model_id)
            pipe = AnimateDiffPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                motion_adapter=motion_adapter,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            st.success("✅ Model loaded on GPU!")
        else:
            st.warning("⚠️ Using CPU (slower). Videos may take longer to generate.")
            pipe = pipe.to("cpu")
        
        # Enable memory optimization
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        
        return pipe
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        logger.error(f"Model loading error: {str(e)}")
        return None

def generate_video(pipe, prompt, negative_prompt, num_steps, guidance_scale, height, width):
    """Generate video from text prompt"""
    try:
        logger.info(f"Generating video with prompt: {prompt}")
        
        with st.spinner("🎬 Generating your video... Please wait."):
            # Simple inference - works with most pipelines
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=int(num_steps),
                guidance_scale=float(guidance_scale),
                height=int(height),
                width=int(width),
                generator=torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
            )
            
            # Extract video from output
            if hasattr(output, 'videos'):
                return output.videos[0]
            elif hasattr(output, 'frames'):
                return output.frames
            else:
                return output
                
    except Exception as e:
        st.error(f"❌ Error during generation: {str(e)}")
        logger.error(f"Generation error: {str(e)}")
        return None

def save_video(video_frames, filename):
    """Save video frames to MP4 file"""
    try:
        import cv2
        
        filepath = OUTPUT_DIR / filename
        
        # Convert video frames to numpy array
        if hasattr(video_frames, 'numpy'):
            frames = video_frames.numpy()
        elif isinstance(video_frames, list):
            frames = np.array([np.array(f) for f in video_frames])
        else:
            frames = np.array(video_frames)
        
        # Ensure frames are in the right shape
        if frames.ndim == 3:  # Single image
            frames = np.expand_dims(frames, 0)
        
        # Define codec and create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 8  # Lower FPS for compatibility
        
        if frames.ndim >= 3:
            height, width = frames[0].shape[:2]
            out = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))
            
            frame_count = 0
            for frame in frames:
                if frame.dtype in [np.float32, np.float64]:
                    frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                
                if frame.shape[2] == 3:  # RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                out.write(frame)
                frame_count += 1
            
            out.release()
            logger.info(f"Video saved: {frame_count} frames to {filepath}")
            return filepath
        
        return None
        
    except Exception as e:
        logger.error(f"Error saving video: {str(e)}")
        st.error(f"Could not save video: {str(e)}")
        return None

# Main UI
st.markdown('<div class="header-title">🎬 Benjini Video Tool</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Generate Videos from Text | Streamlit Cloud Edition</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    st.subheader("Model Selection")
    model_choice = st.radio(
        "Choose a model:",
        [
            "Stable Video Diffusion (Fast ⚡)",
            "Text2Video-Zero (Lightweight)",
            "AnimateDiff (Smooth)"
        ],
        index=0,
        help="Stable Video Diffusion is fastest for Streamlit Cloud"
    )
    
    st.divider()
    
    st.subheader("Video Parameters")
    height = st.selectbox("Video Height:", [480, 576, 720], index=1)
    width = st.selectbox("Video Width:", [768, 1024, 1280], index=1)
    
    num_inference_steps = st.slider("Quality Steps:", 20, 50, 30)
    guidance_scale = st.slider("Prompt Adherence:", 1.0, 10.0, 7.5)
    
    st.divider()
    
    st.subheader("About")
    st.info("""
    **Benjini Video Tool** generates videos from text.
    
    ⚡ **Streamlit Cloud Edition**
    - Uses lighter models
    - Faster generation
    - Mobile friendly
    
     **Tips:**
    - Be descriptive
    - Include style
    - First run may be slower
    
    📖 Learn more: https://github.com/miketony7109-bot/benjini-video-tool
    """)

# Main content
st.subheader("📝 Prompt")
prompt = st.text_area(
    "Enter your video prompt:",
    placeholder="e.g., A serene mountain landscape at sunset, cinematic quality, 4K",
    height=80
)

negative_prompt = st.text_area(
    "Negative prompt (optional):",
    placeholder="e.g., blurry, low quality, distorted, ugly",
    height=60
)

st.divider()

# Generation info
col1, col2 = st.columns(2)
with col1:
    st.metric("Resolution", f"{width}x{height}")
with col2:
    st.metric("Steps", num_inference_steps)

st.divider()

# Generate button
if st.button("🚀 Generate Video", use_container_width=True, key="generate_btn"):
    if not prompt.strip():
        st.error("❌ Please enter a prompt!")
    else:
        # Load model with caching
        pipe = load_model(model_choice)
        
        if pipe:
            # Generate video
            video = generate_video(
                pipe,
                prompt,
                negative_prompt,
                num_inference_steps,
                guidance_scale,
                height,
                width
            )
            
            if video is not None:
                st.success("✅ Video generated successfully!")
                
                # Save video
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"video_{timestamp}.mp4"
                
                filepath = save_video(video, filename)
                
                if filepath and filepath.exists():
                    st.info(f"✅ Video saved successfully!")
                    
                    # Display video
                    try:
                        with open(filepath, "rb") as f:
                            st.video(f)
                    except Exception as e:
                        st.warning(f"Could not display video: {str(e)}")
                        st.download_button(
                            label="📥 Download Video",
                            data=open(filepath, "rb"),
                            file_name=filename,
                            mime="video/mp4"
                        )
            else:
                st.error("❌ Video generation failed. Please try again or adjust parameters.")

st.divider()

# Footer
st.markdown("""
---
**Benjini Video Tool** | Streamlit Cloud Edition | 🚀 Open Source

For issues: https://github.com/miketony7109-bot/benjini-video-tool/issues
""")