"""
Benjini Video Tool - AI Text-to-Video Generator (Streamlit Cloud Edition)
Simplified version with better error handling and fallback support
"""

import streamlit as st
import torch
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Benjini Video Tool",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 1rem; }
    .header-title { text-align: center; color: #667eea; font-size: 2rem; font-weight: bold; }
    .info-box { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; }
    </style>
""", unsafe_allow_html=True)

# Create output directory
OUTPUT_DIR = Path("generated_videos")
OUTPUT_DIR.mkdir(exist_ok=True)

st.markdown('<div class="header-title">🎬 Benjini Video Tool</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">Generate Videos from Text | Streamlit Cloud</p>', unsafe_allow_html=True)

st.divider()

# Check GPU availability
gpu_available = torch.cuda.is_available()
if gpu_available:
    st.success(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
else:
    st.warning("⚠️ GPU not available - using CPU (generation will be slower)")

st.divider()

# Model selection
st.subheader("⚙️ Settings")

col1, col2 = st.columns(2)
with col1:
    model_choice = st.selectbox(
        "Model:",
        ["Stable Video Diffusion", "Sketch to Video"],
        index=0
    )

with col2:
    inference_steps = st.slider("Quality Steps:", 20, 50, 30)

st.divider()

# Prompt input
st.subheader("📝 Create Your Video")
prompt = st.text_area(
    "Describe your video:",
    placeholder="e.g., A sunset over mountains, cinematic, 4K",
    height=80
)

negative_prompt = st.text_area(
    "Negative prompt (what to avoid):",
    placeholder="e.g., blurry, low quality",
    height=60
)

st.divider()

# Cache decorator for model loading
@st.cache_resource
def load_svd_model():
    """Load Stable Video Diffusion model"""
    try:
        from diffusers import StableVideoDiffusionPipeline
        import torch
        
        st.info("Loading model... Please wait.")
        
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16 if gpu_available else torch.float32,
            variant="fp16" if gpu_available else None
        )
        
        if gpu_available:
            pipe = pipe.to("cuda")
        
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        
        return pipe
        
    except Exception as e:
        logger.error(f"Error loading SVD model: {e}")
        st.error(f"Failed to load model: {str(e)}")
        return None

@st.cache_resource
def load_sketch_model():
    """Load Sketch to Video model as fallback"""
    try:
        from diffusers import ControlNetModel, StableVideoDiffusionPipeline
        import torch
        
        st.info("Loading Sketch model... Please wait.")
        
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid",
            torch_dtype=torch.float16 if gpu_available else torch.float32,
        )
        
        if gpu_available:
            pipe = pipe.to("cuda")
        
        return pipe
        
    except Exception as e:
        logger.error(f"Error loading Sketch model: {e}")
        st.error(f"Failed to load fallback model: {str(e)}")
        return None

# Generate button
if st.button("🚀 Generate Video", use_container_width=True, type="primary"):
    if not prompt.strip():
        st.error("❌ Please enter a prompt!")
    else:
        try:
            # Load appropriate model
            if model_choice == "Stable Video Diffusion":
                pipe = load_svd_model()
            else:
                pipe = load_sketch_model()
            
            if pipe is None:
                st.error("❌ Failed to load model. Please check the logs.")
            else:
                st.info(f"🎬 Generating video from: '{prompt}'")
                st.info(f"⏳ This may take 3-10 minutes depending on hardware...")
                
                # Simple generation call
                try:
                    with st.spinner("🎥 Generating video..."):
                        output = pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt if negative_prompt else None,
                            num_inference_steps=inference_steps,
                            guidance_scale=7.5,
                            generator=torch.Generator("cuda" if gpu_available else "cpu").manual_seed(42)
                        )
                    
                    # Get video from output
                    if hasattr(output, 'videos') and output.videos is not None:
                        video = output.videos[0]
                    elif hasattr(output, 'frames') and output.frames is not None:
                        video = output.frames
                    else:
                        video = output
                    
                    st.success("✅ Video generated successfully!")
                    
                    # Save and display
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filepath = OUTPUT_DIR / f"video_{timestamp}.mp4"
                    
                    # Try to display video
                    try:
                        if hasattr(video, 'save'):
                            video.save(str(filepath))
                        else:
                            import imageio
                            if hasattr(video, 'numpy'):
                                video_array = video.numpy()
                            else:
                                video_array = video
                            imageio.mimsave(str(filepath), video_array, fps=8)
                        
                        st.video(str(filepath))
                        
                        with open(filepath, "rb") as f:
                            st.download_button(
                                label="📥 Download Video",
                                data=f.read(),
                                file_name=f.name,
                                mime="video/mp4"
                            )
                    except Exception as e:
                        logger.error(f"Error saving/displaying video: {e}")
                        st.warning(f"Video generated but could not be displayed: {str(e)}")
                
                except Exception as gen_error:
                    logger.error(f"Generation error: {gen_error}")
                    st.error(f"❌ Generation failed: {str(gen_error)}")
                    st.info("Try reducing the inference steps or checking your GPU memory.")
        
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            st.error(f"❌ An unexpected error occurred: {str(e)}")
            st.info("Please try again or contact support if the issue persists.")

st.divider()

# Tips section
with st.expander("💡 Tips for Better Results"):
    st.markdown("""
    - **Be descriptive**: "A cinematic sunset over mountains with birds flying" works better than "sunset"
    - **Specify style**: Add "cinematic", "anime", "oil painting", etc.
    - **Mention quality**: "4K", "high quality", "detailed"
    - **Avoid conflicts**: Don't ask for contradictory things
    - **Use negative prompts**: Tell the AI what NOT to do
    
    **First run note**: Models are downloaded on first use (~5-10GB). This may take several minutes.
    """)

st.divider()

# Footer
st.markdown("""
<div style="text-align: center; color: #999; font-size: 0.9rem; margin-top: 2rem;">
<p>Benjini Video Tool | Powered by Stable Video Diffusion</p>
<p><a href="https://github.com/miketony7109-bot/benjini-video-tool">GitHub</a></p>
</div>
""", unsafe_allow_html=True)