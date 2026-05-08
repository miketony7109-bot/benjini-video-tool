"""
Benjini Video Tool - Using Replicate API
Simple Streamlit app that generates videos using Replicate's GPU infrastructure
"""

import streamlit as st
import replicate
import os
from pathlib import Path

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
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-title">🎬 Benjini Video Tool</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">AI Text-to-Video Generator | Powered by Replicate</p>', unsafe_allow_html=True)

st.divider()

# API Key setup
st.subheader("🔑 Setup")
st.info("""
**How to get your API key:**
1. Go to https://replicate.com/account/api-tokens
2. Create a new token
3. Paste it below (or set REPLICATE_API_TOKEN environment variable)
""")

api_key = st.text_input("Enter your Replicate API Key:", type="password", placeholder="r8_...")

if api_key:
    os.environ["REPLICATE_API_TOKEN"] = api_key
    st.success("✅ API Key set!")
else:
    st.warning("⚠️ Please enter your Replicate API key to continue")

st.divider()

# Main interface
st.subheader("📝 Create Your Video")

prompt = st.text_area(
    "Describe your video:",
    placeholder="e.g., A serene sunset over mountains, cinematic quality, 4K",
    height=100
)

col1, col2 = st.columns(2)
with col1:
    model_choice = st.selectbox(
        "Model:",
        [
            "Stable Video Diffusion",
            "ZeroScope",
            "ModelScope"
        ],
        index=0
    )

with col2:
    num_steps = st.slider("Quality (steps):", 20, 50, 30)

negative_prompt = st.text_area(
    "Negative prompt (optional):",
    placeholder="e.g., blurry, low quality, distorted",
    height=80
)

st.divider()

# Model mapping for Replicate
MODEL_MAP = {
    "Stable Video Diffusion": "stability-ai/stable-video-diffusion-img2vid-xt",
    "ZeroScope": "cjwbw/zeroscope-v2-576w",
    "ModelScope": "damo-viton-xl/modelscope-text-to-video-synthesis"
}

# Generate button
if st.button("🚀 Generate Video", use_container_width=True, type="primary"):
    if not api_key:
        st.error("❌ Please enter your Replicate API key first!")
    elif not prompt.strip():
        st.error("❌ Please enter a video prompt!")
    else:
        try:
            st.info("🎬 Generating your video...")
            st.info("⏳ This typically takes 1-5 minutes depending on the model and parameters")
            
            with st.spinner("🎥 Video generation in progress..."):
                model_id = MODEL_MAP[model_choice]
                
                # Call Replicate API
                output = replicate.run(
                    model_id,
                    input={
                        "prompt": prompt,
                        "negative_prompt": negative_prompt if negative_prompt else None,
                        "num_inference_steps": int(num_steps),
                        "guidance_scale": 7.5,
                    }
                )
                
                # Handle different output types
                if isinstance(output, list) and len(output) > 0:
                    video_url = output[0]
                elif isinstance(output, str):
                    video_url = output
                else:
                    video_url = output
                
                st.success("✅ Video generated successfully!")
                
                # Display video
                if video_url:
                    st.video(video_url)
                    
                    st.info(f"**Video URL:** {video_url}")
                    st.markdown(f"[📥 Download Video]({video_url})")
                else:
                    st.warning("Video generated but no URL returned")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("""
            **Troubleshooting:**
            - Check your API key is correct
            - Ensure you have credits available
            - Try a simpler prompt
            - Visit: https://replicate.com/account/billing to check status
            """)

st.divider()

# Tips
with st.expander("💡 Tips for Better Results"):
    st.markdown("""
    **Prompt Tips:**
    - Be descriptive and specific
    - Include style: "cinematic", "anime", "oil painting"
    - Mention quality: "4K", "high quality", "detailed"
    - Specify camera movement: "slow pan left", "zoom in"
    
    **Negative Prompt Tips:**
    - "blurry, low quality, distorted"
    - "text, watermark, logo"
    - "artifacts, glitches"
    
    **Cost:**
    - Each generation costs ~$0.01-0.10 depending on model
    - Check your credits at https://replicate.com/account/billing
    """)

st.divider()

# Pricing info
st.info("""
**Replicate Pricing:**
- Sign up: Free credits to start
- Pay as you go: ~$0.001 per second of GPU time
- Typical video: $0.01 - $0.10
- Models vary in speed/cost

**Better Deal:** Get a free trial at replicate.com
""")

st.divider()

st.markdown("""
---
**Benjini Video Tool** | Powered by Replicate | [GitHub](https://github.com/miketony7109-bot/benjini-video-tool)
""")