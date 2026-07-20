import streamlit as st
import os
import tempfile
import time
import asyncio
import replicate
import requests
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import moviepy.editor as mp
from moviepy.editor import ImageSequenceClip, AudioFileClip, CompositeVideoClip, TextClip, concatenate_videoclips, ColorClip
import base64
import uuid
import io
import json

# Page config
st.set_page_config(
    page_title="AI Video Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { background: #f8f9fa; }
    .stButton>button {
        background: #ff4b4b;
        color: white;
        border-radius: 30px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: #e60000;
        transform: scale(1.02);
    }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e2a3a;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #555;
        margin-bottom: 2rem;
    }
    .contact-info {
        background: #1e2a3a;
        padding: 1rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🎬 AI Video Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Generate a video from AI‑created images using prompts</div>', unsafe_allow_html=True)

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    replicate_api_key = st.text_input("Replicate API Key", type="password", 
                                      placeholder="Enter your Replicate API key", 
                                      help="Get your key from replicate.com")
    if replicate_api_key:
        os.environ["REPLICATE_API_TOKEN"] = replicate_api_key
    
    st.markdown("---")
    model_choice = st.selectbox("Image Model", 
                                ["stability-ai/stable-diffusion-3.5-large", 
                                 "black-forest-labs/FLUX.1-dev",
                                 "stability-ai/stable-diffusion-2-1"])
    st.caption("Note: Some models may have a cost per generation.")
    
    st.markdown("---")
    st.subheader("Video Settings")
    duration_per_image = st.slider("Duration per image (seconds)", 1, 10, 3)
    transition_duration = st.slider("Transition duration (seconds)", 0, 2, 1)
    video_resolution = st.selectbox("Resolution", ["720p (1280x720)", "1080p (1920x1080)"])
    res_map = {"720p (1280x720)": (1280,720), "1080p (1920x1080)": (1920,1080)}
    video_size = res_map[video_resolution]
    
    st.markdown("---")
    st.subheader("Title Overlay")
    product_name = st.text_input("Product / Service Name", placeholder="e.g. Prisme Transfer Haiti")
    title_color = st.color_picker("Title Color", "#FFFFFF")
    title_font_size = st.slider("Title font size", 30, 150, 80)
    
    st.markdown("---")
    st.subheader("🎵 Background Music")
    music_file = st.file_uploader("Upload MP3/WAV (optional)", type=["mp3", "wav"])
    
    st.markdown("---")
    st.markdown("""
    <div class="contact-info">
        📞 (509) 4738-5663<br>
        📧 deslandes78@gmail.com
    </div>
    """, unsafe_allow_html=True)

# Main area
st.subheader("📝 Prompts for Images")
st.caption("Enter one prompt per line. Each prompt will generate a slide in the video.")
prompts_text = st.text_area("Prompts", height=200, 
                            placeholder="e.g.\nA modern money transfer app on a smartphone\nA family receiving money in Haiti\nA fast digital payment interface\nA happy customer using mobile money")
prompts = [p.strip() for p in prompts_text.split('\n') if p.strip()]

col1, col2, col3 = st.columns([1,1,1])
with col1:
    generate_btn = st.button("🎬 Generate Video", use_container_width=True)
with col2:
    clear_btn = st.button("🗑️ Clear Prompts", use_container_width=True)
    if clear_btn:
        st.session_state['prompts'] = ""

video_placeholder = st.empty()

# Helper functions
def generate_image(prompt, model, api_key):
    """Generate an image using Replicate and return the URL."""
    if not api_key:
        return None
    try:
        # Different models have different input schemas; we'll use common parameters
        if "stable-diffusion-3.5" in model:
            input = {
                "prompt": prompt,
                "aspect_ratio": "16:9",
                "output_format": "png",
                "output_quality": 90
            }
        elif "FLUX" in model:
            input = {
                "prompt": prompt,
                "aspect_ratio": "16:9",
                "output_format": "png"
            }
        else:
            input = {
                "prompt": prompt,
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "width": 1024,
                "height": 576  # 16:9
            }
        output = replicate.run(model, input=input)
        # output can be a list of URLs or a single URL
        if isinstance(output, list):
            return output[0]
        elif isinstance(output, str):
            return output
        else:
            st.error(f"Unexpected output format: {output}")
            return None
    except Exception as e:
        st.error(f"Failed to generate image: {e}")
        return None

def download_image(url):
    """Download an image from a URL and return a PIL Image."""
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content)).convert('RGB')
            return img
        else:
            st.error(f"Failed to download image: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Download error: {e}")
        return None

def create_slide_image(img, text=None, text_color="#FFFFFF", font_size=80, size=(1920,1080)):
    """Resize image to target size and optionally add text overlay."""
    if img.size != size:
        img.thumbnail(size, Image.Resampling.LANCZOS)
        # Create a new image with the target size and paste centered
        new_img = Image.new('RGB', size, (0,0,0))
        x = (size[0] - img.width) // 2
        y = (size[1] - img.height) // 2
        new_img.paste(img, (x, y))
        img = new_img
    if text:
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("Arial", font_size)
        except:
            font = ImageFont.load_default()
        # Wrap text to fit width
        max_width = size[0] - 100
        lines = []
        words = text.split()
        if words:
            line = ""
            for word in words:
                test_line = line + word + " "
                bbox = draw.textbbox((0,0), test_line, font=font)
                w = bbox[2] - bbox[0]
                if w <= max_width:
                    line = test_line
                else:
                    if line:
                        lines.append(line.strip())
                    line = word + " "
            if line:
                lines.append(line.strip())
        # Draw each line centered
        total_height = 0
        for line in lines:
            bbox = draw.textbbox((0,0), line, font=font)
            total_height += bbox[3] - bbox[1]
        total_height += (len(lines) - 1) * 10
        y_text = (size[1] - total_height) // 2
        for line in lines:
            bbox = draw.textbbox((0,0), line, font=font)
            w = bbox[2] - bbox[0]
            x_text = (size[0] - w) // 2
            # Draw with black stroke for readability
            for dx in (-2,0,2):
                for dy in (-2,0,2):
                    draw.text((x_text+dx, y_text+dy), line, font=font, fill='black')
            draw.text((x_text, y_text), line, font=font, fill=text_color)
            y_text += bbox[3] + 10
    return img

def create_video(image_list, durations, output_path, music_path=None, transition_duration=1, size=(1920,1080)):
    """Create a video from a list of images with crossfade transitions."""
    clips = []
    for idx, img in enumerate(image_list):
        # Convert PIL to numpy array for moviepy
        img_np = np.array(img)
        clip = mp.ImageClip(img_np).set_duration(durations[idx]).resize(size)
        if idx > 0 and transition_duration > 0:
            clip = clip.crossfadein(transition_duration)
        clips.append(clip)
    
    video = mp.concatenate_videoclips(clips, method="compose")
    if music_path and os.path.exists(music_path):
        audio = mp.AudioFileClip(music_path)
        if audio.duration < video.duration:
            audio = audio.loop(duration=video.duration)
        else:
            audio = audio.subclip(0, video.duration)
        audio = audio.volumex(0.3)  # lower volume
        video = video.set_audio(audio)
    
    video.write_videofile(output_path, fps=24, codec='libx264', audio_codec='aac', 
                          temp_audiofile='temp_audio.m4a', remove_temp=True, verbose=False, logger=None)
    return output_path

# Generate logic
if generate_btn:
    if not prompts:
        st.error("Please enter at least one prompt.")
    else:
        if not replicate_api_key:
            st.warning("No Replicate API key provided. Using placeholder images for demonstration (no AI generation).")
            # Use placeholder images (gradient colors) for demo
            use_demo = True
        else:
            use_demo = False
        
        with st.spinner("Generating images and composing video... This may take a few minutes."):
            images = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            if use_demo:
                # Generate placeholder images (gradient) with prompt text
                for i, prompt in enumerate(prompts):
                    status_text.text(f"Creating placeholder {i+1}/{len(prompts)}: {prompt[:30]}...")
                    img = Image.new('RGB', video_size, color=(20,40,80))
                    draw = ImageDraw.Draw(img)
                    try:
                        font = ImageFont.truetype("Arial", 40)
                    except:
                        font = ImageFont.load_default()
                    # Draw prompt on image
                    lines = prompt.split('\n')
                    y = 200
                    for line in lines:
                        draw.text((50, y), line, font=font, fill='white')
                        y += 50
                    images.append(img)
                    progress_bar.progress((i+1)/len(prompts))
            else:
                # Real generation via Replicate
                for i, prompt in enumerate(prompts):
                    status_text.text(f"Generating image {i+1}/{len(prompts)}: {prompt[:30]}...")
                    img_url = generate_image(prompt, model_choice, replicate_api_key)
                    if img_url:
                        pil_img = download_image(img_url)
                        if pil_img:
                            # Optionally add title overlay (first slide gets product name)
                            text_overlay = product_name if i == 0 and product_name else None
                            slide_img = create_slide_image(pil_img, text=text_overlay, 
                                                           text_color=title_color, 
                                                           font_size=title_font_size, 
                                                           size=video_size)
                            images.append(slide_img)
                        else:
                            st.error(f"Failed to download image for prompt: {prompt}")
                            break
                    else:
                        st.error(f"Failed to generate image for prompt: {prompt}")
                        break
                    progress_bar.progress((i+1)/len(prompts))
            
            if images:
                # Durations: first and last slightly longer? or all same
                durations = [duration_per_image] * len(images)
                # If a title overlay is present, maybe make first slide longer
                # Not needed; we can adjust.
                
                # Create temporary files
                temp_dir = tempfile.mkdtemp()
                video_path = os.path.join(temp_dir, f"output_{uuid.uuid4().hex[:8]}.mp4")
                
                # Handle music file
                music_path = None
                if music_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                        tmp.write(music_file.read())
                        music_path = tmp.name
                
                create_video(images, durations, video_path, music_path, transition_duration, video_size)
                
                # Read video and display
                with open(video_path, "rb") as f:
                    video_bytes = f.read()
                b64 = base64.b64encode(video_bytes).decode()
                video_html = f"""
                <video width="100%" controls autoplay>
                    <source src="data:video/mp4;base64,{b64}" type="video/mp4">
                </video>
                """
                video_placeholder.markdown(video_html, unsafe_allow_html=True)
                
                st.download_button(
                    label="⬇️ Download Video (MP4)",
                    data=video_bytes,
                    file_name="generated_video.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
                
                # Cleanup temp files
                for f in os.listdir(temp_dir):
                    try:
                        os.remove(os.path.join(temp_dir, f))
                    except:
                        pass
                os.rmdir(temp_dir)
                if music_path and os.path.exists(music_path):
                    os.remove(music_path)
            else:
                st.error("No images generated. Please check your prompts and API key.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ by Gesner Deslandes | GlobalInternet.py")
