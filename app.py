import streamlit as st
import os
import tempfile
import time
import asyncio
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
st.markdown('<div class="subtitle">Create a video from prompts or your own images</div>', unsafe_allow_html=True)

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Image source selection
    image_source = st.radio("Image Source", 
                            ["Generate with Hugging Face", 
                             "Generate with Replicate", 
                             "Upload your own images",
                             "Placeholder images (no API required)"],
                            help="Placeholder images work without any API key.")
    
    if image_source == "Generate with Hugging Face":
        hf_token = st.text_input("Hugging Face API Token", type="password", 
                                 placeholder="Get token from huggingface.co/settings/tokens",
                                 help="Free; get a token from huggingface.co")
        hf_model = st.selectbox("Model", ["runwayml/stable-diffusion-v1-5", "stabilityai/stable-diffusion-2-1"])
    elif image_source == "Generate with Replicate":
        replicate_key = st.text_input("Replicate API Key", type="password", 
                                      placeholder="Get key from replicate.com/account/api-tokens",
                                      help="Requires credits. See replicate.com/pricing")
        replicate_model = st.selectbox("Model", ["stability-ai/stable-diffusion-3.5-large", "black-forest-labs/FLUX.1-dev"])
    elif image_source == "Upload your own images":
        st.info("You will upload images in the main area.")
    else:
        st.info("Placeholder images will be generated with gradient backgrounds and your prompt text.")
    
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
st.caption("Enter one prompt per line. Each prompt will generate a slide.")
prompts_text = st.text_area("Prompts", height=200, 
                            placeholder="e.g.\nA modern money transfer app on a smartphone\nA family receiving money in Haiti\nA fast digital payment interface\nA happy customer using mobile money")
prompts = [p.strip() for p in prompts_text.split('\n') if p.strip()]

# If user uploads own images
uploaded_images = []
if image_source == "Upload your own images":
    uploaded_files = st.file_uploader("Upload images (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if uploaded_files:
        for f in uploaded_files:
            img = Image.open(f).convert('RGB')
            uploaded_images.append(img)

col1, col2, col3 = st.columns([1,1,1])
with col1:
    generate_btn = st.button("🎬 Generate Video", use_container_width=True)
with col2:
    clear_btn = st.button("🗑️ Clear Prompts", use_container_width=True)
    if clear_btn:
        st.session_state['prompts'] = ""

video_placeholder = st.empty()

# Helper functions
def generate_placeholder(prompt, size=(1920,1080)):
    """Generate a gradient placeholder image with prompt text."""
    # Create a colorful gradient
    img = Image.new('RGB', size)
    draw = ImageDraw.Draw(img)
    # Gradient from dark blue to purple
    for y in range(size[1]):
        ratio = y / size[1]
        r = int(20 + 50 * ratio)
        g = int(40 + 30 * ratio)
        b = int(80 + 100 * ratio)
        draw.line([(0, y), (size[0], y)], fill=(r, g, b))
    # Draw prompt text
    try:
        font = ImageFont.truetype("Arial", 60)
    except:
        font = ImageFont.load_default()
    # Wrap text
    max_width = size[0] - 100
    lines = []
    words = prompt.split()
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
        draw.text((x_text, y_text), line, font=font, fill='white')
        y_text += bbox[3] + 10
    return img

def generate_image_hf(prompt, token, model):
    """Generate an image using Hugging Face inference API with retry."""
    if not token:
        return None
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": prompt}
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                return response.content
            elif response.status_code == 503:
                # Model loading, wait and retry
                time.sleep(5)
                continue
            else:
                st.warning(f"HF attempt {attempt+1} failed: {response.status_code}")
                time.sleep(2)
        except Exception as e:
            st.warning(f"HF attempt {attempt+1} error: {e}")
            time.sleep(2)
    return None

def generate_image_replicate(prompt, token, model):
    """Generate an image using Replicate."""
    if not token:
        return None
    import replicate
    os.environ["REPLICATE_API_TOKEN"] = token
    try:
        if "stable-diffusion" in model or "FLUX" in model:
            input = {"prompt": prompt, "aspect_ratio": "16:9", "output_format": "png"}
        else:
            input = {"prompt": prompt, "num_inference_steps": 30, "guidance_scale": 7.5}
        output = replicate.run(model, input=input)
        if isinstance(output, list):
            return output[0]
        elif isinstance(output, str):
            return output
        else:
            return None
    except Exception as e:
        st.warning(f"Replicate error: {e}")
        return None

def create_slide_image(img, text=None, text_color="#FFFFFF", font_size=80, size=(1920,1080)):
    """Resize image and optionally add text overlay."""
    if img.size != size:
        img.thumbnail(size, Image.Resampling.LANCZOS)
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
            for dx in (-2,0,2):
                for dy in (-2,0,2):
                    draw.text((x_text+dx, y_text+dy), line, font=font, fill='black')
            draw.text((x_text, y_text), line, font=font, fill=text_color)
            y_text += bbox[3] + 10
    return img

def create_video(image_list, durations, output_path, music_path=None, transition_duration=1, size=(1920,1080)):
    clips = []
    for idx, img in enumerate(image_list):
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
        audio = audio.volumex(0.3)
        video = video.set_audio(audio)
    video.write_videofile(output_path, fps=24, codec='libx264', audio_codec='aac',
                          temp_audiofile='temp_audio.m4a', remove_temp=True, verbose=False, logger=None)
    return output_path

# Generate logic
if generate_btn:
    images = []
    
    if image_source == "Upload your own images":
        if not uploaded_images:
            st.error("Please upload at least one image.")
        else:
            images = [img.copy() for img in uploaded_images]
    elif image_source == "Placeholder images (no API required)":
        if not prompts:
            st.error("Please enter at least one prompt.")
        else:
            with st.spinner("Generating placeholder images..."):
                for i, prompt in enumerate(prompts):
                    img = generate_placeholder(prompt, size=video_size)
                    text_overlay = product_name if i == 0 and product_name else None
                    slide_img = create_slide_image(img, text=text_overlay,
                                                   text_color=title_color,
                                                   font_size=title_font_size,
                                                   size=video_size)
                    images.append(slide_img)
    else:
        # HF or Replicate generation
        if not prompts:
            st.error("Please enter at least one prompt.")
        else:
            # Check if API key is provided
            if image_source == "Generate with Hugging Face" and not hf_token:
                st.error("Hugging Face token is required.")
            elif image_source == "Generate with Replicate" and not replicate_key:
                st.error("Replicate token is required.")
            else:
                with st.spinner("Generating images... (this may take a while)"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    for i, prompt in enumerate(prompts):
                        status_text.text(f"Generating image {i+1}/{len(prompts)}: {prompt[:30]}...")
                        img_data = None
                        if image_source == "Generate with Hugging Face":
                            img_bytes = generate_image_hf(prompt, hf_token, hf_model)
                            if img_bytes:
                                try:
                                    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                                    img_data = img
                                except:
                                    pass
                        else:  # Replicate
                            img_url = generate_image_replicate(prompt, replicate_key, replicate_model)
                            if img_url:
                                try:
                                    resp = requests.get(img_url, timeout=30)
                                    if resp.status_code == 200:
                                        img = Image.open(io.BytesIO(resp.content)).convert('RGB')
                                        img_data = img
                                except:
                                    pass
                        # If generation failed, fallback to placeholder
                        if img_data is None:
                            st.warning(f"Using placeholder for prompt {i+1} due to API failure.")
                            img_data = generate_placeholder(prompt, size=video_size)
                        # Add title overlay on first slide
                        text_overlay = product_name if i == 0 and product_name else None
                        slide_img = create_slide_image(img_data, text=text_overlay,
                                                       text_color=title_color,
                                                       font_size=title_font_size,
                                                       size=video_size)
                        images.append(slide_img)
                        progress_bar.progress((i+1)/len(prompts))
                    status_text.empty()
                    progress_bar.empty()
    
    if images:
        # Build video
        with st.spinner("Composing video..."):
            durations = [duration_per_image] * len(images)
            temp_dir = tempfile.mkdtemp()
            video_path = os.path.join(temp_dir, f"output_{uuid.uuid4().hex[:8]}.mp4")
            music_path = None
            if music_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp.write(music_file.read())
                    music_path = tmp.name
            create_video(images, durations, video_path, music_path, transition_duration, video_size)
            
            # Display & download
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
            # Cleanup
            for f in os.listdir(temp_dir):
                try:
                    os.remove(os.path.join(temp_dir, f))
                except:
                    pass
            os.rmdir(temp_dir)
            if music_path and os.path.exists(music_path):
                os.remove(music_path)
    else:
        st.error("No images generated. Please check your prompts, API keys, or uploaded images.")

st.markdown("---")
st.caption("Built with ❤️ by Gesner Deslandes | GlobalInternet.py")
