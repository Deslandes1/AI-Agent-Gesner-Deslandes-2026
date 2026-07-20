import streamlit as st
import os
import tempfile
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import moviepy.editor as mp
from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip, concatenate_videoclips
import base64
import uuid
import io
import asyncio
import edge_tts
import wave
import struct
import math

# Page config
st.set_page_config(page_title="Talking Head Video Generator", page_icon="🎭", layout="wide")

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

st.markdown('<h1 style="text-align:center;">🎭 AI Talking Head Generator</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;">Upload a portrait photo, type a script – and watch it come to life!</p>', unsafe_allow_html=True)

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    voice_lang = st.selectbox("Voice Language", ["en-US", "en-GB", "fr-FR", "es-ES", "pt-BR"])
    voice_gender = st.radio("Voice Gender", ["Female", "Male"])
    
    voice_map = {
        ("en-US", "Female"): "en-US-JennyNeural",
        ("en-US", "Male"): "en-US-GuyNeural",
        ("en-GB", "Female"): "en-GB-SoniaNeural",
        ("en-GB", "Male"): "en-GB-RyanNeural",
        ("fr-FR", "Female"): "fr-FR-DeniseNeural",
        ("fr-FR", "Male"): "fr-FR-HenriNeural",
        ("es-ES", "Female"): "es-ES-ElviraNeural",
        ("es-ES", "Male"): "es-ES-AlvaroNeural",
        ("pt-BR", "Female"): "pt-BR-FranciscaNeural",
        ("pt-BR", "Male"): "pt-BR-AntonioNeural",
    }
    selected_voice = voice_map.get((voice_lang, voice_gender), "en-US-JennyNeural")
    st.info(f"Voice: {selected_voice}")
    
    st.markdown("---")
    st.subheader("Animation")
    head_rotation = st.slider("Head sway (degrees)", 0, 5, 2, help="Subtle rotation for natural movement")
    zoom_speed = st.slider("Zoom speed", 0, 3, 1, help="Slow zoom in/out effect")
    mouth_amplitude = st.slider("Mouth opening sensitivity", 0.5, 2.0, 1.0, step=0.1)
    
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
col1, col2 = st.columns([1, 2])
with col1:
    uploaded_image = st.file_uploader("Upload a portrait photo", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        st.image(uploaded_image, caption="Input photo", use_column_width=True)

with col2:
    script_text = st.text_area("Enter the text to speak", height=200, 
                               placeholder="e.g. Hello, I am your AI assistant. This is a demo of our talking head generator.")
    generate_btn = st.button("🚀 Generate Talking Head Video", use_container_width=True)

video_placeholder = st.empty()

# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------

def get_audio_amplitude(audio_path, num_frames=100):
    """Extract amplitude envelope from audio to drive mouth movement."""
    try:
        audio_clip = mp.AudioFileClip(audio_path)
        framerate = audio_clip.fps if audio_clip.fps else 22050
        audio_array = audio_clip.to_soundarray(n_channels=1, fps=framerate)
        audio_array = audio_array.flatten()
        
        segment_len = len(audio_array) // num_frames
        if segment_len == 0:
            segment_len = 1
        amplitudes = []
        for i in range(num_frames):
            start = i * segment_len
            end = min(start + segment_len, len(audio_array))
            seg = audio_array[start:end]
            rms = np.sqrt(np.mean(seg**2)) if len(seg) > 0 else 0
            amplitudes.append(rms)
        
        max_amp = max(amplitudes) if max(amplitudes) > 0 else 1
        amplitudes = [a / max_amp for a in amplitudes]
        return amplitudes, audio_clip.duration
    except Exception as e:
        st.warning(f"Audio analysis failed: {e}. Using fallback.")
        return [0.5] * num_frames, 5.0

def detect_face_region(image_path):
    """
    Detect face using Haar cascade if available; fallback to heuristic.
    Returns (mouth_x, mouth_y, mouth_w, mouth_h) or (center region) on fallback.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Try to load cascade
    cascade = None
    try:
        # Try built-in path
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        cascade = cv2.CascadeClassifier(cascade_path)
    except AttributeError:
        # Fallback: try to load from local file if present (we'll not rely on it)
        pass
    
    faces = []
    if cascade is not None and not cascade.empty():
        faces = cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        # No face detected: assume face is in the upper-center area
        st.warning("No face detected. Using center of image for mouth.")
        # Assume mouth is at roughly 60% of height, 20-80% of width
        mouth_x = int(w * 0.25)
        mouth_y = int(h * 0.55)
        mouth_w = int(w * 0.5)
        mouth_h = int(h * 0.15)
        return (mouth_x, mouth_y, mouth_w, mouth_h), None, gray
    
    (x, y, w_face, h_face) = faces[0]
    # Mouth region: lower third of face, centered
    mouth_x = x + int(w_face * 0.2)
    mouth_y = y + int(h_face * 0.6)
    mouth_w = int(w_face * 0.6)
    mouth_h = int(h_face * 0.2)
    return (mouth_x, mouth_y, mouth_w, mouth_h), (x, y, w_face, h_face), gray

def create_mouth_images(mouth_width, mouth_height, open_ratio=0.8):
    """Generate two mouth images: closed (thin line) and open (ellipse)."""
    # Closed mouth: a thin line
    closed_img = Image.new('RGBA', (mouth_width, mouth_height), (0,0,0,0))
    draw = ImageDraw.Draw(closed_img)
    draw.line((10, mouth_height//2, mouth_width-10, mouth_height//2), fill=(255,200,150,255), width=3)
    
    # Open mouth: an ellipse
    open_img = Image.new('RGBA', (mouth_width, mouth_height), (0,0,0,0))
    draw = ImageDraw.Draw(open_img)
    draw.ellipse((5, 5, mouth_width-5, mouth_height-5), fill=(255,180,130,255), outline=(200,100,50,255), width=2)
    return closed_img, open_img

def generate_talking_video(image_path, script, voice, output_path, music_path=None,
                           head_sway=2, zoom_speed=1, mouth_amp=1.0):
    """
    Main video generation:
    - Generate audio with edge-tts
    - Extract amplitude envelope
    - Create frames with mouth sync and subtle head motion
    - Compose with music
    """
    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, "speech.mp3")
    
    # 1. Generate speech audio
    async def tts():
        communicate = edge_tts.Communicate(script, voice)
        await communicate.save(audio_path)
    asyncio.run(tts())
    
    # 2. Load image and detect face
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    # Save temp image for OpenCV
    temp_img_path = os.path.join(temp_dir, "input.jpg")
    img.save(temp_img_path)
    
    mouth_region, face_rect, gray = detect_face_region(temp_img_path)
    if mouth_region is None:
        st.warning("Could not detect face. Using center of image for mouth.")
        h, w = img_np.shape[:2]
        mouth_x = w//2 - 100
        mouth_y = h//2 + 50
        mouth_w = 200
        mouth_h = 80
    else:
        mouth_x, mouth_y, mouth_w, mouth_h = mouth_region
    
    # 3. Get audio amplitude
    amplitudes, audio_duration = get_audio_amplitude(audio_path, num_frames=100)
    frame_count = len(amplitudes)
    frame_duration = audio_duration / frame_count if frame_count > 0 else 0.1
    
    # 4. Prepare mouth images
    closed_mouth, open_mouth = create_mouth_images(mouth_w, mouth_h)
    
    # 5. Generate frames with mouth sync
    frames = []
    for i, amp in enumerate(amplitudes):
        threshold = 0.3 * mouth_amp
        mouth_img = open_mouth if amp > threshold else closed_mouth
        
        frame = img.copy().convert('RGBA')
        frame.paste(mouth_img, (mouth_x, mouth_y), mouth_img)
        
        # Head sway
        if head_sway > 0:
            angle = head_sway * math.sin(i / frame_count * 2 * math.pi * 2)
            frame = frame.rotate(angle, center=(frame.width//2, frame.height//2), resample=Image.BICUBIC, fillcolor=(0,0,0,0))
        
        # Zoom effect
        if zoom_speed > 0:
            zoom_factor = 1 + 0.02 * math.sin(i / frame_count * 2 * math.pi) * zoom_speed
            new_size = (int(frame.width * zoom_factor), int(frame.height * zoom_factor))
            frame = frame.resize(new_size, Image.Resampling.LANCZOS)
            # Center crop to original size
            left = (frame.width - img.width) // 2
            top = (frame.height - img.height) // 2
            right = left + img.width
            bottom = top + img.height
            frame = frame.crop((left, top, right, bottom))
        
        frames.append(np.array(frame))
    
    # 6. Create video clip from frames
    video_clip = mp.ImageSequenceClip(frames, fps=1/frame_duration) if frame_duration > 0 else mp.ImageSequenceClip(frames, fps=24)
    audio_clip = mp.AudioFileClip(audio_path)
    video_clip = video_clip.set_audio(audio_clip)
    
    # 7. Add background music if provided
    if music_path and os.path.exists(music_path):
        bg_music = mp.AudioFileClip(music_path)
        if bg_music.duration < video_clip.duration:
            bg_music = bg_music.loop(duration=video_clip.duration)
        else:
            bg_music = bg_music.subclip(0, video_clip.duration)
        bg_music = bg_music.volumex(0.3)
        final_audio = mp.CompositeAudioClip([audio_clip, bg_music])
        video_clip = video_clip.set_audio(final_audio)
    
    # 8. Write output
    video_clip.write_videofile(output_path, fps=24, codec='libx264', audio_codec='aac', 
                               temp_audiofile=os.path.join(temp_dir, 'temp_audio.m4a'), remove_temp=True,
                               verbose=False, logger=None)
    
    # Cleanup
    for f in os.listdir(temp_dir):
        try:
            os.remove(os.path.join(temp_dir, f))
        except:
            pass
    os.rmdir(temp_dir)
    return output_path

# -------------------------------------------------------------------
# MAIN GENERATE LOGIC
# -------------------------------------------------------------------

if generate_btn:
    if uploaded_image is None:
        st.error("Please upload a portrait photo.")
    elif not script_text.strip():
        st.error("Please enter the text to speak.")
    else:
        with st.spinner("Generating talking head video... This may take a minute."):
            try:
                # Save uploaded image
                temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                temp_image.write(uploaded_image.getvalue())
                temp_image.close()
                
                # Handle music
                music_path = None
                if music_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                        tmp.write(music_file.read())
                        music_path = tmp.name
                
                # Generate video
                out_dir = tempfile.mkdtemp()
                out_path = os.path.join(out_dir, f"talking_head_{uuid.uuid4().hex[:8]}.mp4")
                
                generate_talking_video(
                    temp_image.name,
                    script_text,
                    selected_voice,
                    out_path,
                    music_path,
                    head_rotation,
                    zoom_speed,
                    mouth_amplitude
                )
                
                # Display and download
                with open(out_path, "rb") as f:
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
                    file_name="talking_head_video.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
                
                # Cleanup temp files
                try:
                    os.remove(temp_image.name)
                except:
                    pass
                if music_path and os.path.exists(music_path):
                    try:
                        os.remove(music_path)
                    except:
                        pass
                for f in os.listdir(out_dir):
                    try:
                        os.remove(os.path.join(out_dir, f))
                    except:
                        pass
                os.rmdir(out_dir)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.exception(e)

st.markdown("---")
st.caption("Built with ❤️ by Gesner Deslandes | GlobalInternet.py")
