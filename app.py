import streamlit as st
import os
import tempfile
import asyncio
import edge_tts
import numpy as np
import base64
import shutil
from PIL import Image, ImageDraw
import moviepy.editor as mp
import librosa

# Page config
st.set_page_config(page_title="AI Talking Head", page_icon="🎭", layout="wide")

st.markdown("""
<style>
    .stButton>button { background: #ff4b4b; color: white; border-radius: 30px; padding: 0.5rem 2rem; font-weight: bold; border: none; }
    .stButton>button:hover { background: #e60000; transform: scale(1.02); }
    .contact-info { background: #1e2a3a; padding: 1rem; border-radius: 12px; color: white; text-align: center; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 style="text-align:center;">🎭 AI Talking Head Generator</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;">Upload a portrait – the mouth moves naturally with the audio.</p>', unsafe_allow_html=True)

# Sidebar
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
    st.subheader("Mouth Position")
    offset_x = st.slider("Horizontal offset", -100, 100, 0, help="Move mouth left/right")
    offset_y = st.slider("Vertical offset", -100, 100, 20, help="Move mouth up/down")
    mouth_scale = st.slider("Mouth size", 0.5, 2.0, 1.0, step=0.1, help="Scale of the mouth")

    st.markdown("---")
    st.markdown("""
    <div class="contact-info">
        📞 (509) 4738-5663<br>
        📧 deslandes78@gmail.com
    </div>
    """, unsafe_allow_html=True)

# Main inputs
col1, col2 = st.columns([1, 2])
with col1:
    uploaded_image = st.file_uploader("Upload a portrait photo", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        st.image(uploaded_image, caption="Input photo", use_column_width=True)

with col2:
    script_text = st.text_area("Enter the text to speak", height=200,
                               placeholder="e.g. Welcome to GlobalInternet.py...")
    generate_btn = st.button("🚀 Generate Talking Head", use_container_width=True)

video_placeholder = st.empty()

# ------------------------------------------------------------------
# Core functions
# ------------------------------------------------------------------

def draw_mouth(draw, center_x, center_y, width, height, open_ratio=0.0, color=(200, 150, 130)):
    if open_ratio < 0.05:
        draw.line((center_x - width//2, center_y, center_x + width//2, center_y), fill=color, width=4)
        draw.line((center_x - width//2, center_y+2, center_x + width//2, center_y+2), fill=(100,70,60), width=2)
        return
    current_height = int(height * (0.2 + 0.8 * open_ratio))
    draw.ellipse(
        (center_x - width//2, center_y - current_height//2,
         center_x + width//2, center_y + current_height//2),
        fill=(50, 30, 20, 200)
    )
    draw.ellipse(
        (center_x - width//2 - 2, center_y - current_height//2 - 2,
         center_x + width//2 + 2, center_y + current_height//2 + 2),
        outline=(color[0], color[1], color[2], 255), width=3
    )
    if current_height > 10:
        draw.ellipse(
            (center_x - width//3, center_y - current_height//4,
             center_x + width//3, center_y + current_height//4),
            fill=(210, 160, 120, 150)
        )

def generate_video(image_path, script, voice, output_path, offset_x=0, offset_y=20, mouth_scale=1.0):
    # 1. Generate speech audio
    audio_path = tempfile.mktemp(suffix=".mp3")
    async def tts():
        communicate = edge_tts.Communicate(script, voice)
        await communicate.save(audio_path)
    asyncio.run(tts())

    # 2. Load image
    img = Image.open(image_path).convert('RGBA')
    w, h = img.size

    face_center_x = w // 2
    face_center_y = int(h * 0.35)
    mouth_x = face_center_x + offset_x
    mouth_y = face_center_y + int(h * 0.2) + offset_y
    base_width = int(w * 0.15 * mouth_scale)
    base_height = int(w * 0.06 * mouth_scale)

    # 3. Extract amplitude using librosa (robust)
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        # Envelope: RMS per frame of about 1/24 sec
        hop_length = int(sr / 24)  # one frame per video frame
        rms = librosa.feature.rms(y=y, frame_length=2*hop_length, hop_length=hop_length)[0]
        # Convert to list of amplitudes (0-1 normalized)
        max_rms = max(rms) if len(rms) > 0 else 1
        amplitudes = rms / max_rms if max_rms > 0 else np.ones_like(rms) * 0.5
    except Exception as e:
        # Fallback: generate a simple sine wave
        st.warning(f"Audio analysis failed: {e}. Using synthetic waveform.")
        duration = 5  # assume 5 seconds
        num_frames = int(duration * 24)
        t = np.linspace(0, duration, num_frames)
        amplitudes = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)  # 2 Hz
        amplitudes = np.clip(amplitudes, 0, 1)

    # Ensure we have at least 1 frame
    if len(amplitudes) == 0:
        amplitudes = np.array([0.5])

    # 4. Generate frames
    frames = []
    lip_color = (200, 150, 130)
    for amp in amplitudes:
        open_ratio = min(1.0, amp * 1.5)  # amplify a bit
        frame = img.copy()
        draw = ImageDraw.Draw(frame, 'RGBA')
        draw_mouth(draw, mouth_x, mouth_y, base_width, base_height, open_ratio, lip_color)
        frames.append(np.array(frame.convert('RGB')))

    # 5. Create video
    # If frames list is empty, create a single frame
    if not frames:
        frames.append(np.array(img.convert('RGB')))
    clips = [mp.ImageClip(frame).set_duration(1/24) for frame in frames]
    video = mp.concatenate_videoclips(clips, method="chain")
    audio = mp.AudioFileClip(audio_path)
    # If video duration is shorter than audio, loop or extend
    if video.duration < audio.duration:
        # Loop the video to match audio length
        video = video.loop(duration=audio.duration)
    elif video.duration > audio.duration:
        video = video.subclip(0, audio.duration)
    video = video.set_audio(audio)
    video.write_videofile(output_path, fps=24, codec='libx264', audio_codec='aac',
                          verbose=False, logger=None)
    # Cleanup
    os.remove(audio_path)
    return output_path

# ------------------------------------------------------------------
# MAIN GENERATE
# ------------------------------------------------------------------
if generate_btn:
    if uploaded_image is None:
        st.error("Please upload a portrait photo.")
    elif not script_text.strip():
        st.error("Please enter the text to speak.")
    else:
        with st.spinner("🎬 Generating video... (this may take a minute)"):
            try:
                temp_dir = tempfile.mkdtemp()
                image_path = os.path.join(temp_dir, "input.jpg")
                img = Image.open(uploaded_image).convert('RGB')
                img.save(image_path)
                output_path = os.path.join(temp_dir, "output.mp4")
                generate_video(
                    image_path,
                    script_text,
                    selected_voice,
                    output_path,
                    offset_x,
                    offset_y,
                    mouth_scale
                )
                with open(output_path, "rb") as f:
                    video_bytes = f.read()
                b64 = base64.b64encode(video_bytes).decode()
                video_html = f"""
                <video width="100%" controls autoplay>
                    <source src="data:video/mp4;base64,{b64}" type="video/mp4">
                </video>
                """
                video_placeholder.markdown(video_html, unsafe_allow_html=True)
                st.download_button("⬇️ Download Video (MP4)", video_bytes,
                                   file_name="talking_head.mp4", mime="video/mp4",
                                   use_container_width=True)
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.exception(e)

st.markdown("---")
st.caption("Built by Gesner Deslandes | GlobalInternet.py")
