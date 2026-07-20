import streamlit as st
import os
import tempfile
import subprocess
import sys
import asyncio
import edge_tts
import numpy as np
import cv2
import base64
import uuid
import shutil
import math
from PIL import Image, ImageDraw, ImageFont
import dlib
import face_recognition

# Page config
st.set_page_config(page_title="Realistic AI Talking Head", page_icon="🎭", layout="wide")

st.markdown("""
<style>
    .stButton>button { background: #ff4b4b; color: white; border-radius: 30px; padding: 0.5rem 2rem; font-weight: bold; border: none; }
    .stButton>button:hover { background: #e60000; transform: scale(1.02); }
    .contact-info { background: #1e2a3a; padding: 1rem; border-radius: 12px; color: white; text-align: center; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 style="text-align:center;">🎭 AI Talking Head Generator</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;">Upload a portrait – Wav2Lip (realistic) or fallback (face‑warping).</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    method = st.radio("Method", ["Wav2Lip (realistic, heavy)", "Face‑warping (light, faster)"],
                      help="Wav2Lip requires a one‑time model download (≈300MB).")
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
# FALLBACK: Face-warping using facial landmarks
# ------------------------------------------------------------------
def get_landmarks(image_path):
    """Detect face landmarks using face_recognition (dlib)."""
    image = face_recognition.load_image_file(image_path)
    face_landmarks = face_recognition.face_landmarks(image)
    if not face_landmarks:
        return None
    return face_landmarks[0]

def warp_mouth(image, landmarks, open_ratio=0.5):
    """
    Warp the mouth region to simulate open/closed mouth.
    Uses affine transformation on the mouth points.
    """
    if landmarks is None:
        return image
    # Get mouth outer points
    mouth_points = landmarks['top_lip'] + landmarks['bottom_lip']
    if len(mouth_points) < 6:
        return image
    # Compute mouth center and bounding box
    mouth_pts = np.array(mouth_points)
    cx = int(np.mean(mouth_pts[:, 0]))
    cy = int(np.mean(mouth_pts[:, 1]))
    # Determine mouth height (distance between upper and lower lip)
    top = min([p[1] for p in landmarks['top_lip']])
    bottom = max([p[1] for p in landmarks['bottom_lip']])
    mouth_height = bottom - top
    # Scale mouth height based on open_ratio (0=closed, 1=full open)
    new_height = int(mouth_height * (0.2 + 0.8 * open_ratio))
    # Create a mask of the mouth region
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(mouth_points)], 255)
    # Crop mouth region
    x_min = min(mouth_pts[:, 0]) - 10
    x_max = max(mouth_pts[:, 0]) + 10
    y_min = top - 5
    y_max = bottom + 5
    if x_min < 0: x_min = 0
    if y_min < 0: y_min = 0
    if x_max > image.shape[1]: x_max = image.shape[1]
    if y_max > image.shape[0]: y_max = image.shape[0]
    mouth_roi = image[y_min:y_max, x_min:x_max]
    # Resize vertically to simulate opening
    h, w = mouth_roi.shape[:2]
    if h < 2 or w < 2:
        return image
    # Stretch vertically
    new_h = int(h * (0.5 + 0.5 * open_ratio))
    if new_h < 1:
        new_h = 1
    stretched = cv2.resize(mouth_roi, (w, new_h), interpolation=cv2.INTER_LINEAR)
    # Place back in original position (centered)
    y_offset = (h - new_h) // 2
    if y_offset < 0: y_offset = 0
    result = image.copy()
    result[y_min:y_min+new_h, x_min:x_max] = stretched[:new_h, :]
    return result

# ------------------------------------------------------------------
# Wav2Lip integration (if selected)
# ------------------------------------------------------------------
REPO_DIR = "wav2lip"
MODEL_PATH = os.path.join(REPO_DIR, "wav2lip_gan.pth")
CHECKPOINT_URL = "https://github.com/justinjohn0306/Wav2Lip/releases/download/Models/wav2lip_gan.pth"

def download_wav2lip():
    if not os.path.exists(REPO_DIR):
        st.info("⏳ Cloning Wav2Lip repository...")
        subprocess.run(["git", "clone", "https://github.com/Rudrabha/Wav2Lip.git", REPO_DIR], check=True)
    if not os.path.exists(MODEL_PATH):
        st.info("⏳ Downloading Wav2Lip model (≈300MB) – this will take a few minutes...")
        import requests
        response = requests.get(CHECKPOINT_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            total = int(response.headers.get('content-length', 0))
            progress = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                progress += len(chunk)
                if total > 0:
                    st.progress(min(progress / total, 1.0))
        st.success("✅ Model ready.")

def run_wav2lip(face_path, audio_path, output_path):
    """Run Wav2Lip inference – pre-install dependencies first."""
    # Install required packages (skip if already installed)
    try:
        import face_recognition
        import librosa
        import tensorflow
    except ImportError:
        st.warning("Installing Wav2Lip dependencies... (this may take a minute)")
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "face_recognition", "librosa", "tensorflow"], check=True)
    # Install the repo's requirements
    req_file = os.path.join(REPO_DIR, "requirements.txt")
    if os.path.exists(req_file):
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_file, "--no-cache-dir"], check=True)
    cmd = [
        sys.executable, os.path.join(REPO_DIR, "inference.py"),
        "--checkpoint_path", MODEL_PATH,
        "--face", face_path,
        "--audio", audio_path,
        "--outfile", output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        st.error(f"Inference error: {result.stderr}")
        return False
    return True

# ------------------------------------------------------------------
# MAIN GENERATE LOGIC
# ------------------------------------------------------------------
if generate_btn:
    if uploaded_image is None:
        st.error("Please upload a portrait photo.")
    elif not script_text.strip():
        st.error("Please enter the text to speak.")
    else:
        with st.spinner("🎬 Generating... (this may take a few minutes)"):
            try:
                temp_dir = tempfile.mkdtemp()
                face_path = os.path.join(temp_dir, "face.jpg")
                with open(face_path, "wb") as f:
                    f.write(uploaded_image.getvalue())

                # 1. Generate audio
                audio_path = os.path.join(temp_dir, "speech.mp3")
                async def tts():
                    communicate = edge_tts.Communicate(script_text, selected_voice)
                    await communicate.save(audio_path)
                asyncio.run(tts())

                # 2. Choose method
                if method == "Wav2Lip (realistic, heavy)":
                    download_wav2lip()
                    output_path = os.path.join(temp_dir, "output.mp4")
                    success = run_wav2lip(face_path, audio_path, output_path)
                    if not success:
                        st.warning("Wav2Lip failed. Falling back to face-warping method.")
                        method = "Face-warping (light, faster)"

                if method == "Face-warping (light, faster)":
                    # Use fallback: animate mouth using landmarks
                    st.info("Using face-warping for lip sync...")
                    # Get audio amplitude
                    audio_clip = mp.AudioFileClip(audio_path)
                    framerate = audio_clip.fps if audio_clip.fps else 22050
                    audio_array = audio_clip.to_soundarray(n_channels=1, fps=framerate)
                    audio_array = audio_array.flatten()
                    num_frames = 60  # we'll generate 60 frames
                    segment_len = len(audio_array) // num_frames
                    amplitudes = []
                    for i in range(num_frames):
                        start = i * segment_len
                        end = start + segment_len
                        seg = audio_array[start:end]
                        rms = np.sqrt(np.mean(seg**2)) if len(seg) > 0 else 0
                        amplitudes.append(rms)
                    max_amp = max(amplitudes) if max(amplitudes) > 0 else 1
                    amplitudes = [a / max_amp for a in amplitudes]

                    # Load image and detect landmarks
                    img = cv2.imread(face_path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    landmarks = get_landmarks(face_path)
                    if landmarks is None:
                        st.warning("No face landmarks found. Using simple mouth overlay.")
                        # fallback to simple overlay (but we'll keep it)
                        from PIL import Image, ImageDraw, ImageFont
                        pil_img = Image.open(face_path).convert('RGBA')
                        # simple open/close using ellipse overlay
                        frames = []
                        for amp in amplitudes:
                            frame = pil_img.copy()
                            draw = ImageDraw.Draw(frame)
                            # Draw mouth at a fixed position (approximate)
                            w, h = frame.size
                            mouth_x = w//2 - 100
                            mouth_y = h//2 + 80
                            open_ratio = 0.2 + 0.8 * amp
                            if open_ratio > 0.5:
                                # Draw open mouth
                                draw.ellipse((mouth_x, mouth_y, mouth_x+200, mouth_y+80), fill=(255,180,130,255), outline=(200,100,50,255))
                            else:
                                # Draw closed mouth line
                                draw.line((mouth_x+20, mouth_y+40, mouth_x+180, mouth_y+40), fill=(255,200,150,255), width=6)
                            frames.append(np.array(frame))
                    else:
                        # Use warp_mouth
                        frames = []
                        for amp in amplitudes:
                            open_ratio = 0.2 + 0.8 * amp
                            warped = warp_mouth(img_rgb, landmarks, open_ratio)
                            frames.append(warped)
                    # Create video from frames
                    import moviepy.editor as mp
                    clips = [mp.ImageClip(frame).set_duration(1/24) for frame in frames]
                    video = mp.concatenate_videoclips(clips, method="chain")
                    audio = mp.AudioFileClip(audio_path)
                    video = video.set_audio(audio)
                    output_path = os.path.join(temp_dir, "output.mp4")
                    video.write_videofile(output_path, fps=24, codec='libx264', audio_codec='aac', verbose=False, logger=None)

                # Display video
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

                # Cleanup
                shutil.rmtree(temp_dir, ignore_errors=True)

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.exception(e)

st.markdown("---")
st.caption("Built by Gesner Deslandes | GlobalInternet.py")
