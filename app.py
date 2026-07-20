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
import mediapipe as mp

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
st.markdown('<p style="text-align:center;">Upload a portrait – choose between realistic Wav2Lip or fast face‑warping.</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    method = st.radio("Method", ["Face‑warping (fast, no download)", "Wav2Lip (realistic, heavy)"],
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
# MediaPipe Face Mesh – get mouth landmarks
# ------------------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

def get_mouth_landmarks(image_path):
    """Detect face landmarks using MediaPipe and return mouth outer contour points."""
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0]
    h, w, _ = image.shape
    # Mouth outer contour indices (0-11 for outer lip)
    mouth_indices = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 325, 375, 402, 405, 415]
    # Actually we just need the outer lip contour (like dlib's outer mouth)
    # We'll use a simpler set: 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308
    # But for warping we need the convex hull of the mouth
    # Let's get all mouth points (outer and inner)
    mouth_points = []
    for idx in [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]:
        x = int(landmarks.landmark[idx].x * w)
        y = int(landmarks.landmark[idx].y * h)
        mouth_points.append((x, y))
    # Add top lip and bottom lip for better coverage
    return np.array(mouth_points, dtype=np.int32)

def warp_mouth_mediapipe(image, mouth_pts, open_ratio=0.5):
    """
    Warp the mouth region using affine transforms on the convex hull.
    """
    if mouth_pts is None or len(mouth_pts) < 6:
        return image
    # Compute bounding box of mouth
    x_min = max(0, np.min(mouth_pts[:, 0]) - 5)
    x_max = min(image.shape[1], np.max(mouth_pts[:, 0]) + 5)
    y_min = max(0, np.min(mouth_pts[:, 1]) - 5)
    y_max = min(image.shape[0], np.max(mouth_pts[:, 1]) + 5)
    if x_max - x_min < 10 or y_max - y_min < 10:
        return image
    # Crop mouth ROI
    mouth_roi = image[y_min:y_max, x_min:x_max]
    h, w = mouth_roi.shape[:2]
    # Stretch vertically based on open_ratio (0=closed, 1=full open)
    new_h = int(h * (0.3 + 0.7 * open_ratio))
    if new_h < 1:
        new_h = 1
    stretched = cv2.resize(mouth_roi, (w, new_h), interpolation=cv2.INTER_LINEAR)
    # Place back, centered vertically
    y_offset = (h - new_h) // 2
    result = image.copy()
    result[y_min:y_min+new_h, x_min:x_max] = stretched[:new_h, :]
    return result

# ------------------------------------------------------------------
# Wav2Lip – keep as optional (with proper import handling)
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
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success("✅ Model ready.")

def run_wav2lip(face_path, audio_path, output_path):
    """Run Wav2Lip inference – install dependencies on the fly."""
    # Ensure dependencies are installed
    try:
        import face_recognition
        import librosa
        import tensorflow
    except ImportError:
        st.info("Installing Wav2Lip dependencies... (this may take a minute)")
        subprocess.run([sys.executable, "-m", "pip", "install", "face_recognition", "librosa", "tensorflow", "--no-cache-dir"], check=True)
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
        st.error(f"Wav2Lip error: {result.stderr}")
        return False
    return True

# ------------------------------------------------------------------
# MAIN GENERATE
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

                # 1. Generate speech audio
                audio_path = os.path.join(temp_dir, "speech.mp3")
                async def tts():
                    communicate = edge_tts.Communicate(script_text, selected_voice)
                    await communicate.save(audio_path)
                asyncio.run(tts())

                # 2. Choose method
                output_path = os.path.join(temp_dir, "output.mp4")
                if method == "Wav2Lip (realistic, heavy)":
                    download_wav2lip()
                    success = run_wav2lip(face_path, audio_path, output_path)
                    if not success:
                        st.warning("Wav2Lip failed. Falling back to fast face‑warping.")
                        method = "Face‑warping (fast, no download)"

                if method == "Face‑warping (fast, no download)":
                    st.info("Using fast face‑warping with MediaPipe landmarks.")
                    # Get audio amplitude
                    import moviepy.editor as mp
                    audio_clip = mp.AudioFileClip(audio_path)
                    framerate = audio_clip.fps if audio_clip.fps else 22050
                    audio_array = audio_clip.to_soundarray(n_channels=1, fps=framerate)
                    audio_array = audio_array.flatten()
                    num_frames = 60
                    segment_len = max(1, len(audio_array) // num_frames)
                    amplitudes = []
                    for i in range(num_frames):
                        start = i * segment_len
                        end = min(start + segment_len, len(audio_array))
                        seg = audio_array[start:end]
                        rms = np.sqrt(np.mean(seg**2)) if len(seg) > 0 else 0
                        amplitudes.append(rms)
                    max_amp = max(amplitudes) if max(amplitudes) > 0 else 1
                    amplitudes = [a / max_amp for a in amplitudes]

                    # Load image and detect mouth landmarks
                    img = cv2.imread(face_path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    mouth_pts = get_mouth_landmarks(face_path)
                    if mouth_pts is None:
                        st.warning("No face landmarks detected. Using simple mouth overlay.")
                        # fallback overlay
                        pil_img = Image.open(face_path).convert('RGBA')
                        frames = []
                        for amp in amplitudes:
                            frame = pil_img.copy()
                            draw = ImageDraw.Draw(frame)
                            w, h = frame.size
                            mouth_x = w//2 - 100
                            mouth_y = h//2 + 80
                            open_ratio = 0.2 + 0.8 * amp
                            if open_ratio > 0.5:
                                draw.ellipse((mouth_x, mouth_y, mouth_x+200, mouth_y+80), fill=(255,180,130,255), outline=(200,100,50,255))
                            else:
                                draw.line((mouth_x+20, mouth_y+40, mouth_x+180, mouth_y+40), fill=(255,200,150,255), width=6)
                            frames.append(np.array(frame))
                    else:
                        frames = []
                        for amp in amplitudes:
                            open_ratio = 0.2 + 0.8 * amp
                            warped = warp_mouth_mediapipe(img_rgb, mouth_pts, open_ratio)
                            frames.append(warped)
                    # Create video
                    clips = [mp.ImageClip(frame).set_duration(1/24) for frame in frames]
                    video = mp.concatenate_videoclips(clips, method="chain")
                    audio = mp.AudioFileClip(audio_path)
                    video = video.set_audio(audio)
                    video.write_videofile(output_path, fps=24, codec='libx264', audio_codec='aac', verbose=False, logger=None)

                # Display and download
                if os.path.exists(output_path):
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
                else:
                    st.error("Video generation failed.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.exception(e)
            finally:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)

st.markdown("---")
st.caption("Built by Gesner Deslandes | GlobalInternet.py")
