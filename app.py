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
from PIL import Image

# Page config
st.set_page_config(page_title="Realistic Talking Head", page_icon="🎭", layout="wide")

st.markdown("""
<style>
    .stButton>button { background: #ff4b4b; color: white; border-radius: 30px; padding: 0.5rem 2rem; font-weight: bold; border: none; }
    .stButton>button:hover { background: #e60000; transform: scale(1.02); }
    .contact-info { background: #1e2a3a; padding: 1rem; border-radius: 12px; color: white; text-align: center; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 style="text-align:center;">🎭 Realistic AI Talking Head</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;">Upload a portrait photo – Wav2Lip will animate the lips naturally.</p>', unsafe_allow_html=True)

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
# Wav2Lip integration – download model and run inference
# ------------------------------------------------------------------

REPO_DIR = "wav2lip"
MODEL_PATH = os.path.join(REPO_DIR, "wav2lip_gan.pth")
CHECKPOINT_URL = "https://github.com/justinjohn0306/Wav2Lip/releases/download/Models/wav2lip_gan.pth"

def download_wav2lip():
    """Clone Wav2Lip repo and download pretrained model."""
    if not os.path.exists(REPO_DIR):
        st.info("⏳ Downloading Wav2Lip repository (first time only)...")
        subprocess.run(["git", "clone", "https://github.com/Rudrabha/Wav2Lip.git", REPO_DIR], check=True)
    if not os.path.exists(MODEL_PATH):
        st.info("⏳ Downloading pretrained model (≈300 MB, first time only)...")
        import requests
        response = requests.get(CHECKPOINT_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success("✅ Model downloaded.")

def run_wav2lip(face_path, audio_path, output_path):
    """Run Wav2Lip inference."""
    # Ensure required packages are installed
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", os.path.join(REPO_DIR, "requirements.txt")], check=True)
    cmd = [
        sys.executable, os.path.join(REPO_DIR, "inference.py"),
        "--checkpoint_path", MODEL_PATH,
        "--face", face_path,
        "--audio", audio_path,
        "--outfile", output_path
    ]
    # Run inference
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        st.error(f"Inference failed: {result.stderr}")
        return False
    return True

# ------------------------------------------------------------------
# Generate
# ------------------------------------------------------------------

if generate_btn:
    if uploaded_image is None:
        st.error("Please upload a portrait photo.")
    elif not script_text.strip():
        st.error("Please enter the text to speak.")
    else:
        with st.spinner("🎬 Generating – this will take 2‑5 minutes (first run downloads model)..."):
            try:
                # 1. Save uploaded image
                temp_dir = tempfile.mkdtemp()
                face_path = os.path.join(temp_dir, "face.jpg")
                with open(face_path, "wb") as f:
                    f.write(uploaded_image.getvalue())

                # 2. Generate speech audio
                audio_path = os.path.join(temp_dir, "speech.mp3")
                async def tts():
                    communicate = edge_tts.Communicate(script_text, selected_voice)
                    await communicate.save(audio_path)
                asyncio.run(tts())

                # 3. Ensure Wav2Lip is ready
                download_wav2lip()

                # 4. Run Wav2Lip
                output_path = os.path.join(temp_dir, "output.mp4")
                success = run_wav2lip(face_path, audio_path, output_path)

                if success and os.path.exists(output_path):
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
                    st.error("Video generation failed. Check logs for details.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.exception(e)
            finally:
                # Cleanup temporary files
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)

st.markdown("---")
st.caption("Powered by Wav2Lip – open‑source lip‑sync | Built by Gesner Deslandes, GlobalInternet.py")
