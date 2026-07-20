import streamlit as st
import os
import tempfile
import asyncio
import edge_tts
import numpy as np
import base64
import uuid
import shutil
import math
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp
import moviepy.editor as mp_editor

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
st.markdown('<p style="text-align:center;">Upload a portrait – the app animates the mouth naturally using AI (no OpenCV needed).</p>', unsafe_allow_html=True)

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
# MediaPipe Face Mesh – get mouth landmarks (no cv2)
# ------------------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh

def get_mouth_points(image_pil):
    """Detect mouth contour using MediaPipe, return list of (x,y) points in pixel coords."""
    # Convert PIL to RGB numpy array (MediaPipe expects RGB)
    img_rgb = np.array(image_pil.convert('RGB'))
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0]
    h, w, _ = img_rgb.shape
    # We'll collect outer mouth points (indices from MediaPipe Face Mesh)
    # Use a comprehensive set that forms a closed loop
    mouth_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    points = []
    for idx in mouth_indices:
        x = int(landmarks.landmark[idx].x * w)
        y = int(landmarks.landmark[idx].y * h)
        points.append((x, y))
    # Also add inner lip points for better coverage
    inner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    for idx in inner:
        x = int(landmarks.landmark[idx].x * w)
        y = int(landmarks.landmark[idx].y * h)
        points.append((x, y))
    return np.array(points, dtype=np.int32)

def warp_mouth_pil(image_pil, mouth_pts, open_ratio=0.5):
    """
    Warp the mouth region vertically using an affine stretch.
    Returns a new PIL Image.
    """
    if mouth_pts is None or len(mouth_pts) < 6:
        return image_pil
    # Convert PIL to numpy for processing
    img_np = np.array(image_pil.convert('RGB'))
    h, w, _ = img_np.shape
    # Compute mouth bounding box
    x_min = max(0, np.min(mouth_pts[:, 0]) - 5)
    x_max = min(w, np.max(mouth_pts[:, 0]) + 5)
    y_min = max(0, np.min(mouth_pts[:, 1]) - 5)
    y_max = min(h, np.max(mouth_pts[:, 1]) + 5)
    if x_max - x_min < 10 or y_max - y_min < 10:
        return image_pil
    # Crop mouth ROI
    mouth_roi = img_np[y_min:y_max, x_min:x_max]
    roi_h, roi_w = mouth_roi.shape[:2]
    # Stretch vertically: new height = roi_h * (0.3 + 0.7 * open_ratio)
    new_h = int(roi_h * (0.3 + 0.7 * open_ratio))
    if new_h < 1:
        new_h = 1
    # Use PIL to resize (resize in PIL uses high-quality interpolation)
    roi_pil = Image.fromarray(mouth_roi)
    stretched = roi_pil.resize((roi_w, new_h), Image.Resampling.LANCZOS)
    # Paste back into the image (centered vertically)
    y_offset = (roi_h - new_h) // 2
    result_pil = image_pil.copy()
    result_pil.paste(stretched, (x_min, y_min + y_offset))
    return result_pil

# ------------------------------------------------------------------
# MAIN GENERATE
# ------------------------------------------------------------------
if generate_btn:
    if uploaded_image is None:
        st.error("Please upload a portrait photo.")
    elif not script_text.strip():
        st.error("Please enter the text to speak.")
    else:
        with st.spinner("🎬 Generating... (this may take a minute)"):
            try:
                temp_dir = tempfile.mkdtemp()
                # 1. Save uploaded image as PIL
                pil_img = Image.open(uploaded_image).convert('RGB')
                # 2. Generate speech audio
                audio_path = os.path.join(temp_dir, "speech.mp3")
                async def tts():
                    communicate = edge_tts.Communicate(script_text, selected_voice)
                    await communicate.save(audio_path)
                asyncio.run(tts())

                # 3. Detect mouth landmarks
                mouth_points = get_mouth_points(pil_img)
                if mouth_points is None:
                    st.warning("No face landmarks detected. Using a simple mouth overlay.")
                    # Use a fallback overlay (ellipse)
                    frames = []
                    audio_clip = mp_editor.AudioFileClip(audio_path)
                    duration = audio_clip.duration
                    num_frames = int(duration * 24)  # 24 fps
                    # We'll still generate an audio-driven mouth overlay
                    # For simplicity, we'll just use a constant open ratio
                    # but we can compute amplitude from audio
                    audio_array = audio_clip.to_soundarray(n_channels=1, fps=22050).flatten()
                    segment_len = max(1, len(audio_array) // num_frames)
                    amplitudes = []
                    for i in range(num_frames):
                        start = i * segment_len
                        end = min(start + segment_len, len(audio_array))
                        seg = audio_array[start:end]
                        rms = np.sqrt(np.mean(seg**2)) if len(seg) > 0 else 0
                        amplitudes.append(rms)
                    max_amp = max(amplitudes) if amplitudes else 1
                    amplitudes = [a / max_amp for a in amplitudes]
                    # Create frames with overlay
                    base_img = pil_img.copy()
                    w, h = base_img.size
                    for amp in amplitudes:
                        frame = base_img.copy()
                        draw = ImageDraw.Draw(frame)
                        open_ratio = 0.2 + 0.8 * amp
                        mouth_x = w//2 - 100
                        mouth_y = h//2 + 80
                        if open_ratio > 0.5:
                            draw.ellipse((mouth_x, mouth_y, mouth_x+200, mouth_y+80), fill=(255,180,130,255), outline=(200,100,50,255))
                        else:
                            draw.line((mouth_x+20, mouth_y+40, mouth_x+180, mouth_y+40), fill=(255,200,150,255), width=6)
                        frames.append(np.array(frame))
                else:
                    # Use mediapipe warping
                    audio_clip = mp_editor.AudioFileClip(audio_path)
                    duration = audio_clip.duration
                    num_frames = int(duration * 24)  # 24 fps
                    audio_array = audio_clip.to_soundarray(n_channels=1, fps=22050).flatten()
                    segment_len = max(1, len(audio_array) // num_frames)
                    amplitudes = []
                    for i in range(num_frames):
                        start = i * segment_len
                        end = min(start + segment_len, len(audio_array))
                        seg = audio_array[start:end]
                        rms = np.sqrt(np.mean(seg**2)) if len(seg) > 0 else 0
                        amplitudes.append(rms)
                    max_amp = max(amplitudes) if amplitudes else 1
                    amplitudes = [a / max_amp for a in amplitudes]
                    frames = []
                    for amp in amplitudes:
                        open_ratio = 0.2 + 0.8 * amp
                        warped = warp_mouth_pil(pil_img, mouth_points, open_ratio)
                        frames.append(np.array(warped))

                # 4. Create video from frames
                clips = [mp_editor.ImageClip(frame).set_duration(1/24) for frame in frames]
                video = mp_editor.concatenate_videoclips(clips, method="chain")
                audio = mp_editor.AudioFileClip(audio_path)
                video = video.set_audio(audio)
                output_path = os.path.join(temp_dir, "output.mp4")
                video.write_videofile(output_path, fps=24, codec='libx264', audio_codec='aac',
                                      verbose=False, logger=None)

                # 5. Display and download
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
