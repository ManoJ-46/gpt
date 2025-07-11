import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import random
import time
from streamlit_autorefresh import st_autorefresh
from CNNModel import CNNModel

st.set_page_config(page_title="Sign Language Recognition", page_icon="🤟", layout="wide")

# CSS Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');
* { font-family: 'Montserrat', sans-serif !important; }
.stButton>button { animation: pulse 2s infinite; border-radius: 12px !important; }
@keyframes pulse {
  0% { transform: scale(1); }
  50% { transform: scale(1.05); }
  100% { transform: scale(1); }
}
.webcam-container {
  max-width: 100%;
  border-radius: 15px;
  overflow: hidden;
  box-shadow: 0 10px 30px rgba(0,0,0,0.3);
  position: relative;
}
.prediction-badge {
  font-size: 1.5rem;
  font-weight: bold;
  padding: 10px 20px;
  border-radius: 30px;
  background-color: #4CAF50;
  color: white;
}
</style>
""", unsafe_allow_html=True)

# Mediapipe hands
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands_detector = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.2)

@st.cache_resource
def load_models():
    model = CNNModel()
    model.load_state_dict(torch.load("trained.pth", map_location="cpu"))
    model.eval()
    return model

model_alpha = load_models()

alphabet_classes = {
    i: chr(65+i) for i in range(26)
}

# Predict Sign Frame
def predict_sign_realtime(frame, model, classes_reverse):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(frame_rgb)

    coordinates = []
    x_coords, y_coords, z_coords = [], [], []
    predicted_character = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            data = {}
            for i in range(len(hand_landmarks.landmark)):
                lm = hand_landmarks.landmark[i]
                x_coords.append(lm.x)
                y_coords.append(lm.y)
                z_coords.append(lm.z)

            for i, landmark in enumerate(mp_hands.HandLandmark):
                lm = hand_landmarks.landmark[i]
                data[f'{landmark.name}_x'] = lm.x - min(x_coords)
                data[f'{landmark.name}_y'] = lm.y - min(y_coords)
                data[f'{landmark.name}_z'] = lm.z - min(z_coords)

            coordinates.append(data)

            coordinates_df = pd.DataFrame(coordinates)
            coords_reshaped = np.reshape(coordinates_df.values, (coordinates_df.shape[0], 63, 1))
            coords_tensor = torch.from_numpy(coords_reshaped).float()

            with torch.no_grad():
                outputs = model(coords_tensor)
                _, predicted = torch.max(outputs.data, 1)
                pred_idx = predicted.cpu().numpy()[0]
                predicted_character = classes_reverse[pred_idx]

            h, w, _ = frame.shape
            x1 = int(min(x_coords) * w) - 10
            y1 = int(min(y_coords) * h) - 10
            x2 = int(max(x_coords) * w) + 10
            y2 = int(max(y_coords) * h) + 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            return frame, predicted_character
    return frame, None

# Streamlit WebRTC Processor
class SignVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.prediction = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        processed_frame, prediction = predict_sign_realtime(img, model_alpha, alphabet_classes)
        self.prediction = prediction
        return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")

def load_sign_image(letter):
    image_path = f"alphabets/{letter}.jpg"
    if os.path.exists(image_path):
        return Image.open(image_path)
    else:
        img = Image.new('RGB', (200, 200), color='white')
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 100)
        except:
            font = ImageFont.load_default()
        text_width = draw.textlength(letter, font=font)
        x = (200 - text_width) // 2
        y = 50
        draw.text((x, y), letter, fill='black', font=font)
        return img

def guess_the_character_game():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Guess the Sign Language Character Game")
    if 'score' not in st.session_state: st.session_state.score = 0
    if 'total' not in st.session_state: st.session_state.total = 0
    if 'random_letter' not in st.session_state:
        st.session_state.random_letter = random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    if 'start_time' not in st.session_state: st.session_state.start_time = time.time()
    if 'guess_result' not in st.session_state: st.session_state.guess_result = None
    if 'game_locked' not in st.session_state: st.session_state.game_locked = False

    TIME_LIMIT = 10
    elapsed = int(time.time() - st.session_state.start_time)
    time_left = max(0, TIME_LIMIT - elapsed)

    if not st.session_state.game_locked:
        st_autorefresh(interval=1000, key="timer_refresh")

    st.info(f"Score: {st.session_state.score} / {st.session_state.total}")
    st.warning(f"Time left: {time_left} seconds")

    letter = st.session_state.random_letter
    img_path = f"alphabets/{letter}.jpg"
    if os.path.exists(img_path):
        st.image(img_path, caption="Which letter is this?", width=300)

    if time_left <= 0 and not st.session_state.game_locked:
        st.session_state.total += 1
        st.session_state.guess_result = f"Time's up! The correct letter was '{letter}'"
        st.session_state.game_locked = True

    if not st.session_state.game_locked:
        guess = st.selectbox("Select your guess:", list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        if st.button("Submit Guess"):
            st.session_state.total += 1
            if guess == letter:
                st.session_state.score += 1
                st.session_state.guess_result = "Correct!"
            else:
                st.session_state.guess_result = f"Incorrect! It was '{letter}'"
            st.session_state.game_locked = True

    if st.session_state.game_locked and st.session_state.guess_result:
        if "Correct" in st.session_state.guess_result:
            st.success(st.session_state.guess_result)
        elif "Time's up" in st.session_state.guess_result:
            st.warning(st.session_state.guess_result)
        else:
            st.error(st.session_state.guess_result)
        if st.button("Next"):
            st.session_state.random_letter = random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
            st.session_state.start_time = time.time()
            st.session_state.guess_result = None
            st.session_state.game_locked = False
    st.markdown('</div>', unsafe_allow_html=True)

# --- UI Main ---
st.markdown('<p class="main-title">Sign Language Recognition System</p>', unsafe_allow_html=True)

app_mode = st.sidebar.selectbox("Select Mode:", ["Live Detection", "English to Sign Language", "Guess the Character"])

if app_mode == "Live Detection":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Live Sign Language Detection")

    webrtc_ctx = webrtc_streamer(
        key="sign-detect",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=SignVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    if webrtc_ctx.video_processor and webrtc_ctx.video_processor.prediction:
        st.markdown(f'<div class="prediction-badge">Prediction: {webrtc_ctx.video_processor.prediction}</div>', unsafe_allow_html=True)
    else:
        st.info("Waiting for hand detection...")
    st.markdown('</div>', unsafe_allow_html=True)

elif app_mode == "English to Sign Language":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("English to Sign Language Converter")
    user_text = st.text_input("Enter text to convert:", placeholder="Type your message...")
    if st.button("Convert to Sign Language"):
        if not user_text:
            st.error("Please enter some text!")
        else:
            filtered_text = ''.join([c.upper() for c in user_text if c.isalpha()])
            if filtered_text:
                st.subheader(f"Converted Text: '{user_text}'")
                sign_images = [load_sign_image(c) for c in filtered_text]
                cols = st.columns(5)
                for i, img in enumerate(sign_images):
                    with cols[i % 5]:
                        st.image(img, caption=filtered_text[i], use_column_width=True)
                    if (i + 1) % 5 == 0 and (i + 1) < len(sign_images):
                        cols = st.columns(5)
                st.success("Conversion successful!")
            else:
                st.warning("No alphabetic characters to convert!")
    st.markdown('</div>', unsafe_allow_html=True)

elif app_mode == "Guess the Character":
    guess_the_character_game()
