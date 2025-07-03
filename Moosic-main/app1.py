import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser

col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    st.write("")

with col2:
    st.image("./Images/logo.png", width=530, use_column_width=True)

with col3:
    st.write("")

st.title("Moosic")
st.write('Moosic is an emotion detection-based music recommendation system. Capture your emotion from a single frame, then get music recommendations!')

model = load_model("model.h5")
label = np.load("labels.npy")

holistic = mp.solutions.holistic
hol = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Function to capture a single frame for emotion detection
def capture_emotion():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)  # Flipping the frame horizontally

        res = hol.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        lst = []
        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            for _ in range(42):
                lst.append(0.0)

        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            for _ in range(42):
                lst.append(0.0)

        lst = np.array(lst).reshape(1, -1)
        pred = label[np.argmax(model.predict(lst))]
        st.write(f"Detected Emotion: {pred}")
        np.save("detected_emotion.npy", np.array([pred]))
        drawing.draw_landmarks(frame, res.face_landmarks, holistic.FACEMESH_TESSELATION)
        drawing.draw_landmarks(frame, res.left_hand_landmarks, holistic.HAND_CONNECTIONS)
        drawing.draw_landmarks(frame, res.right_hand_landmarks, holistic.HAND_CONNECTIONS)
        
        st.image(frame, channels="BGR", use_column_width=True)
        
    cap.release()

lang = st.text_input("Enter your preferred language")
artist = st.text_input("Enter your preferred artist")

if st.button("Capture Emotion"):
    capture_emotion()

btn = st.button("Recommend music")

if btn:
    try:
        detected_emotion = np.load("detected_emotion.npy")[0]
    except:
        detected_emotion = ""
    
    if not detected_emotion:
        st.warning("Please capture your emotion first!")
    else:
        webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{detected_emotion}+songs+{artist}")
        np.save("detected_emotion.npy", np.array([""]))

# Streamlit Customisation
st.markdown("""
<style>
header {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
