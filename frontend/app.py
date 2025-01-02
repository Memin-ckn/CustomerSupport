import streamlit as st
import requests
import speech_recognition as sr
from gtts import gTTS
import os
from playsound3 import playsound

# FastAPI backend URL
BACKEND_URL = "http://127.0.0.1:8000/chat"

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Title
st.title("AI Chatbot with Text and Speech")

# Display chat history
for user, bot in st.session_state.chat_history:
    st.write(f"**You:** {user}")
    st.write(f"**Bot:** {bot}")

# Text input
user_message = st.text_input("Type your message here:")

if st.button("Send"):
    if user_message:
        try:
            response = requests.post(BACKEND_URL, json={"text": user_message})
            response_data = response.json()
            bot_response = response_data["response"]

            # Add to chat history
            st.session_state.chat_history.append((user_message, bot_response))

            # Clear the input box after sending
            st.session_state.user_message = ""
        except Exception as e:
            st.error(f"Error communicating with the backend: {e}")

# Speech input
if st.button("Speak"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        try:
            audio = recognizer.listen(source)
            user_message = recognizer.recognize_google(audio)
            st.success(f"You said: {user_message}")

            # Send to backend
            response = requests.post(BACKEND_URL, json={"text": user_message})
            response_data = response.json()
            bot_response = response_data["response"]

            # Add to chat history
            st.session_state.chat_history.append((user_message, bot_response))

            # Clear the input box after sending
            st.session_state.user_message = ""
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand your speech.")
        except sr.RequestError:
            st.error("Speech recognition service is unavailable.")
        except Exception as e:
            st.error(f"Error communicating with the backend: {e}")

# Text-to-speech output
if st.session_state.chat_history:
    latest_bot_response = st.session_state.chat_history[-1][1]
    if st.button("Listen"):
        tts = gTTS(text=latest_bot_response, lang="tr")
        tts.save("response.mp3")
        playsound("response.mp3")
        os.remove("response.mp3")
