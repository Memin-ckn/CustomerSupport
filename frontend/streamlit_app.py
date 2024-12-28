import streamlit as st
import requests
import speech_recognition as sr
from audio_recorder_streamlit import audio_recorder
import io

st.title("Türkçe Chatbot")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Text input
text_input = st.chat_input("Mesajınızı yazın...")

# Audio recording
audio_bytes = audio_recorder()

if text_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": text_input})
    
    # Send request to backend
    response = requests.post(
        "http://localhost:8000/process_text",
        json={"text": text_input}
    )
    
    bot_response = response.json()["response"]
    
    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    st.rerun()

if audio_bytes:
    # Convert audio bytes to file-like object
    audio_file = io.BytesIO(audio_bytes)
    
    # Send audio to backend
    files = {"file": ("audio.wav", audio_file, "audio/wav")}
    response = requests.post("http://localhost:8000/process_speech", files=files)
    
    bot_response = response.json()["response"]
    
    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    st.rerun() 