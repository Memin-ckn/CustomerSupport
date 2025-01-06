"""import streamlit as st
import requests
import speech_recognition as sr
from gtts import gTTS
import os
from playsound3 import playsound
import assemblyai as aai
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
            # Prepare the conversation history
            conversation_history = "\n".join([f"User: {user}\nBot: {bot}" for user, bot in st.session_state.chat_history])
            input_text = f"{conversation_history}\nUser: {user_message}\nBot:"

            response = requests.post(BACKEND_URL, json={"text": input_text})
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
    if st.button("Dinle"):
        try:
            tts = gTTS(text=latest_bot_response, lang="tr")
            tts.save("response.mp3")
            playsound("response.mp3")
            os.remove("response.mp3")
        except Exception as e:
            st.error(f"Ses dönüşümünde hata: {e}")
"""
import streamlit as st
import requests
import speech_recognition as sr
from gtts import gTTS
import os
from playsound3 import playsound
import assemblyai as aai
import tempfile

# FastAPI backend URL
BACKEND_URL = "http://127.0.0.1:8000/chat"

# AssemblyAI setup
aai.settings.api_key = "ba93c725a52542a3afa0b4a312005c9b"
transcriber = aai.Transcriber()

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

def process_message_and_get_response(message):
    conversation_history = "\n".join([f"User: {user}\nBot: {bot}" for user, bot in st.session_state.chat_history])
    input_text = f"{conversation_history}\nUser: {message}\nBot:"
    response = requests.post(BACKEND_URL, json={"text": input_text})
    return response.json()["response"]

if st.button("Send"):
    if user_message:
        try:
            bot_response = process_message_and_get_response(user_message)
            st.session_state.chat_history.append((user_message, bot_response))
            st.session_state.user_message = ""
        except Exception as e:
            st.error(f"Error communicating with the backend: {e}")

def save_audio_to_file(audio_data):
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "temp_audio.wav")
    with open(temp_path, "wb") as f:
        f.write(audio_data.get_wav_data())
    return temp_path

if st.button("Speak"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        try:
            audio = recognizer.listen(source)
            temp_audio_path = save_audio_to_file(audio)
            
            # Transcribe using AssemblyAI
            transcript = transcriber.transcribe(temp_audio_path)
            user_message = transcript.text
            
            st.success(f"You said: {user_message}")
            
            # Get bot response
            bot_response = process_message_and_get_response(user_message)
            st.session_state.chat_history.append((user_message, bot_response))
            
            # Cleanup
            os.remove(temp_audio_path)
            os.rmdir(os.path.dirname(temp_audio_path))
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Text-to-speech output
if st.session_state.chat_history:
    latest_bot_response = st.session_state.chat_history[-1][1]
    if st.button("Dinle"):
        try:
            tts = gTTS(text=latest_bot_response, lang="tr")
            tts.save("response.mp3")
            playsound("response.mp3")
            os.remove("response.mp3")
        except Exception as e:
            st.error(f"Ses dönüşümünde hata: {e}")