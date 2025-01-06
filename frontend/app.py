import streamlit as st
import requests
from gtts import gTTS
import os
from playsound3 import playsound
import whisper
import tempfile

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


if st.button("Speak"):
    st.write("Button pressed")  # Debug print 1
    
    model = whisper.load_model("base")
    st.write("Model loaded")  # Debug print 2
    
    try:
        import sounddevice as sd
        from scipy.io.wavfile import write

        fs = 16000
        seconds = 5
        st.write("Starting recording...")  # Debug print 3
        
        audio_data = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        st.write("Recording completed")  # Debug print 4
        st.write(f"Audio data shape: {audio_data.shape}")  # Debug print 5

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            temp_file_path = temp_audio_file.name
            write(temp_file_path, fs, audio_data)
            st.write(f"Audio saved to: {temp_file_path}")  # Debug print 6

        st.write("Starting transcription...")  # Debug print 7
        result = model.transcribe(temp_file_path, language="tr")
        st.write("Transcription completed")  # Debug print 8
        st.write(f"Raw result: {result}")  # Debug print 9
        
        user_message = result["text"]
        st.write(f"Extracted text: {user_message}")  # Debug print 10

        # Rest of your code...
        st.success(f"You said: {user_message}")

        response = requests.post(BACKEND_URL, json={"text": user_message})
        response_data = response.json()
        bot_response = response_data["response"]

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        st.session_state.chat_history.append((user_message, bot_response))

        # Corrected chat history display
        for user_msg, bot_msg in st.session_state.chat_history:
            st.write(f"You: {user_msg}")
            st.write(f"Bot: {bot_msg}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

    finally:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


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

def test_audio():
    import sounddevice as sd
    fs = 16000
    duration = 3  # seconds
    
    st.write("Testing audio device...")
    st.write(f"Available audio devices:")
    st.write(sd.query_devices())
    
    try:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        st.write("Recording successful")
        st.write(f"Recording shape: {recording.shape}")
        st.write(f"Recording min/max values: {recording.min()}, {recording.max()}")
        return True
    except Exception as e:
        st.error(f"Recording failed: {e}")
        return False

if st.button("Test Audio"):
    test_audio()

if st.button("Test Whisper"):
    try:
        model = whisper.load_model("base")
        st.write("Whisper model loaded successfully")
        st.write(f"Model device: {model.device}")
        st.write(f"Model is multilingual: {model.is_multilingual}")
    except Exception as e:
        st.error(f"Whisper model loading failed: {e}")