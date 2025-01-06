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
st.title("Sesli ve Yazılı Müsteri Destek Asistanı")

# Display chat history
for user, bot in st.session_state.chat_history:
    st.write(f"**Siz:** {user}")
    st.write(f"**Asistan:** {bot}")

# Text input
user_message = st.text_input("Mesajınızı buraya yazınız:")

if st.button("Gönder"):
    if user_message:
        try:
            # Prepare the conversation history in the format expected by the backend
            conversation_history = []
            for user, bot in st.session_state.chat_history:
                conversation_history.extend([
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": bot}
                ])
            
            # Add current user message
            conversation_history.append({"role": "user", "content": user_message})
            
            response = requests.post(BACKEND_URL, json={
                "text": user_message,
                "conversation_history": conversation_history
            })
            response_data = response.json()
            bot_response = response_data["response"]

            # Add to chat history
            st.session_state.chat_history.append((user_message, bot_response))

            # Clear the input box after sending
            user_message = ""  # Reset the local variable
            st.session_state.user_message = ""  # Reset the session state variable
        except Exception as e:
            st.error(f"Error communicating with the backend: {e}")


if st.button("Konuş"):
    st.write("Butona basılıd")  # Debug print 1
    
    model = whisper.load_model("base")
    st.write("Model yüklendi")  # Debug print 2
    
    try:
        import sounddevice as sd
        from scipy.io.wavfile import write

        fs = 16000
        seconds = 5
        st.write("Kayıt başladı...")  # Debug print 3
        
        audio_data = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        st.write("Kayıt tamamlandı")  # Debug print 4
        st.write(f"Ses verisi sekli: {audio_data.shape}")  # Debug print 5

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            temp_file_path = temp_audio_file.name
            write(temp_file_path, fs, audio_data)
            st.write(f"Ses buraya kaydedildi: {temp_file_path}")  # Debug print 6

        st.write("Metne çevriliyor...")  # Debug print 7
        result = model.transcribe(temp_file_path, language="tr")
        st.write("Metne çevrildi")  # Debug print 8
        st.write(f"Ham sonuç: {result}")  # Debug print 9
        
        user_message = result["text"]
        st.write(f"Çıkartılan metin: {user_message}")  # Debug print 10

        # Rest of your code...
        st.success(f"Dediniz ki: {user_message}")

        response = requests.post(BACKEND_URL, json={"text": user_message})
        response_data = response.json()
        bot_response = response_data["response"]

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        st.session_state.chat_history.append((user_message, bot_response))

        # Corrected chat history display
        for user_msg, bot_msg in st.session_state.chat_history:
            st.write(f"Siz: {user_msg}")
            st.write(f"Asistan: {bot_msg}")

    except Exception as e:
        st.error(f"Bir hata oluştu: {e}")

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
    
    st.write("Ses cihazı test ediliyor...")
    st.write(f"Müsait ses cihazları:")
    st.write(sd.query_devices())
    
    try:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        st.write("Kayıt başarılı")
        st.write(f"Kayıt sekli: {recording.shape}")
        st.write(f"Kayıt min/max değerleri: {recording.min()}, {recording.max()}")
        return True
    except Exception as e:
        st.error(f"Kayıt başarısız: {e}")
        return False

if st.button("Sesi Test Et"):
    test_audio()

if st.button("Whisper Modelini Test Et"):
    try:
        model = whisper.load_model("base")
        st.write("Whisper modeli başarıyla yüklendi")
        st.write(f"Model cihazı: {model.device}")
    except Exception as e:
        st.error(f"Whisper model loading failed: {e}")