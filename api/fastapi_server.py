from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import speech_recognition as sr
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

app = FastAPI()

# Load models
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

class TextRequest(BaseModel):
    text: str
    language: str = "tr"

@app.post("/process_text")
async def process_text(request: TextRequest):
    input_text = request.text
    
    # Tokenize and generate response
    inputs = tokenizer.encode("respond: " + input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {"response": response}

@app.post("/process_speech")
async def process_speech(file: UploadFile = File(...)):
    # Initialize speech recognizer
    recognizer = sr.Recognizer()
    
    # Convert speech to text
    with sr.AudioFile(file.file) as source:
        audio = recognizer.record(source)
        text = recognizer.recognize_google(audio, language="tr-TR")
    
    # Process text using the same model
    return await process_text(TextRequest(text=text)) 