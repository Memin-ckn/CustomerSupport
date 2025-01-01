from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

app = FastAPI()

class Message(BaseModel):
    text: str

# Load trained T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('./results')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

@app.post("/chat")
def chat(message: Message):
    try:
        input_text = f"chat: {message.text}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        # Generate response using T5
        outputs = model.generate(input_ids, max_length=100, num_beams=5, early_stopping=True)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
