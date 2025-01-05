from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

app = FastAPI()

class Message(BaseModel):
    text: str

# Load trained T5 model and tokenizer from the correct checkpoint path
model_path = 'results'
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

@app.post("/chat")
def chat(message: Message):
    try:
        # Encode the input text
        input_text = message.text
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        # Generate response using T5 with parameters tuned for conversation
        outputs = model.generate(
            input_ids,
            max_length=200,  # Increased for potentially longer customer support responses
            num_beams=4,
            temperature=0.8,  # Slightly higher temperature for more diverse responses
            no_repeat_ngram_size=3,
            early_stopping=True,
            do_sample=True  # Enable sampling for more natural responses
        )
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {"response": response_text}
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")  # Turkish error logging
        raise HTTPException(status_code=500, detail=str(e))
        
        


