from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from typing import List, Dict
import uuid

app = FastAPI()

class Message(BaseModel):
    text: str
    conversation_id: str = None

class Conversation(BaseModel):
    messages: List[dict]  # Store both user messages and bot responses with their roles
    response: str = ""

# Load trained T5 model and tokenizer from the correct checkpoint path
model_path = 'results'
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Store conversations in memory
conversations: Dict[str, Conversation] = {}

def format_conversation_history(messages: List[dict]) -> str:
    """Format the conversation history in a structured way for the model."""
    formatted_history = []
    for msg in messages[-3:]:  # Keep last 3 message pairs
        role = msg["role"]
        content = msg["content"]
        formatted_history.append(f"{role}: {content}")
    return " | ".join(formatted_history)

def generate_response(input_text: str, conversation_history: List[dict]) -> str:
    # Format the conversation history with the current input
    formatted_input = format_conversation_history(conversation_history) + f" | user: {input_text}"
    
    # Add a prompt to help the model understand it should generate a response
    model_input = f"Generate response: {formatted_input}"
    
    # Encode the input text
    input_ids = tokenizer.encode(model_input, return_tensors="pt").to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=200,
            num_beams=4,
            temperature=0.8,
            no_repeat_ngram_size=3,
            early_stopping=True,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            attention_mask=torch.ones_like(input_ids)
        )
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up the response if it starts with "assistant:" or similar
    response_text = response_text.split(":", 1)[-1].strip()