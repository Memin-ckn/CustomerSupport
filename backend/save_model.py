import os
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Path to the directory where the model will be saved
save_directory = './results/checkpoint-1875'

# Create the directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Save the trained T5 model and tokenizer
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"Model and tokenizer saved to {save_directory}")
