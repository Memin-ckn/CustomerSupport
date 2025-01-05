from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the preprocessed data
data = pd.read_csv('data/processed_training_turkish_first5k.csv')

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instruction = self.data.iloc[idx]['instruction_turkish']
        response = self.data.iloc[idx]['response_turkish']
        inputs = self.tokenizer(instruction, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        labels = self.tokenizer(response, return_tensors='pt', padding='max_length', truncation=True, max_length=512).input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100  # Replace padding token id's in labels by -100
        return {
            "input_ids": inputs.input_ids.flatten(), 
            "labels": labels.flatten()
        }

# Initialize the dataset and dataloader
dataset = CustomDataset(data)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize the model and move it to the GPU
model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Define optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
epochs = 3
model.train()

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    epoch_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix({"Loss": loss.item()})
    
    print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(dataloader)}")

# Save the trained T5 model and tokenizer
save_directory = './results'
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"Model and tokenizer saved to {save_directory}")
