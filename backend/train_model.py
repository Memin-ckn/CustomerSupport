from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import gc
import os
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
import numpy as np

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        # Daha agresif GPU bellek temizleme
        with torch.cuda.device('cuda'):
            torch.cuda.synchronize()

# GPU kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# GPU bellek optimizasyonları
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

CHUNK_SIZE = 5000  # 2 kat artış
MAX_LENGTH = 512
BATCH_SIZE = 48   # 2 kat artış
GRADIENT_ACCUMULATION_STEPS = 1  # Gradient accumulation'a gerek kalmadı

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=MAX_LENGTH):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encoded_data = self._pre_tokenize()

    def _pre_tokenize(self):
        encoded = []
        for idx in range(len(self.data)):
            instruction = self.data.iloc[idx]['instruction_turkish']
            response = self.data.iloc[idx]['response_turkish']

            # CPU'da tokenization
            inputs = self.tokenizer(
                instruction,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors=None
            )

            labels = self.tokenizer(
                response,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors=None
            )

            encoded.append({
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask'],
                'labels': labels['input_ids']
            })

            if idx % 500 == 0:
                clear_gpu_memory()

        return encoded

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        item = self.encoded_data[idx]
        return {
            'input_ids': torch.tensor(item['input_ids']),
            'attention_mask': torch.tensor(item['attention_mask']),
            'labels': torch.tensor(item['labels'])
        }

def train_on_chunk(model, dataloader, optimizer, scaler, device, gradient_accumulation_steps):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch_idx, batch in enumerate(progress_bar):
        with autocast():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Padding token'ları -100 ile değiştir
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # Loss'u gradient accumulation steps'e böl
            loss = outputs.loss / gradient_accumulation_steps

        # Gradient scaling ve backward
        scaler.scale(loss).backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps

        # Bellek optimizasyonu
        del outputs, loss
        clear_gpu_memory()

        progress_bar.set_postfix({"Loss": total_loss / (batch_idx + 1)})

    return total_loss / len(dataloader)

# Model ve tokenizer initialization
print("Loading model and tokenizer...")
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Bellek optimizasyonları
model.gradient_checkpointing_enable()  # Gradient checkpointing aktif
model = model.to(device)

# Optimizer ve scaler
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)  # 5e-5'ten 1e-4'e
scaler = GradScaler()

# Training loop
epochs = 10

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    epoch_loss = 0
    chunk_count = 0

    # Her epoch için veriyi yeniden yükle
    data_chunks = pd.read_csv(
        '/content/drive/MyDrive/training_turkish.csv',
        chunksize=CHUNK_SIZE
    )

    for chunk in data_chunks:
        chunk_count += 1
        print(f"Processing chunk {chunk_count}")

        dataset = CustomDataset(chunk, tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            pin_memory=True,
            num_workers=6  # Daha fazla worker
        )

        chunk_loss = train_on_chunk(
            model,
            dataloader,
            optimizer,
            scaler,
            device,
            GRADIENT_ACCUMULATION_STEPS
        )

        epoch_loss += chunk_loss

        # Chunk işleme sonrası bellek temizliği
        del dataset, dataloader
        clear_gpu_memory()

        # Her 5 chunk'ta bir checkpoint kaydet
        if chunk_count % 5 == 0:
            checkpoint_path = f'./checkpoints/epoch_{epoch + 1}_chunk_{chunk_count}'
            os.makedirs(checkpoint_path, exist_ok=True)
            model.save_pretrained(checkpoint_path)

    avg_epoch_loss = epoch_loss / chunk_count
    print(f"Epoch {epoch + 1} Average Loss: {avg_epoch_loss}")

    # Epoch sonu checkpoint
    checkpoint_path = f'./checkpoints/epoch_{epoch + 1}'
    os.makedirs(checkpoint_path, exist_ok=True)
    model.save_pretrained(checkpoint_path)

# Final model kaydetme
save_directory = './results'
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"Model and tokenizer saved to {save_directory}")