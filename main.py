import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BlenderbotTokenizerFast, 
    BlenderbotForConditionalGeneration,
    BlenderbotConfig,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler

#--- Configuration Class ---
class Config:
    def __init__(self):
        # Model and Data Configuration
        self.model_id = "facebook/blenderbot-400M-distill"
        self.filepath = "christianlillelund/joe-rogan-experience-1169-elon-musk"
        
        # Optimized for RTX 4050
        self.max_length = 256  # Reduced from 512
        self.num_samples = 2  # Reduced number of samples
        self.batch_size = 8  # Adjusted for RTX 4050
        
        # Training Hyperparameters Optimized for Faster Training
        self.learning_rate = 3e-5  # Slightly reduced
        self.weight_decay = 0.01
        self.epochs = 3
        
        # Mixed Precision and Gradient Accumulation
        self.gradient_accumulation_steps = 2
        self.max_grad_norm = 1.0
        
        # Device Configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Inference Parameters
        self.max_new_tokens = 32
        self.do_sample = True
        self.top_p = 0.9
        self.temperature = 0.8
        
        # Logging and Saving
        self.model_output_dir = 'models'
        os.makedirs(self.model_output_dir, exist_ok=True)
        
        print(f"Running on Device: {self.device}")
        self.print_config()
    
    def print_config(self):
        print("\n--- Configuration ---")
        for attr, value in self.__dict__.items():
            if not attr.startswith('__') and not callable(value):
                print(f"{attr}: {value}")
        print("--------------------\n")

# Initialize configuration
config = Config()

#--- Dataset Class ---
class ChatDataset(Dataset):
    def __init__(self, filepath, tokenizer_name, max_length, num_samples):
        self.tokenizer = BlenderbotTokenizerFast.from_pretrained(tokenizer_name, truncation_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
        self.num_samples = num_samples
        
        # Load dataset
        self.dataset = self._download_data()
        self.data_array = self._preprocess_data()
        print(f"Loaded {len(self.data_array)} data samples")

    def _download_data(self):
        try:
            dataset = load_dataset('csv', data_files='joe-rogan-experience-1169-elon-musk.csv')
            print(f"Successfully loaded dataset with {len(dataset['train'])} examples")
            return dataset
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise

    def _preprocess_data(self):
        data_array = []
        for i in range(self.num_samples, len(self.dataset['train'])):
            discussion = ""
            bot_output = self.tokenizer.bos_token + self.dataset['train'][i]["Text"] + self.tokenizer.eos_token
            for j in reversed(range(i-self.num_samples, i)):
                discussion = self.tokenizer.bos_token + self.dataset['train'][j]["Text"] + self.tokenizer.eos_token + discussion
                data_array.append([discussion, bot_output])
        return data_array

    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, idx):
        discussion, bot_output = self.data_array[idx]
        
        # Tokenize inputs
        inputs = self.tokenizer(
            discussion, 
            padding='max_length', 
            max_length=self.max_length, 
            truncation=True, 
            return_tensors='pt'
        )
        
        # Tokenize labels
        labels = self.tokenizer(
            bot_output, 
            padding='max_length', 
            max_length=self.max_length, 
            truncation=True, 
            return_tensors='pt'
        )['input_ids']
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }

#--- Model Loading Function ---
def load_model(model_id, device):
    try:
        # Load model configuration
        model_config = BlenderbotConfig.from_pretrained(model_id)
        model_config.max_position_embeddings = config.max_length
        
        # Load model with mixed precision support
        model = BlenderbotForConditionalGeneration.from_pretrained(
            model_id, 
            config=model_config,
            ignore_mismatched_sizes=True,
            torch_dtype=torch.float16  # Use float16 for memory efficiency
        ).to(device)
        
        print(f"Model loaded successfully on {device}")
        return model
    
    except Exception as e:
        print(f"Model loading error: {e}")
        raise

#--- Training Function ---
def train(config, model, dataset):
    # Prepare DataLoader
    train_dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    
    # Prepare optimizer with weight decay
    optimizer = AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=100, 
        num_training_steps=len(train_dataloader) * config.epochs
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    train_losses = []
    
    print("Starting training process...")
    model.train()
    
    start_time = time.time()
    
    for epoch in range(config.epochs):
        total_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", unit="batch")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            # Mixed precision training
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / config.gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient accumulation and optimization
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # Clip gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                # Step optimizer
                scaler.step(optimizer)
                scaler.update()
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Step learning rate scheduler
                lr_scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss / (batch_idx + 1))
        
        train_losses.append(total_loss / len(train_dataloader))
        print(f"Epoch {epoch + 1} completed. Average Loss: {train_losses[-1]:.4f}")
    
    end_time = time.time()
    total_training_time = end_time - start_time
    print(f"\nTotal Training Time: {total_training_time:.2f} seconds")
    print(f"Average Time per Epoch: {total_training_time / config.epochs:.2f} seconds")
    
    return train_losses

#--- Main Execution Block ---
if __name__ == "__main__":
    # Load model and dataset
    model = load_model(config.model_id, config.device)
    dataset = ChatDataset(config.filepath, config.model_id, config.max_length, config.num_samples)
    
    # Train the model
    train_losses = train(config, model, dataset)
    
    # Save the model
    model.save_pretrained(config.model_output_dir)
    print(f"Model saved to {config.model_output_dir}")