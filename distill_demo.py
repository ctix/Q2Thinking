###3. 实战代码框架 (基于PyTorch和Hugging Face)
# Script for model distillation and pruning
# Teacher: deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
# Student: google/gemma-2b-it (or other chosen student)
# Includes: KL Divergence Distillation, Cross-Entropy Task Loss, L1 Regularization

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm # For progress bar

# 0. Configuration
teacher_model_name = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
student_model_name = "google/gemma-2b-it"
dataset_name = 'mlabonne/guanaco-llama2-1k'
dataset_split = 'train[:100]' # Using a small subset for demonstration
max_length = 512 # Max sequence length for tokenization
batch_size = 2   # Batch size for DataLoader
num_epochs = 1   # Number of training epochs
learning_rate = 1e-4
temperature = 4.0 # Distillation temperature
alpha = 0.9       # Weight for distillation loss vs student loss
l1_lambda = 1e-5  # L1 regularization strength
output_dir = "./distilled_student_model" # Directory to save the distilled model

# 1. Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Tokenizer
# We use the student's tokenizer as it's generally a good practice.
tokenizer = AutoTokenizer.from_pretrained(student_model_name)
# Handle potential missing pad_token: Gemma tokenizer might not have a pad_token_id by default.
if tokenizer.pad_token_id is None:
    if tokenizer.eos_token_id is not None:
        print("Tokenizer missing pad_token_id, using eos_token_id as pad_token_id.")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        # Add a new pad token if eos is also missing (less common for models like gemma)
        print("Tokenizer missing pad_token_id and eos_token_id. Adding a new pad token.")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # Important: If a new token is added, student_model embeddings might need resizing.
        # student_model.resize_token_embeddings(len(tokenizer)) # Add this if new token added before model loading

# 3. Dataset and DataLoader
class DistillationDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokenized_output = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length', # Pad to max_length
            truncation=True,
            return_tensors='pt'
        )
        input_ids = tokenized_output.input_ids.squeeze(0) # Remove batch dim
        labels = input_ids.clone()
        return {'input_ids': input_ids, 'labels': labels}

print(f"Loading dataset: {dataset_name}, split: {dataset_split}")
raw_dataset = load_dataset(dataset_name, split=dataset_split)
texts = [item['text'] for item in raw_dataset] # Assuming 'text' column is present

distill_dataset = DistillationDataset(texts=texts, tokenizer=tokenizer, max_length=max_length)
dataloader = DataLoader(distill_dataset, batch_size=batch_size)

# 4. Models
print(f"Loading teacher model: {teacher_model_name}")
# trust_remote_code=True is often required for custom architectures.
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name, trust_remote_code=True)
teacher_model.to(device)
teacher_model.eval() # Teacher model is in evaluation mode

print(f"Loading student model: {student_model_name}")
student_model = AutoModelForCausalLM.from_pretrained(student_model_name)
# If tokenizer was resized by adding new tokens, model embeddings must be resized *after* loading the model
if tokenizer.pad_token_id is not None and tokenizer.pad_token_id >= student_model.config.vocab_size:
     print(f"Resizing student model token embeddings to fit tokenizer vocab size: {len(tokenizer)}")
     student_model.resize_token_embeddings(len(tokenizer))
student_model.to(device)
student_model.train() # Student model is in training mode

# TODO: Implement layer freezing if specific layers are identified for pruning.
# Example:
# for name, param in student_model.named_parameters():
#     if 'some_layer_to_freeze' in name:
#         param.requires_grad = False

# 5. Optimizer
optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)

# 6. Training Loop
print("Starting distillation training...")
for epoch in range(num_epochs):
    print(f"--- Epoch {epoch+1}/{num_epochs} ---")
    total_loss = 0.0
    student_model.train() # Ensure student model is in training mode each epoch

    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
    inputs = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)

    # Get teacher predictions (no gradient)
    with torch.no_grad():
        teacher_outputs = teacher_model(inputs)
        teacher_logits = teacher_outputs.logits

    # Get student predictions
    student_outputs = student_model(inputs)
    student_logits = student_outputs.logits
    
    # Calculate losses
    # Distillation loss (KL divergence)
    loss_distill = F.kl_div(
        input=F.log_softmax(student_logits / temperature, dim=-1),
        target=F.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchmean'  # Using 'batchmean' averages over the batch
    ) * (temperature ** 2) # Scale correction for temperature

    # Student task loss (Cross-Entropy)
    loss_student = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)), # (batch*seq_len, vocab_size)
        labels.view(-1),                                   # (batch*seq_len)
        ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
    )

    # Combined loss
    current_loss = alpha * loss_distill + (1 - alpha) * loss_student

    # Add L1 Regularization
    l1_penalty = torch.tensor(0.0).to(device)
    for param in student_model.parameters():
        if param.requires_grad: # Only apply to trainable parameters
            l1_penalty += torch.abs(param).sum()
    current_loss += l1_lambda * l1_penalty

    # Backpropagation and optimization
    optimizer.zero_grad()
    current_loss.backward()
    optimizer.step()

    total_loss += current_loss.item()

    avg_epoch_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")

# 7. Save Model and Tokenizer
print(f"Saving distilled student model and tokenizer to {output_dir}")
student_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Distillation and pruning script finished.")
