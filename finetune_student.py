# finetune_student.py
# Script for fine-tuning a pre-trained student model (e.g., a distilled model)
# on a custom dataset like 'electronic_circuit_questions.jsonl'.

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset # For loading JSONL directly
from tqdm import tqdm
import argparse
import logging
import os
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Argument Parser
def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune a student model on a custom dataset.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the pre-trained student model to be fine-tuned.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the custom dataset (e.g., electronic_circuit_questions.jsonl).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model and tokenizer.")

    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length for tokenization.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for DataLoader.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--trust_remote_code", action='store_true', help="Boolean flag for models requiring trust_remote_code=True.")

    args = parser.parse_args()
    return args

# 2. Custom Dataset Class (FineTuningDataset)
class FineTuningDataset(Dataset):
    def __init__(self, dataset_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eos_token = tokenizer.eos_token if tokenizer.eos_token else ""

        logging.info(f"Loading dataset from: {dataset_path}")
        self.data = []
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        self.data.append(json.loads(line))
                    except json.JSONDecodeError:
                        logging.warning(f"Skipping invalid JSON line: {line.strip()}")
            logging.info(f"Loaded {len(self.data)} samples from {dataset_path}.")
        except FileNotFoundError:
            logging.error(f"Dataset file not found: {dataset_path}")
            raise
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            raise

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")

        # Format the text. Add EOS token to ensure the model learns to stop.
        formatted_text = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output_text}{self.eos_token}"

        # Tokenize the formatted text
        # We use padding='max_length' and truncation=True.
        # For Causal LM, labels are typically the same as input_ids.
        # The model itself handles the shifting of labels for next-token prediction.
        tokenized_output = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = tokenized_output.input_ids.squeeze(0) # Remove batch dim from tokenizer output
        # For Causal LM fine-tuning, labels are usually input_ids shifted internally by the model.
        # If providing labels directly, they should be a copy of input_ids.
        # Padding tokens in labels will be ignored by the loss function (typically -100).
        labels = input_ids.clone()

        # Optional: If you want to ensure padded tokens are ignored in loss calculation explicitly
        # This is often handled by setting model.config.pad_token_id or by the Trainer class.
        # If labels[i] == tokenizer.pad_token_id, set labels[i] = -100
        if self.tokenizer.pad_token_id is not None:
             labels[labels == self.tokenizer.pad_token_id] = -100

        return {'input_ids': input_ids, 'labels': labels}


def main():
    args = parse_arguments()

    # 3. Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 4. Load Tokenizer
    logging.info(f"Loading tokenizer from: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)

    # Handle pad_token: Common for Gemma and other models not to have it set by default.
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            logging.info("Tokenizer missing pad_token, using eos_token as pad_token.")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            logging.info("Tokenizer missing pad_token and eos_token. Adding a new pad token: '[PAD]'.")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # Set pad_token_id for consistency if not already set from pad_token
    if tokenizer.pad_token_id is None and tokenizer.pad_token is not None:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    logging.info(f"Tokenizer pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
    logging.info(f"Tokenizer eos_token: {tokenizer.eos_token}, eos_token_id: {tokenizer.eos_token_id}")


    # 5. Load Model
    logging.info(f"Loading model from: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        # torch_dtype=torch.bfloat16 if device.type == 'cuda' else None # Optional: for memory saving on GPU
    )

    # Resize embeddings if a new pad token was added to tokenizer *after* it was loaded with the model
    # This ensures the model's embedding matrix matches the tokenizer's vocabulary size.
    if len(tokenizer) > model.config.vocab_size:
        logging.info(f"Resizing model token embeddings from {model.config.vocab_size} to {len(tokenizer)} to match tokenizer.")
        model.resize_token_embeddings(len(tokenizer))
        # Also important: if you added a new pad token, make sure the model's pad_token_id is updated for generation.
        if tokenizer.pad_token_id is not None: # Should be set by now
            model.config.pad_token_id = tokenizer.pad_token_id

    model.to(device)

    # 6. Create FineTuningDataset and DataLoader
    logging.info("Creating dataset and dataloader...")
    finetune_dataset = FineTuningDataset(args.dataset_path, tokenizer, args.max_length)
    # Note: For fine-tuning, shuffle=True is typical for the training DataLoader
    dataloader = DataLoader(finetune_dataset, batch_size=args.batch_size, shuffle=True)

    # 7. Optimizer
    logging.info("Setting up optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 8. Fine-tuning Loop
    logging.info("Starting fine-tuning...")
    model.train() # Set model to training mode

    for epoch in range(args.num_epochs):
        logging.info(f"--- Epoch {epoch+1}/{args.num_epochs} ---")
        total_epoch_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=True)
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss # Hugging Face models return loss when labels are provided

            if loss is None:
                logging.error("Loss is None. This should not happen when labels are provided.")
                continue

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()
            progress_bar.set_postfix({'batch_loss': f'{loss.item():.4f}'})

            # Optional: Log batch loss more periodically if many steps
            # if (batch_idx + 1) % 10 == 0:
            #    logging.info(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)}, Batch Loss: {loss.item():.4f}")

        avg_epoch_loss = total_epoch_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        logging.info(f"Epoch {epoch+1} finished. Average Epoch Loss: {avg_epoch_loss:.4f}")

    # 9. Model Saving
    logging.info(f"Fine-tuning finished. Saving model and tokenizer to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logging.info("Fine-tuning script completed successfully.")

if __name__ == "__main__":
    main()
