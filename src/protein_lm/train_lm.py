import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import argparse
from pathlib import Path
import os

from src.protein_lm.config import ProteinLMConfig, load_config
from src.protein_lm.tokenizer import ProteinTokenizer
from src.protein_lm.models import ProteinConditionalTransformer
from src.protein_lm.data import create_dataloader

def train(config_path: str):
    """
    Trains the protein language model.
    """
    # --- 1. Load Configuration ---
    # This keeps the model and training parameters separate from the code.
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    lm_config = load_config(config_path, ProteinLMConfig)
    training_config = config_data.get('training', {})
    data_config = config_data.get('data', {})

    # --- 2. Setup ---
    run_id = Path(config_path).stem
    output_dir = Path("outputs") / "protein_lm" / run_id
    output_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # The tokenizer vocabulary size is needed for the model's embedding layer.
    tokenizer = ProteinTokenizer()
    lm_config.vocab_size = len(tokenizer.vocab)

    # --- 3. Data Loading ---
    # Using a helper function to create dataloaders keeps this script cleaner.
    num_workers = os.cpu_count() // 2
    train_loader = create_dataloader(
        data_config['train_path'],
        training_config['batch_size'],
        num_workers=num_workers,
        tokenizer=tokenizer,
        block_size=lm_config.block_size,
        shuffle=True
    )
    val_loader = create_dataloader(
        data_config['val_path'],
        training_config['batch_size'],
        num_workers=num_workers,
        tokenizer=tokenizer,
        block_size=lm_config.block_size,
        shuffle=False
    )

    # --- 4. Model Initialization ---
    model = ProteinConditionalTransformer(lm_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config['lr'],
        weight_decay=training_config.get('weight_decay', 0.01)
    )
    # The ignore_index parameter tells the loss function to ignore padding tokens.
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # A learning rate scheduler can help improve training performance.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_config['epochs'])

    # --- 5. Training Loop ---
    for epoch in range(training_config['epochs']):
        model.train()
        for i, batch in enumerate(train_loader):
            input_ids = batch.to(device)
            # The targets are the input sequence shifted by one token to the left.
            targets = input_ids[:, 1:].contiguous()
            # The model's input is the sequence up to the second to last token.
            logits = model(input_ids[:, :-1]).contiguous()

            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Gradient accumulation allows for a larger effective batch size.
            loss = loss / training_config.get('grad_accum_steps', 1)
            loss.backward()

            if (i + 1) % training_config.get('grad_accum_steps', 1) == 0:
                optimizer.step()
                optimizer.zero_grad()

            if i % 100 == 0:
                print(f"Epoch {epoch+1}/{training_config['epochs']}, Step {i}, Loss: {loss.item():.4f}")
        
        scheduler.step()

        # --- 6. Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch.to(device)
                targets = input_ids[:, 1:].contiguous()
                logits = model(input_ids[:, :-1]).contiguous()
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                val_loss += loss.item()

        val_loss /= len(val_loader)
        perplexity = torch.exp(torch.tensor(val_loss))
        print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Val PPL: {perplexity:.4f}")

        # --- 7. Checkpointing ---
        # Saving the model allows for resuming training later.
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, output_dir / f"checkpoint_epoch_{epoch+1}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a protein language model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config YAML file.")
    args = parser.parse_args()
    train(args.config)