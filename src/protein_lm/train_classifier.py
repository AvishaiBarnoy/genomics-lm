
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import argparse
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

from src.protein_lm.config import ProteinClassifierConfig, load_config
from src.protein_lm.tokenizer import ProteinTokenizer
from src.protein_lm.models import ProteinClassifier
from src.protein_lm.data import create_dataloader, ProteinClassificationDataset

def train_classifier(config_path: str):
    # --- Load Config ---
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    classifier_config = load_config(config_path, ProteinClassifierConfig)
    training_config = config_data.get('training', {})
    data_config = config_data.get('data', {})
    
    # --- Setup ---
    run_id = Path(config_path).stem
    output_dir = Path("outputs") / "protein_classifier" / run_id
    output_dir.mkdir(exist_ok=True, parents=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = ProteinTokenizer()
    classifier_config.vocab_size = len(tokenizer.vocab)
    
    # --- Data ---
    train_loader = create_dataloader(
        data_config['train_path'],
        training_config['batch_size'],
        num_workers=4,
        tokenizer=tokenizer,
        block_size=classifier_config.block_size,
        shuffle=True,
        dataset_class=ProteinClassificationDataset,
        label_field='func_label' # Assumes 'func_label' is the target
    )
    val_loader = create_dataloader(
        data_config['val_path'],
        training_config['batch_size'],
        num_workers=4,
        tokenizer=tokenizer,
        block_size=classifier_config.block_size,
        shuffle=False,
        dataset_class=ProteinClassificationDataset,
        label_field='func_label'
    )

    # --- Model ---
    model = ProteinClassifier(classifier_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=training_config['lr'], 
        weight_decay=training_config.get('weight_decay', 0.01)
    )
    criterion = nn.CrossEntropyLoss()
    
    # --- Training Loop ---
    for epoch in range(training_config['epochs']):
        model.train()
        for i, (input_ids, labels) in enumerate(train_loader):
            input_ids, labels = input_ids.to(device), labels.to(device)
            
            logits = model(input_ids)
            loss = criterion(logits, labels)
            
            loss.backward()
            if (i + 1) % training_config.get('grad_accum_steps', 1) == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            if i % 100 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}")

        # --- Validation ---
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids, labels = input_ids.to(device), labels.to(device)
                logits = model(input_ids)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        print(f"Epoch {epoch}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        # --- Save Checkpoint ---
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, output_dir / f"checkpoint_epoch_{epoch}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config YAML file.")
    args = parser.parse_args()
    train_classifier(args.config)
