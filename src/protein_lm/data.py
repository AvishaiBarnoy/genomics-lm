
import torch
from torch.utils.data import Dataset, DataLoader
import json
from src.protein_lm.tokenizer import ProteinTokenizer

class ProteinDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: ProteinTokenizer, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.samples = []

        with open(file_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sequence = sample['sequence']
        
        # Convert labels to condition tokens
        conditions = []
        if 'func_label' in sample:
            conditions.append(f"<FUNC:{sample['func_label'].upper()}>")
        if 'topo_label' in sample:
            conditions.append(f"<TOPO:{sample['topo_label'].upper()}>")
            
        # Encode everything
        condition_ids = self.tokenizer.encode_conditions(conditions)
        sequence_ids = self.tokenizer.encode_sequence(sequence)
        
        # Combine into final input_ids
        input_ids = (
            [self.tokenizer.bos_token_id] + 
            condition_ids + 
            sequence_ids
        )
        
        # Pad or truncate
        if len(input_ids) < self.block_size:
            padding = [self.tokenizer.pad_token_id] * (self.block_size - len(input_ids))
            input_ids += padding
        else:
            input_ids = input_ids[:self.block_size]
            
        return torch.tensor(input_ids, dtype=torch.long)


def create_dataloader(split_path, batch_size, num_workers, tokenizer, block_size, shuffle=True):
    dataset = ProteinDataset(split_path, tokenizer, block_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )
