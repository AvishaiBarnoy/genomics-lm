
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



class ProteinClassificationDataset(ProteinDataset):
    def __init__(self, file_path: str, tokenizer: ProteinTokenizer, block_size: int, label_field: str):
        super().__init__(file_path, tokenizer, block_size)
        self.label_field = label_field
        
        # Build label map dynamically
        self.labels = sorted(list(set(s[self.label_field] for s in self.samples if self.label_field in s)))
        self.label_map = {label: i for i, label in enumerate(self.labels)}

    def __getitem__(self, idx):
        input_ids = super().__getitem__(idx)
        sample = self.samples[idx]
        label = sample.get(self.label_field)
        return input_ids, torch.tensor(self.label_map.get(label, -1), dtype=torch.long)


def create_dataloader(split_path, batch_size, num_workers, tokenizer, block_size, shuffle=True, dataset_class=ProteinDataset, label_field=None):
    if dataset_class == ProteinClassificationDataset:
        dataset = dataset_class(split_path, tokenizer, block_size, label_field=label_field)
    else:
        dataset = dataset_class(split_path, tokenizer, block_size)
        
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )
