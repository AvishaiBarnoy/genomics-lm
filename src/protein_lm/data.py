import torch
from torch.utils.data import Dataset, DataLoader
import json
from src.protein_lm.tokenizer import ProteinTokenizer

class ProteinDataset(Dataset):
    """
    A dataset for loading protein sequences and their conditional labels from a JSONL file.
    """
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

        # Convert labels from the JSONL file to the condition tokens the tokenizer expects.
        conditions = []
        if 'func_label' in sample:
            conditions.append(f"<FUNC:{sample['func_label'].upper()}>")
        if 'topo_label' in sample:
            conditions.append(f"<TOPO:{sample['topo_label'].upper()}>")

        # Encode the conditions and the sequence into token IDs.
        condition_ids = self.tokenizer.encode_conditions(conditions)
        sequence_ids = self.tokenizer.encode_sequence(sequence)

        # The final input sequence is [BOS] token, followed by conditions, then the sequence itself.
        input_ids = (
            [self.tokenizer.bos_token_id] +
            condition_ids +
            sequence_ids
        )

        # Pad the sequence to a fixed length (block_size) or truncate it if it's too long.
        if len(input_ids) < self.block_size:
            padding = [self.tokenizer.pad_token_id] * (self.block_size - len(input_ids))
            input_ids += padding
        else:
            input_ids = input_ids[:self.block_size]

        return torch.tensor(input_ids, dtype=torch.long)


def create_dataloader(
    split_path,
    batch_size,
    num_workers,
    tokenizer,
    block_size,
    shuffle=True,
    dataset_class=ProteinDataset,
    label_field=None,
    label_map=None,
    generator=None,
):
    """
    Creates a DataLoader for a given dataset split.
    Can be used for both language modeling and classification.
    """
    if dataset_class == ProteinClassificationDataset:
        dataset = dataset_class(
            split_path,
            tokenizer,
            block_size,
            label_field=label_field,
            label_map=label_map,
        )
    else:
        dataset = dataset_class(split_path, tokenizer, block_size)
        
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        generator=generator,
    )

class ProteinClassificationDataset(ProteinDataset):
    def __init__(
        self,
        file_path: str,
        tokenizer: ProteinTokenizer,
        block_size: int,
        label_field: str,
        label_map: dict | None = None,
    ):
        super().__init__(file_path, tokenizer, block_size)
        self.label_field = label_field

        observed_labels = sorted(
            {sample[self.label_field] for sample in self.samples if self.label_field in sample}
        )
        self.label_map = (
            dict(label_map)
            if label_map is not None
            else {label: index for index, label in enumerate(observed_labels)}
        )
        unknown = sorted(set(observed_labels).difference(self.label_map))
        if unknown:
            raise ValueError(
                f"{file_path} contains labels absent from the training label map: {unknown}"
            )
        self.labels = [
            label for label, _ in sorted(self.label_map.items(), key=lambda item: item[1])
        ]
        print(f"Found labels: {self.label_map}")

    def __getitem__(self, idx):
        # We need to get the input_ids from the parent class, but without the label processing.
        # Let's re-implement the __getitem__ for the classification dataset to be cleaner.
        sample = self.samples[idx]
        sequence = sample['sequence']

        conditions = []
        if 'func_label' in sample and self.label_field != 'func_label':
            conditions.append(f"<FUNC:{sample['func_label'].upper()}>")
        if 'topo_label' in sample and self.label_field != 'topo_label':
            conditions.append(f"<TOPO:{sample['topo_label'].upper()}>")

        condition_ids = self.tokenizer.encode_conditions(conditions)
        sequence_ids = self.tokenizer.encode_sequence(sequence)

        input_ids = (
            [self.tokenizer.bos_token_id] +
            condition_ids +
            sequence_ids
        )

        if len(input_ids) < self.block_size:
            padding = [self.tokenizer.pad_token_id] * (self.block_size - len(input_ids))
            input_ids += padding
        else:
            input_ids = input_ids[:self.block_size]
            
        if self.label_field not in sample:
            raise ValueError(
                f"sample {idx} has no required label field {self.label_field!r}"
            )
        label = sample[self.label_field]

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(
            self.label_map[label], dtype=torch.long
        )
