import json
import torch
from torch.utils.data import Dataset, Sampler


class MultiTaskProteinDataset(Dataset):
    def __init__(
        self,
        jsonl_path,
        tokenizer,
        max_length=512,
        dynamic_padding=False,
        multi_label_tasks=None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dynamic_padding = dynamic_padding
        self.multi_label_tasks = set(multi_label_tasks or [])
        self.samples = []

        with open(jsonl_path, "r") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Tokenize (Add BOS, pad/truncate, Add EOS handled by tokenizer logic if needed)
        tokens = (
            [self.tokenizer.bos_token_id]
            + self.tokenizer.encode_sequence(s["sequence"])[: self.max_length - 2]
            + [self.tokenizer.eos_token_id]
        )

        attention_mask = [1] * len(tokens)
        if not self.dynamic_padding:
            pad_len = self.max_length - len(tokens)
            input_ids = tokens + [self.tokenizer.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
        else:
            input_ids = tokens

        if "stability_score" in s:
            stability = torch.tensor(
                float(s["stability_score"])
                if s["stability_score"] is not None
                else float("nan"),
                dtype=torch.float32,
            )
        else:
            stability = torch.tensor(s.get("stability_id", -1), dtype=torch.long)

        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "sequence": s["sequence"],
            "family": torch.tensor(s.get("pfam_id", -1), dtype=torch.long),
            "function": torch.tensor(s.get("ec_id", -1), dtype=torch.long),
            "stability": stability,
        }
        for task in self.multi_label_tasks:
            labels = s.get(task)
            if labels is None:
                labels = s.get(f"{task}_labels")
            if labels is None:
                labels = []
            item[task] = torch.tensor(labels, dtype=torch.float32)
        return item

    def sequence_length(self, idx):
        sequence = self.samples[idx]["sequence"]
        return min(len(sequence) + 2, self.max_length)


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Batch similar-length proteins together to reduce padding waste."""

    def __init__(self, dataset, batch_size, shuffle=True, seed=1337):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = list(range(len(self.dataset)))
        indices.sort(key=self.dataset.sequence_length)
        batches = [
            indices[i : i + self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]
        if self.shuffle:
            order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[i] for i in order]
        yield from batches

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def collate_protein_batch(batch, pad_token_id=0):
    """Pad a variable-length protein batch to its local max length."""
    max_len = max(item["input_ids"].numel() for item in batch)
    result = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if key in {"input_ids", "attention_mask"}:
            pad_value = pad_token_id if key == "input_ids" else 0
            out = torch.full((len(batch), max_len), pad_value, dtype=values[0].dtype)
            for i, value in enumerate(values):
                out[i, : value.numel()] = value
            result[key] = out
        elif key == "sequence":
            result[key] = values
        else:
            result[key] = torch.stack(values)
    return result
