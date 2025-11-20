
from dataclasses import dataclass
import yaml

@dataclass
class ProteinLMConfig:
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    dropout: float

@dataclass
class ProteinClassifierConfig:
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    dropout: float
    num_classes: int

def load_config(path: str, config_class):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Assuming the config is under a 'model' key
    model_data = data.get('model', {})
    
    # Filter only the arguments that the config_class expects
    expected_fields = {f.name for f in fields(config_class)}
    filtered_data = {k: v for k, v in model_data.items() if k in expected_fields}

    return config_class(**filtered_data)

from dataclasses import fields
