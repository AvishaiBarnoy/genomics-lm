from dataclasses import dataclass, fields
import yaml

@dataclass
class ProteinLMConfig:
    """Configuration for the Protein Language Model."""
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    dropout: float

@dataclass
class ProteinClassifierConfig:
    """Configuration for the Protein Classifier."""
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    dropout: float
    num_classes: int

def load_config(path: str, config_class):
    """
    Loads a model configuration from a YAML file.

    Args:
        path: The path to the YAML file.
        config_class: The dataclass to instantiate (e.g., ProteinLMConfig).

    Returns:
        An instance of the provided config_class.
    """
    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    # The configuration is expected to be under a 'model' key in the YAML
    model_data = data.get('model', {})

    # Filter the loaded data to include only the fields expected by the dataclass
    expected_fields = {f.name for f in fields(config_class)}
    filtered_data = {k: v for k, v in model_data.items() if k in expected_fields}

    return config_class(**filtered_data)