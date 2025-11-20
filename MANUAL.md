# Protein Language Model (`protein_lm`) Manual

This document provides instructions on how to use the `protein_lm` module, which includes a function-conditional protein language model and a protein classifier.

## Overview

The `protein_lm` module is designed to model protein sequences and their functional properties. It consists of two main components:

1.  **`ProteinConditionalTransformer`**: A language model that learns to predict the next amino acid in a sequence, conditioned on functional or topological labels.
2.  **`ProteinClassifier`**: A classifier that uses the language model's architecture to predict the functional class of a given protein sequence.

## Tokenization

The tokenizer, located in `src/protein_lm/tokenizer.py`, is responsible for converting protein sequences and condition tokens into integer IDs that can be fed into the models.

-   **Vocabulary**: Includes 20 standard amino acids, an 'X' token for unknown residues, special tokens (`<BOS>`, `<EOS>`, `<PAD>`), and condition tokens (e.g., `<FUNC:ENZYME>`, `<TOPO:TM>`).
-   **Input Format**: The models expect input in the format `[BOS] + [condition_ids] + [sequence_ids]`.

## Configuration

Model architecture, training parameters, and data paths are defined in YAML configuration files located in `configs/protein_lm/`.

-   `small.yaml`: A sample configuration for training the language model.
-   `classifier_small.yaml`: A sample configuration for training the classifier. This file is similar to `small.yaml` but includes a `num_classes` parameter for the classification head.

## Training

The module includes two training scripts in `src/protein_lm/`.

### Language Model Training

To train the language model, run the `train_lm.py` script with a configuration file:

```bash
python -m src.protein_lm.train_lm --config configs/protein_lm/small.yaml
```

The script will:
1.  Load the model configuration and training parameters.
2.  Initialize the `ProteinConditionalTransformer` model.
3.  Load the training and validation data using the `ProteinDataset`.
4.  Train the model using cross-entropy loss to predict the next token.
5.  Save checkpoints to `outputs/protein_lm/<run_id>/`.

### Classifier Training

To train the classifier, run the `train_classifier.py` script:

```bash
python -m src.protein_lm.train_classifier --config configs/protein_lm/classifier_small.yaml
```

This script will:
1.  Load the classifier configuration.
2.  Initialize the `ProteinClassifier` model.
3.  Load the training and validation data using the `ProteinClassificationDataset`, which is designed to handle class labels.
4.  Train the model using cross-entropy loss for classification.
5.  Log validation accuracy and F1 score.
6.  Save checkpoints to `outputs/protein_classifier/<run_id>/`.