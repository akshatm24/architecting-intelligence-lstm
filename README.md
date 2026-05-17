# Architecting Intelligence: Character-Level LSTM

This repository packages my submission for the MatSoc IIT Kanpur winter project **Architecting Intelligence**, focused on modern AI foundations, neural networks, sequence modeling, and text generation.

## What This Project Shows

- Built a character-level language model in **PyTorch** using an LSTM architecture.
- Trained on the **WikiText-2 raw** dataset with roughly **10.9M training characters** and a **1,014-character vocabulary**.
- Implemented the full modeling loop: data encoding, batching, model definition, training, validation, loss tracking, and text generation.
- Used temperature sampling to compare conservative and creative generation behavior.

## Key Results

| Item | Detail |
| --- | --- |
| Dataset | WikiText-2 raw |
| Model | 2-layer character-level LSTM |
| Embedding size | 256 |
| Hidden size | 512 |
| Sequence length | 100 |
| Batch size | 128 |
| Optimizer | Adam |
| Training steps | 300 |
| Validation loss | Improved from **3.17** to **2.14** |

## Repository Structure

```text
notebooks/
  char_lstm_wikitext.ipynb        # Complete PyTorch LSTM implementation

docs/
  assignment_prompt.pdf           # Original assignment prompt
  assignment_submission.pdf       # Submitted report/export
```

## How To Review

Open `notebooks/char_lstm_wikitext.ipynb` and scan:

1. Dataset loading and character vocabulary construction.
2. `CharLSTM` model definition.
3. Training loop with validation loss checkpoints.
4. Temperature-based generation examples.

## Tech Stack

Python, PyTorch, Hugging Face Datasets, NumPy, Matplotlib

