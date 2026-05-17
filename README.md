# Architecting Intelligence: Character-Level LSTM

Public project portfolio for my MatSoc IIT Kanpur winter project **Architecting Intelligence**.

**Period:** December 2025 - January 2026  
**Focus:** neural-network fundamentals, NLP, sequence modeling, and generative language modeling

## Recruiter Snapshot

- Implemented a complete **character-level LSTM language model** in PyTorch.
- Worked with **WikiText-2 raw**: roughly **10.9M training characters** and a **1,014-character vocabulary**.
- Built the end-to-end workflow: character encoding, batch sampling, model class, hidden-state handling, training loop, validation, loss visualization, and text generation.
- Tested **temperature sampling** to compare safe/repetitive generation against more creative but noisier outputs.

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

## Resume Claim Traceability

| Resume claim | Where to verify |
| --- | --- |
| PyTorch character-level LSTM | `notebooks/char_lstm_wikitext.ipynb`, `CharLSTM` class |
| WikiText-2, 10.9M characters, 1,014 vocabulary | Data loading and preprocessing cells |
| Validation loss 3.17 to 2.14 | Training output cells |
| Temperature-based generation | Sampling/probing cells |

## Repository Structure

```text
src/
  char_lstm.py                    # Reusable PyTorch model, training loop, and generator

scripts/
  smoke_test.py                   # Fast local validation on a bundled mini corpus

notebooks/
  char_lstm_wikitext.ipynb        # Complete PyTorch LSTM implementation

docs/
  assignment_prompt.pdf           # Original assignment prompt
  assignment_submission.pdf       # Submitted report/export
  project_summary.md              # Short reviewer-facing explanation
  validation.md                   # Local checks run before publishing

requirements.txt                  # Python environment outline
```

## Run Locally

```bash
python3 -m pip install -r requirements.txt
python3 -m compileall src scripts
python3 scripts/smoke_test.py
python3 -m src.char_lstm
```

The script uses a small bundled corpus so reviewers can verify the modeling code quickly without downloading WikiText-2. The notebook contains the full project run on WikiText-2.

## How To Review

For a quick review, open `docs/project_summary.md` first, then `docs/validation.md`. For technical depth, open `notebooks/char_lstm_wikitext.ipynb` and scan:

1. Dataset loading and character vocabulary construction.
2. `CharLSTM` model definition.
3. Training loop with validation loss checkpoints.
4. Temperature-based generation examples.

## Notes

This is a learning and implementation project, not a production LLM. The value is in showing that I can implement the modeling loop, reason about sequence behavior, and document model limitations clearly.

## Tech Stack

Python, PyTorch, Hugging Face Datasets, NumPy, Matplotlib
