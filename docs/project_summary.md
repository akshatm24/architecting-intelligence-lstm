# Project Summary

## Context

Architecting Intelligence was a MatSoc IIT Kanpur winter project covering foundations of modern AI systems. The portfolio artifact here focuses on my most concrete implementation deliverable: a character-level LSTM language model.

## What I Built

- Loaded WikiText-2 raw through Hugging Face Datasets.
- Converted text into a character vocabulary of 1,014 tokens, including an unknown-token fallback.
- Implemented a PyTorch `CharLSTM` with embedding, stacked LSTM layers, dropout, fully connected output projection, and hidden-state initialization.
- Wrote batching and sequence-target generation for next-character prediction.
- Trained with Adam and cross-entropy loss.
- Compared generated text at multiple temperatures to understand the trade-off between repetitive and creative output.

## Key Learning

The model is intentionally small compared with modern transformer systems, but it demonstrates the sequence-modeling fundamentals: tokenization, temporal state, gradient-based training, validation loss tracking, and probabilistic sampling.

## Numbers Worth Reviewing

- Training text: 10,929,707 characters
- Vocabulary: 1,014 characters
- Architecture: 2-layer LSTM, 256 embedding size, 512 hidden size
- Validation loss: 3.17 to 2.14 over 300 training steps

