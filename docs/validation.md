# Validation

Checks run before publishing this portfolio repo:

```bash
python3 -m compileall src scripts
python3 scripts/smoke_test.py
python3 -m src.char_lstm
```

Notebook code cells were also syntax-checked with `nbformat` to catch broken Python cells without re-running the full WikiText-2 training workflow.

Latest smoke-test result:

```text
vocab_size=32
characters=378
initial_loss=3.4587
final_loss=3.1218
```

The smoke test is intentionally lightweight. The full notebook remains the traceable artifact for the 10.9M-character WikiText-2 run and reported validation-loss improvement.
