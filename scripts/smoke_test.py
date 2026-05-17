from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.char_lstm import run_demo


def main() -> None:
    result = run_demo()
    assert result["vocab_size"] > 10
    assert result["characters"] > 100
    assert result["final_loss"] > 0
    assert len(result["sample"]) > 20
    print(result)


if __name__ == "__main__":
    main()

