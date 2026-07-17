"""Compare naive vs. structure-aware chunking on a question that spans a
section boundary. Builds two throwaway indexes in separate temp dirs so
neither touches your real chat index.

    python -m scripts.chunking_demo "What happens if my item is overdue?"
"""

import sys
import tempfile
from pathlib import Path

from app import rag

DEFAULT_QUESTION = "If I renew an item but then it goes overdue anyway, what happens to it?"


def run(question: str):
    print(f"Question: {question}\n")

    for strategy in ("naive", "structured"):
        with tempfile.TemporaryDirectory() as tmp:
            rag.build_index(strategy=strategy, persist_dir=Path(tmp))
            results = rag.query(question, k=3, persist_dir=Path(tmp))
            print(f"--- {strategy} ---")
            for i, r in enumerate(results, 1):
                heading = f" [{r['heading']}]" if r["heading"] else ""
                print(f"{i}. ({r['source']}{heading})")
                print(f"   {r['text'][:220]!r}")
            print()


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    run(q)