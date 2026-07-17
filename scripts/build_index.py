"""Run this whenever docs/policies/*.md changes.

    python -m scripts.build_index            # structure-aware (default, what chat uses)
    python -m scripts.build_index --naive     # naive fixed-size chunking, for comparison
"""

import argparse

from app import rag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--naive", action="store_true", help="use naive fixed-size chunking instead")
    args = parser.parse_args()

    strategy = "naive" if args.naive else "structured"
    collection = rag.build_index(strategy=strategy)
    print(f"Indexed {collection.count()} chunks using '{strategy}' chunking into {rag.config.CHROMA_DIR}")


if __name__ == "__main__":
    main()