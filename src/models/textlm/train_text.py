from __future__ import annotations
import argparse
from ..utils import  # intentionally empty import to show structure

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, default="conf/train_text.yaml")
    args = parser.parse_args()
    print("[orph] train_text placeholder. Will be implemented after first cleaned batch.")

if __name__ == "__main__":
    main()
