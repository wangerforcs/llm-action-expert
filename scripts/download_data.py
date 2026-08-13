#!/usr/bin/env python
"""Download a Hugging Face dataset as the JSON array expected by this prototype."""
import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Salesforce/APIGen-MT-5k")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    ds = load_dataset(args.dataset, split=args.split)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([dict(row) for row in ds], ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(ds)} records to {out}")


if __name__ == "__main__":
    main()
