# -*- coding: utf-8 -*-
"""Merge several dataset files into one, shuffled.

This is how a multilingual dataset is built. Each language is a separate run
with its own config and its own workdir; the runs never mix inside the pipeline,
because a prompt in one language and a scene in another produce nonsense. They
are merged here, at the very end.

    python tools/merge_datasets.py out.jsonl ru/dataset_final.jsonl en/dataset_final.jsonl

    --balance   take the same number of sessions from every file, cut to the
                smallest one. Without it, the largest file dominates and the
                adapter leans towards its language.
    --seed N    shuffle seed (default 42)

Each session stays monolingual. That is the point: a session that switches
language mid-conversation teaches exactly that.
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.files import read_jsonl, write_jsonl  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", help="file to write")
    parser.add_argument("inputs", nargs="+", help="two or more .jsonl files")
    parser.add_argument("--balance", action="store_true",
                        help="take an equal number of sessions from each input")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    pools = []
    for path in args.inputs:
        if not os.path.exists(path):
            sys.exit(f"{path} not found")
        rows = read_jsonl(path)
        assert rows, f"{path} is empty"
        print(f"{path}: {len(rows)} sessions")
        pools.append((path, rows))

    if args.balance and len(pools) > 1:
        smallest = min(len(rows) for _path, rows in pools)
        print(f"\n--balance: taking {smallest} from each file")
        merged = []
        for path, rows in pools:
            picked = rng.sample(rows, smallest)
            merged += picked
            if len(rows) > smallest:
                print(f"  {path}: dropped {len(rows) - smallest}")
    else:
        merged = [row for _path, rows in pools for row in rows]

    rng.shuffle(merged)
    write_jsonl(args.output, merged)
    print(f"\nDone: {len(merged)} sessions -> {args.output}")
    if not args.balance and len(pools) > 1:
        biggest = max(pools, key=lambda item: len(item[1]))
        share = 100 * len(biggest[1]) // max(1, len(merged))
        if share > 65:
            print(f"\n{biggest[0]} is {share}% of the result. If these are different "
                  f"languages, the adapter will lean that way — consider --balance.")


if __name__ == "__main__":
    main()
