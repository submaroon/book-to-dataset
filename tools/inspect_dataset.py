# -*- coding: utf-8 -*-
"""Look at a finished ShareGPT dataset before training on it.

Three modes:

    python tools/inspect_dataset.py --sample 10 dataset_final.jsonl
        Print random sessions with their format directives, so you can check
        that the answers actually follow what the system block promised.

    python tools/inspect_dataset.py --stats dataset_final.jsonl
        Session counts, turn lengths, how many distinct system blocks there are.

    python tools/inspect_dataset.py --health dataset_final.jsonl
        Look for generator garbage: runaway repetition, degenerate endings,
        messages far outside the normal length range.

Sampling is the one that catches real problems. The statistics look fine on a
dataset full of subtly softened rewrites; reading ten of them does not.
"""
import argparse
import hashlib
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.files import read_jsonl  # noqa: E402
from pipeline.textstats import percentile  # noqa: E402


def sample(path, count):
    rows = read_jsonl(path)
    assert rows, f"{path} is empty"
    for number, row in enumerate(random.Random(0).sample(rows, min(count, len(rows))), 1):
        conversation = row.get("conversations") or []
        system = next((m["value"] for m in conversation if m["from"] == "system"), "")
        directives = (system.split("<format>")[1].split("</format>")[0].strip()
                      if "<format>" in system else "— no directives (plain variant)")
        assistant = [m["value"] for m in conversation if m["from"] == "gpt"]
        print("=" * 74)
        print(f"[{number}]  assistant turns: {len(assistant)}")
        print("-" * 74)
        print("FORMAT:")
        print(directives)
        print("-" * 74)
        for text in assistant[:2]:
            print(text[:600] + ("..." if len(text) > 600 else ""))
            print()
    print("=" * 74)
    print("Two things to check: does the formatting match what FORMAT promised, and "
          "has the text been softened relative to your source.")


def stats(path):
    rows = read_jsonl(path)
    assert rows, f"{path} is empty"
    lengths, system_blocks, roles = [], Counter(), Counter()
    turns_per_session = []
    for row in rows:
        conversation = row.get("conversations") or []
        turns_per_session.append(sum(1 for m in conversation if m["from"] != "system"))
        for message in conversation:
            roles[message["from"]] += 1
            if message["from"] == "system":
                system_blocks[hashlib.sha1(message["value"].encode()).hexdigest()[:8]] += 1
            elif message["from"] == "gpt":
                lengths.append(len(message["value"].split()))
    lengths.sort()
    turns_per_session.sort()

    print(f"File: {path}")
    print(f"Sessions: {len(rows)}")
    print(f"Messages: {sum(roles.values())}  "
          f"({', '.join(f'{k} {v}' for k, v in roles.most_common())})")
    print(f"Distinct system blocks: {len(system_blocks)}")
    if lengths:
        print(f"Assistant turn length in words: p10 {percentile(lengths, .1)}, "
              f"median {percentile(lengths, .5)}, p90 {percentile(lengths, .9)}, "
              f"max {lengths[-1]}")
    if turns_per_session:
        print(f"Turns per session: median {percentile(turns_per_session, .5)}, "
              f"max {turns_per_session[-1]}")
    if len(system_blocks) < len(rows) // 20:
        print("\nFew distinct system blocks relative to the session count. If you ran "
              "step 07, check that it accepted variants; if you did not, that is why.")


def health(path, max_chars=6000):
    """Find messages that look like generator garbage rather than text."""
    rows = read_jsonl(path)
    assert rows, f"{path} is empty"
    problems = []
    for row_index, row in enumerate(rows):
        for message_index, message in enumerate(row.get("conversations") or []):
            value = message.get("value", "")
            reason = None
            if len(value) > max_chars:
                reason = f"very long ({len(value)} chars)"
            elif len(value) > 200 and len(set(value[-200:])) < 8:
                # The tail collapsed into a handful of distinct characters: the
                # classic signature of a decoding loop.
                reason = "degenerate ending"
            else:
                words = value.split()
                if len(words) > 40:
                    most_common = Counter(words).most_common(1)[0]
                    if most_common[1] > len(words) * 0.25:
                        reason = f"word '{most_common[0]}' is {most_common[1]}/{len(words)} of the turn"
            if reason:
                problems.append((row_index, message_index, reason,
                                 value[:100].replace("\n", " ")))

    print(f"File: {path}")
    print(f"Suspicious messages: {len(problems)} out of {sum(len(r.get('conversations', [])) for r in rows)}")
    for row_index, message_index, reason, preview in problems[:20]:
        print(f"  session {row_index}, message {message_index}: {reason}")
        print(f"    {preview}...")
    if len(problems) > 20:
        print(f"  ... and {len(problems) - 20} more")
    if not problems:
        print("Nothing found. That is not proof the data is good — read some of it.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", help="ShareGPT .jsonl file")
    parser.add_argument("--sample", type=int, metavar="N", help="print N random sessions")
    parser.add_argument("--stats", action="store_true", help="print aggregate statistics")
    parser.add_argument("--health", action="store_true", help="look for generator garbage")
    args = parser.parse_args()

    if args.sample:
        sample(args.path, args.sample)
    elif args.health:
        health(args.path)
    else:
        stats(args.path)


if __name__ == "__main__":
    main()
