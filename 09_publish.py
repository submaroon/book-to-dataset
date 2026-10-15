# -*- coding: utf-8 -*-
"""Step 09 — final checks, optional split, dataset card, upload to the Hub.

Four things happen here, in order:

  1. Validation. Every session is checked against the rules a trainer will not
     check for you: at least one assistant turn, no trailing user turn, no
     empty messages, no leftover {{macros}}. Broken sessions are reported and
     excluded; the reasons are worth reading, because a systematic one points
     back at an earlier step.
  2. An optional train/eval split, held out at the session level.
  3. A dataset card, generated from the actual numbers of this run.
  4. The upload, which needs huggingface_hub and a token.

Run it with --dry-run first. That does everything except the upload, so you can
read the card and the statistics before anything becomes public.

    python 09_publish.py --dry-run
    python 09_publish.py

Authentication, once per machine:
    pip install huggingface_hub
    hf auth login          # older versions: huggingface-cli login
"""
import os
import random
import re
import sys
from collections import Counter

from pipeline.config import load_config, parse_args
from pipeline.files import read_jsonl, write_jsonl
from pipeline.sharegpt import session_stats
from pipeline.textstats import percentile

LEFTOVER_MACRO = re.compile(r"\{\{.*?\}\}|<USER>|<CHAR>")


def validate_sessions(rows):
    """Split rows into (valid, [(index, reason)])."""
    valid, broken = [], []
    for index, row in enumerate(rows):
        conversation = row.get("conversations")
        if not conversation:
            broken.append((index, "no conversations field"))
            continue
        if any(not str(m.get("value", "")).strip() for m in conversation):
            broken.append((index, "empty message"))
            continue
        roles = [m.get("from") for m in conversation]
        if "gpt" not in roles:
            broken.append((index, "no assistant turn"))
            continue
        if roles[-1] == "human":
            broken.append((index, "ends on a user turn"))
            continue
        non_system = [r for r in roles if r != "system"]
        if non_system and non_system[0] != "human":
            broken.append((index, "does not start on a user turn"))
            continue
        leftover = next((m for m in conversation
                         if LEFTOVER_MACRO.search(str(m.get("value", "")))), None)
        if leftover:
            broken.append((index, "leftover template macro"))
            continue
        valid.append(row)
    return valid, broken


def build_card(cfg, stats, n_train, n_eval, extra_notes=""):
    """Write a dataset card from the numbers this run actually produced."""
    publish = cfg["publish"]
    lengths = stats["gpt_words"]
    lines = [
        "---",
        f"license: {publish['license']}",
        "task_categories:",
        "  - text-generation",
        "language:",
        f"  - {cfg['lang']}",
        "tags:",
        "  - roleplay",
        "  - sharegpt",
        "  - synthetic",
        "---",
        "",
        "# Roleplay dataset",
        "",
        "Built with the roleplay dataset pipeline: a source text is segmented into "
        "scenes, the scenes are converted into roleplay transcripts, synthetic "
        "sessions are generated from the same material, and the result is varied "
        "across formatting axes.",
        "",
        "## Format",
        "",
        "ShareGPT JSONL. One line is one session:",
        "",
        "```json",
        '{"conversations": [',
        '  {"from": "system", "value": "..."},',
        '  {"from": "human",  "value": "..."},',
        '  {"from": "gpt",    "value": "..."}',
        "]}",
        "```",
        "",
        "Train with `train_on_inputs: false` so the loss is computed on the `gpt` "
        "turns only.",
        "",
        "## Contents",
        "",
        f"- Sessions: **{stats['sessions']}**",
        f"- Messages: **{stats['messages']}**",
        f"- Distinct system blocks: **{stats['unique_system_blocks']}**",
        f"- Language: **{cfg['lang']}**",
    ]
    if lengths:
        lines += [
            f"- Assistant turn length in words: p10 {percentile(lengths, .1)}, "
            f"median {percentile(lengths, .5)}, p90 {percentile(lengths, .9)}",
        ]
    if n_eval:
        lines += [f"- Splits: train **{n_train}**, eval **{n_eval}**"]

    lines += [
        "",
        "## How it was made",
        "",
        "1. The source text is segmented into scenes, structurally where markers "
        "exist and with a model where they do not.",
        "2. Speakers are attributed per line.",
        "3. Scenes become transcripts, with the narration carried over rather than "
        "retold.",
        "4. Transcripts are filtered on lexical overlap with their source scene and "
        "checked by a model judge.",
        "5. Synthetic sessions are generated, anchored to real scenes, with "
        "anti-copying and deduplication filters.",
        "6. Sessions are rewritten across formatting axes, with the directives "
        "written into the system block, so the model learns to read it.",
        "",
        "## Caveats",
        "",
        "- Parts of this dataset are model-generated and were not reviewed line by "
        "line. Sample it before you train on it.",
        "- The style is bound to one source text. Mixing in general roleplay data "
        "is recommended; see the pipeline's diluent option.",
    ]
    if extra_notes:
        lines += ["", extra_notes]
    return "\n".join(lines) + "\n"


def upload(cfg, files, card_path):
    """Push the dataset files and the card to the Hub."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("\nhuggingface_hub is not installed: pip install huggingface_hub\n")

    repo_id = cfg["publish"]["repo_id"]
    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset",
                        private=cfg["publish"]["private"], exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"\nCould not create or access {repo_id}: {exc}\n"
                 f"Check that you are logged in (hf auth login) and that the name is right.\n")

    for path in files:
        print(f"  uploading {path} ...")
        api.upload_file(path_or_fileobj=path, path_in_repo=os.path.basename(path),
                        repo_id=repo_id, repo_type="dataset")
    print("  uploading README.md ...")
    api.upload_file(path_or_fileobj=card_path, path_in_repo="README.md",
                    repo_id=repo_id, repo_type="dataset")
    visibility = "private" if cfg["publish"]["private"] else "public"
    print(f"\nDone: https://huggingface.co/datasets/{repo_id}  ({visibility})")


def main():
    def extra(parser):
        parser.add_argument("--dry-run", action="store_true",
                            help="validate, split and write the card, but do not upload")
        parser.add_argument("--input", default=None,
                            help="dataset file to publish (default: assemble.output)")

    args = parse_args(__doc__.splitlines()[0], extra)
    cfg = load_config(args.config)
    publish = cfg["publish"]

    source = args.input or cfg["assemble"]["output"]
    if not os.path.exists(source):
        sys.exit(f"\n{source} not found. Run 08_assemble.py first, or pass --input PATH.\n")

    rows = read_jsonl(source)
    print(f"Loaded {len(rows)} sessions from {source}")

    valid, broken = validate_sessions(rows)
    if broken:
        reasons = Counter(reason for _index, reason in broken)
        print(f"\n{len(broken)} sessions failed validation and were excluded:")
        for reason, count in reasons.most_common():
            print(f"  {count:>5}  {reason}")
        print("  A reason that dominates points back at an earlier step — worth fixing "
              "there rather than dropping the rows here.")
    assert valid, "every session failed validation — nothing to publish"
    print(f"Valid sessions: {len(valid)}")

    # Split
    rng = random.Random(publish["seed"])
    shuffled = list(valid)
    rng.shuffle(shuffled)
    n_eval = int(len(shuffled) * publish["eval_share"])
    # Outputs go next to the dataset they came from, not into the current
    # directory, so a run driven by a config in another folder does not scatter
    # files where you did not expect them.
    folder = os.path.dirname(os.path.abspath(source))
    if n_eval:
        eval_rows, train_rows = shuffled[:n_eval], shuffled[n_eval:]
        train_path = os.path.join(folder, "dataset_train.jsonl")
        eval_path = os.path.join(folder, "dataset_eval.jsonl")
        write_jsonl(train_path, train_rows)
        write_jsonl(eval_path, eval_rows)
        files = [train_path, eval_path]
        print(f"Split: train {len(train_rows)} -> {train_path}, "
              f"eval {len(eval_rows)} -> {eval_path}")
    else:
        clean_path = os.path.join(folder, "dataset_publish.jsonl")
        write_jsonl(clean_path, shuffled)
        files = [clean_path]
        print(f"No eval split (publish.eval_share = 0) -> {clean_path}")

    stats = session_stats(valid)
    card = build_card(cfg, stats, len(shuffled) - n_eval, n_eval)
    # Next to the dataset, like the split files. A bare filename in the config
    # would otherwise be shared by every config in the project, and two runs
    # would overwrite each other's card.
    card_path = publish["card_output"]
    if not os.path.isabs(card_path) and os.sep not in card_path:
        card_path = os.path.join(folder, card_path)
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card)
    print(f"Dataset card -> {card_path}")

    print(f"\nSessions {stats['sessions']}, messages {stats['messages']}, "
          f"distinct system blocks {stats['unique_system_blocks']}")

    if args.dry_run:
        print("\n[--dry-run] Nothing was uploaded. Read the card and the files above, "
              "then rerun without --dry-run.")
        return
    if not publish["repo_id"]:
        sys.exit("\npublish.repo_id is empty. Set it to 'your-username/dataset-name' "
                 "in the config, or keep using --dry-run.\n")

    print(f"\nUploading to {publish['repo_id']} ...")
    upload(cfg, files, card_path)


if __name__ == "__main__":
    main()
