# -*- coding: utf-8 -*-
"""Step 04 — validate the transcripts.

Two filters, cheapest first.

  1. Lexical overlap, no LLM involved: what share of the transcript's content
     words also occur in the source scene. A low share means the converter wrote
     its own prose. This catches the most common failure for free and runs over
     everything before a single model call is made.

  2. An LLM judge: is the narrative voice preserved, is anything invented, are
     the lines on the right characters. Optional — set validate.llm_check to
     false to keep only the cheap filter.

Rejections are kept, with their reasons. Read them: a systematic reason is a
prompt problem in step 03, and deleting the data instead of fixing the prompt
just makes a smaller dataset with the same flaw.

    Input:  <workdir>/rp_raw.jsonl, <workdir>/scenes.jsonl
    Output: <workdir>/rp_clean.jsonl, <workdir>/rp_rejected.jsonl

    python 04_validate.py [--config config.yaml]
"""
import sys
from collections import Counter

from pipeline.config import load_config, parse_args, work_path
from pipeline.files import append_jsonl, read_jsonl, require_file
from pipeline.lang import load_pack, render
from pipeline.llm import LLM, extract_json, make_refusal_checker, parallel_map
from pipeline.textstats import overlap


def main():
    args = parse_args(__doc__.splitlines()[0])
    cfg = load_config(args.config)
    pack = load_pack(cfg["lang"])
    validate = cfg["validate"]
    workers = cfg["llm"]["workers"]
    is_refusal = make_refusal_checker(pack.refusal_markers)

    scenes = {s["id"]: s for s in
              read_jsonl(require_file(work_path(cfg, "scenes.jsonl"), "01_segment.py"))}
    transcripts = read_jsonl(require_file(work_path(cfg, "rp_raw.jsonl"), "03_convert.py"))
    clean_path = work_path(cfg, "rp_clean.jsonl")
    reject_path = work_path(cfg, "rp_rejected.jsonl")

    done = {item["id"] for item in read_jsonl(clean_path)}
    done |= {item["id"] for item in read_jsonl(reject_path)}
    if done:
        print(f"[resume] {len(done)} already checked, continuing")

    # Pass 1: the cheap filter, over everything, before any model call.
    pending = []
    n_clean = n_rejected = 0
    reasons = Counter()
    for transcript in transcripts:
        if transcript["id"] in done:
            continue
        scene = scenes.get(transcript["source_id"])
        if not scene:
            append_jsonl(reject_path, {**transcript, "reject_reason": "no source scene"})
            reasons["no source scene"] += 1
            n_rejected += 1
            continue

        text = "\n".join(turn["text"] for turn in transcript["turns"])
        score = round(overlap(text, scene["text"]), 3)
        transcript["lexical_overlap"] = score
        transcript["_text"] = text
        if score < validate["lexical_overlap_min"]:
            append_jsonl(reject_path, {
                **{k: v for k, v in transcript.items() if k != "_text"},
                "reject_reason": f"lexical_overlap {score} < {validate['lexical_overlap_min']}"})
            reasons["low lexical overlap"] += 1
            n_rejected += 1
            print(f"rp {transcript['id']}: REJECT overlap={score}")
            continue
        pending.append(transcript)

    print(f"Passed the lexical filter: {len(pending)}")

    # Pass 2: the judge. Skipped entirely when disabled.
    if not validate["llm_check"]:
        for transcript in pending:
            transcript.pop("_text", None)
            append_jsonl(clean_path, transcript)
            n_clean += 1
        print(f"\n[validate.llm_check is off] {n_clean} transcripts accepted on the "
              f"lexical filter alone -> {clean_path}")
        return

    llm = LLM(cfg)

    def work(transcript):
        scene = scenes[transcript["source_id"]]
        return llm.chat(
            pack.require("validate.system"),
            render(pack.require("validate.user"),
                   scene=scene["text"], transcript=transcript["_text"]),
            temperature=validate["temperature"],
        )

    for transcript, raw, error in parallel_map(work, pending, workers):
        transcript.pop("_text", None)
        if error is not None:
            print(f"  [warn] rp {transcript['id']}: {error} — will retry on the next run",
                  file=sys.stderr)
            continue
        try:
            verdict = extract_json(raw)
        except Exception:  # noqa: BLE001
            if is_refusal(raw):
                print(f"rp {transcript['id']}: the judge refused — not marked, "
                      f"rerun with a different model")
            else:
                print(f"  [warn] rp {transcript['id']}: invalid JSON — will retry",
                      file=sys.stderr)
            continue

        transcript["verdict"] = verdict
        keys = ("voice_preserved", "no_invention", "split_correct")
        if all(verdict.get(key) for key in keys):
            append_jsonl(clean_path, transcript)
            n_clean += 1
            print(f"rp {transcript['id']}: OK overlap={transcript['lexical_overlap']}")
        else:
            failed = [key for key in keys if not verdict.get(key)]
            append_jsonl(reject_path, {
                **transcript,
                "reject_reason": verdict.get("comment") or f"judge: {', '.join(failed)}"})
            reasons["judge: " + ", ".join(failed)] += 1
            n_rejected += 1
            print(f"rp {transcript['id']}: REJECT {verdict.get('comment', '')}")

    print(f"\nDone this run: {n_clean} accepted, {n_rejected} rejected")
    if reasons:
        print("Rejection reasons:")
        for reason, count in reasons.most_common():
            print(f"  {count:>5}  {reason}")
    print(f"\nRead {reject_path}. A reason that repeats is a step 03 prompt problem, "
          f"not a reason to throw data away.")


if __name__ == "__main__":
    main()
