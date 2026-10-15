# -*- coding: utf-8 -*-
"""Step 08 — assemble the final dataset.

Takes the dataset built from your own source and, optionally, mixes in
third-party roleplay data as a diluent.

About the diluent. Training on one source alone produces an adapter that is
excellent on that source's rails and brittle off them: unfamiliar settings,
unfamiliar formats and unfamiliar user styles degrade sharply. A share of
general roleplay data prevents that. It is off by default, because the files are
yours to choose and the pipeline must work without them — turn it on in the
config when you have sources you trust.

What this step does:
  1. Picks the source-derived part: the augmented dataset from step 07 if it
     exists, otherwise the plain one from step 06. This resolution is logged, so
     it is always visible which file was actually used.
  2. Reads the diluent files, normalises their role names, applies the content
     filter, samples the configured number of sessions and trims them to the
     session budget on turn borders.
  3. Optionally translates the diluent into the run's language, with resume and
     refusal detection.
  4. Shuffles everything and reports the composition.

    Output: assemble.output (default dataset_final.jsonl)

    python 08_assemble.py [--config config.yaml]
"""
import os
import random
import sys

from pipeline.config import load_config, parse_args, work_path
from pipeline.files import append_jsonl, read_dataset, read_jsonl, write_jsonl
from pipeline.axes import clean_macros
from pipeline.lang import load_pack
from pipeline.llm import LLM, make_refusal_checker, parallel_map
from pipeline.safety import filter_sessions
from pipeline.sharegpt import normalize_entry, trim_session


def resolve_source_input(cfg):
    """Which file holds the part of the dataset built from your own text.

    "auto" prefers the augmented output of step 07 and falls back to the plain
    output of step 06. An explicit filename in the config overrides both, which
    is what you want when a file lives elsewhere or was assembled by hand.
    """
    configured = cfg["assemble"]["source_input"]
    if configured and configured != "auto":
        if not os.path.exists(configured):
            sys.exit(f"\nassemble.source_input points at {configured}, which does not exist.\n")
        return configured, "assemble.source_input"

    augmented = cfg["augment"]["output"]
    if os.path.exists(augmented):
        return augmented, "step 07 (augmented)"
    base = cfg["format"]["output"]
    if os.path.exists(base):
        return base, "step 06 (not augmented)"

    sys.exit(
        f"\nNeither {augmented} nor {base} exists.\n"
        f"Run 06_format.py, and optionally 07_augment.py, before this step.\n"
        f"If your file is somewhere else, set assemble.source_input in the config.\n")


def translate_sessions(sessions, cfg, pack):
    """Translate diluent sessions into the run's language, with a resume cache.

    Every message is a separate call. That is slower than sending whole sessions
    but far more reliable: a long session translated in one call regularly comes
    back with turns merged or dropped, which silently corrupts the structure.
    """
    llm = LLM(cfg)
    workers = cfg["llm"]["workers"]
    is_refusal = make_refusal_checker(pack.refusal_markers)
    system = pack.require("translate.system")

    cache_path = work_path(cfg, "diluent_translated.jsonl")
    refusal_path = work_path(cfg, "refusals_08.jsonl")
    cache = {item["index"]: item["conversations"] for item in read_jsonl(cache_path)}
    pending = [(i, s) for i, s in enumerate(sessions) if i not in cache]
    if cache:
        print(f"[resume] {len(cache)} sessions already translated, {len(pending)} left")

    def work(item):
        index, conversation = item
        out = []
        for message in conversation:
            # Short system strings are scene openers and markers, not prose.
            if message["from"] == "system" and len(message["value"]) < 40:
                out.append(message)
                continue
            translated = llm.chat(system, message["value"], temperature=0.2)
            if is_refusal(translated):
                raise RuntimeError("refusal")
            out.append({"from": message["from"], "value": translated.strip()})
        return out

    for item, result, error in parallel_map(work, pending, workers):
        index, _ = item
        if error is not None:
            if "refusal" in str(error):
                append_jsonl(refusal_path, {"index": index})
                print(f"session {index}: translation refused — rerun with another model")
            else:
                print(f"  [warn] session {index}: {error} — will retry", file=sys.stderr)
            continue
        cache[index] = result
        append_jsonl(cache_path, {"index": index, "conversations": result})
        print(f"session {index}: translated ({len(result)} messages)")

    return [cache[i] for i in range(len(sessions)) if i in cache]


def collect_diluent(cfg, pack, rng):
    """Read, filter, sample and trim the diluent. Returns a list of sessions."""
    settings = cfg["assemble"]["diluent"]
    if not settings["enabled"]:
        print("=== diluent: disabled (assemble.diluent.enabled = false) ===")
        return []
    files = settings.get("files") or []
    if not files:
        print("=== diluent: enabled but assemble.diluent.files is empty — nothing to mix in ===")
        return []

    print("=== diluent ===")
    pools = []
    for path in files:
        if not os.path.exists(path):
            print(f"  [warn] {path}: not found, skipped", file=sys.stderr)
            continue
        raw = read_dataset(path)
        usable = [entry for entry in (normalize_entry(e) for e in raw) if entry]
        print(f"  {path}: {len(usable)} usable sessions out of {len(raw)}")
        if usable:
            pools.append(usable)
    if not pools:
        return []

    # Sample evenly across files, so one large source does not swamp the rest.
    want = settings["count"]
    per_file = max(1, want // len(pools))
    sampled = []
    for pool in pools:
        sampled += rng.sample(pool, min(per_file, len(pool)))

    if settings["safety_filter"]:
        kept, dropped = filter_sessions(
            sampled,
            extra_minor=settings.get("extra_minor_markers", []),
            extra_sexual=settings.get("extra_sexual_markers", []))
        if dropped:
            print(f"  content filter dropped {len(dropped)} of {len(sampled)} sessions")
            for _session, reason in dropped[:3]:
                print(f"    example: {reason}")
            if len(dropped) > len(sampled) // 5:
                print("  WARNING: more than 20% dropped. This filter is a backstop, not a "
                      "cleaning tool — replace the source rather than relying on it.")
        sampled = kept

    # Third-party data is full of SillyTavern macros: {{char}}, {{user}},
    # <USER>, conditional blocks. If they reach the training data literally, the
    # model learns to emit curly braces in its answers. The diluent has no known
    # character names, so the language pack's fallbacks are substituted.
    char_name, user_name = pack.fallback_char, pack.fallback_user
    cleaned, emptied = [], 0
    for conversation in sampled:
        session = []
        for message in conversation:
            value = clean_macros(message["value"], char_name, user_name)
            if value:
                session.append({"from": message["from"], "value": value})
        if any(m["from"] == "gpt" for m in session):
            cleaned.append(session)
        else:
            emptied += 1
    if emptied:
        print(f"  {emptied} sessions dropped: nothing left after macro cleanup")

    openers = pack.require("diluent_openers")
    budget = cfg["assemble"]["max_session_chars"]
    trimmed = []
    for conversation in cleaned:
        session = trim_session(conversation, budget, rng.choice(openers))
        if session:
            trimmed.append(session)
    print(f"  after filtering and trimming: {len(trimmed)} sessions")
    return trimmed


def main():
    args = parse_args(__doc__.splitlines()[0])
    cfg = load_config(args.config)
    pack = load_pack(cfg["lang"])
    settings = cfg["assemble"]
    rng = random.Random(settings["seed"])

    source_path, origin = resolve_source_input(cfg)
    print(f"=== own data: {source_path}  ({origin}) ===")
    own = read_jsonl(source_path)
    assert own, f"{source_path} is empty"
    print(f"  sessions loaded: {len(own)}")

    diluent = collect_diluent(cfg, pack, rng)

    if settings["diluent"]["translate"] and diluent:
        print("=== translating the diluent ===")
        diluent = translate_sessions(diluent, cfg, pack)
        print(f"  ready: {len(diluent)}")

    final = own + [{"conversations": conversation} for conversation in diluent]
    rng.shuffle(final)
    output = settings["output"]
    write_jsonl(output, final)

    total = len(final)
    own_share = 100 * len(own) // max(1, total)
    diluent_share = 100 * len(diluent) // max(1, total)
    print(f"\n=== RESULT: {total} sessions -> {output} ===")
    print(f"  from your source: {len(own)} ({own_share}%)")
    print(f"  diluent:          {len(diluent)} ({diluent_share}%)")
    if settings["diluent"]["enabled"] and 0 < diluent_share < 20:
        print("\nThe diluent is under 20% of the dataset. That is usually too little to "
              "keep the adapter usable outside your source — raise "
              "assemble.diluent.count or add more files.")
    print(f"\nNext: python tools/inspect_dataset.py --stats {output}")
    print(f"      python 09_publish.py")


if __name__ == "__main__":
    main()
