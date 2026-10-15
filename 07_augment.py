# -*- coding: utf-8 -*-
"""Step 07 — vary the format so the model learns to read the system block.

The problem. Step 06 puts one system card on every session; only the character
name changes. The model never observes the instruction changing, so it learns
that the system block carries no information and stops attending to it. On
inference the user's own preset is then ignored, which is the single most
common complaint about style adapters.

The fix. Rewrite each transcript several times under different formatting
directives, and write those directives into the system block. The content stays
identical across variants, so the only thing that distinguishes them is the
instruction — which makes reading the instruction the only way to satisfy the
row.

Two modes.

  --rewrite   The default, for your own data. The model receives three inputs
              at once: the source scene as the STYLE reference, the finished
              transcript as the CONTENT, and the directive list as the TARGET
              FORMAT. Anchoring the style to the source scene rather than to the
              transcript is what keeps this from being a copy of a copy.

  --diluent   For third-party data. Regex only, no model: quotes, emphasis,
              paragraph breaks. There is no authorial voice to preserve in a
              diluent, so paying for an LLM rewrite would buy nothing.

Three guards protect the rewritten variants, checked in this order:

  1. refusal          the model declined instead of rewriting
  2. retention        distinctive words of the original survived. Checked first
                      of the two lexical guards because softening lowers both,
                      and reporting "low overlap" for what is actually softening
                      sends you looking in the wrong place
  3. overlap          the vocabulary still comes from the source, not from the
                      rewriter. Skipped for additive axes, whose whole purpose
                      is to add text that was not there

    Input:  <workdir>/rp_clean.jsonl, optionally <workdir>/synth_clean.jsonl,
            <workdir>/scenes.jsonl
    Output: augment.output (default dataset_aug.jsonl)

    python 07_augment.py                       # rewrite, the normal path
    python 07_augment.py --diluent SRC DST     # regex pass over a diluent file
    python 07_augment.py --selftest            # check the regex transforms
"""
import json
import os
import random
import sys
from collections import Counter

import yaml

from pipeline.axes import (ADDITIVE, CHANGED, NA, TIER1, axis_pool, clean_macros,
                           directive, format_block, pick_combo, stable_rng, tier1_pool)
from pipeline.config import load_config, parse_args, work_path
from pipeline.files import append_jsonl, read_jsonl, require_file, write_jsonl
from pipeline.lang import load_pack, render
from pipeline.llm import LLM, extract_json, make_refusal_checker, parallel_map
from pipeline.sharegpt import normalize_entry, to_sharegpt
from pipeline.textstats import build_df, overlap, rarity_cutoff, retention


def render_turns(turns):
    """The transcript as the model sees it."""
    return "\n\n".join(f"[{t['role']}] {t['text']}" for t in turns)


def build_system_card(base_card, directives, behavior_pool, behavior_on, char, user, rng):
    """Assemble the system block: card, then <format>, then behaviour directives."""
    blocks = [base_card.strip()]
    block = format_block(directives)
    if block:
        blocks.append(block)
    if behavior_on and behavior_pool:
        for item in rng.sample(behavior_pool, min(2, len(behavior_pool))):
            text = clean_macros(item.get("text", ""), char, user)
            if text:
                blocks.append(text)
    return "\n\n".join(b for b in blocks if b)


def check_variant(new_turns, old_turns, scene_text, combo, df, cutoff, settings, stem_length,
                  is_refusal):
    """Return (accepted, reason, retention_score)."""
    new_text = "\n".join(t["text"] for t in new_turns if t["role"] == "char")
    old_text = "\n".join(t["text"] for t in old_turns if t["role"] == "char")
    if not new_text.strip():
        return False, "empty", 0.0
    if is_refusal(new_text):
        return False, "refusal", 0.0

    score = retention(old_text, new_text, df, cutoff, stem_length)
    if score < settings["min_distinctive"]:
        return False, f"softened {score:.2f}", score

    if not any((axis, value) in ADDITIVE for axis, value in combo.items()):
        overlap_score = overlap(new_text, scene_text)
        if overlap_score < settings["min_overlap"]:
            return False, f"overlap {overlap_score:.2f}", score

    if combo.get("length") != "short":
        ratio = len(new_text) / max(len(old_text), 1)
        if ratio < settings["min_len_ratio"]:
            return False, f"truncated {ratio:.2f}", score

    return True, "", score


def load_behavior_pool(settings):
    """Optional behavioural directives harvested by tools/extract_axes.py.

    These are not format axes. They are instructions about how a character
    behaves — do not act for the user, avoid clichés — and they vary the system
    block in a second, independent dimension. Entirely optional: without the
    file the step runs on format axes alone.
    """
    path = settings["axes_file"]
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for held_out in settings["holdout_axes"]:
        data.pop(held_out, None)
    pool = []
    for key in ("autonomy", "prose"):
        pool += data.get(key) or []
    return pool


def load_rows(cfg, settings):
    """Transcripts plus synthetic rows, optionally sampled down for a trial."""
    rows = read_jsonl(require_file(work_path(cfg, "rp_clean.jsonl"), "04_validate.py"))
    synth_path = work_path(cfg, "synth_clean.jsonl")
    if os.path.exists(synth_path):
        synth = read_jsonl(synth_path)
        base = max((row["id"] for row in rows), default=-1) + 1
        for offset, row in enumerate(synth):
            row["id"] = base + offset
            row.setdefault("source_id", None)  # synthetic rows have no source scene
        rows += synth
        print(f"[synthetic] +{len(synth)} rows")

    if settings["limit"]:
        # Synthetic rows are appended at the end, so trimming the tail would
        # sample only converted scenes. They behave differently — no source
        # anchor, the row itself is the style reference — so a trial that misses
        # them tells you nothing about half the data. Shuffle, then trim.
        random.Random(settings["seed"]).shuffle(rows)
        rows = rows[:settings["limit"]]
        from_source = sum(1 for row in rows if row.get("source_id") is not None)
        print(f"[trial] {len(rows)} rows: {from_source} converted, "
              f"{len(rows) - from_source} synthetic")
    return rows


# ---------------------------------------------------------------------------
# --rewrite
# ---------------------------------------------------------------------------

def rewrite(cfg):
    settings = cfg["augment"]
    pack = load_pack(cfg["lang"])
    is_refusal = make_refusal_checker(pack.refusal_markers)
    openers = pack.require("openers")
    fallback_user = pack.fallback_user

    scenes = {s["id"]: s for s in
              read_jsonl(require_file(work_path(cfg, "scenes.jsonl"), "01_segment.py"))}
    rows = load_rows(cfg, settings)
    assert rows, "nothing to augment — check the output of step 04"

    df = build_df(scene["text"] for scene in scenes.values())
    cutoff = rarity_cutoff(len(scenes))
    stem_length = pack.stem_length
    print(f"Scenes: {len(scenes)}   transcripts: {len(rows)}   "
          f"rarity cutoff: df <= {cutoff}   stem length: {stem_length}")

    behavior_pool = load_behavior_pool(settings)
    print(f"Behavioural directives: {len(behavior_pool)}"
          + ("   (none — optional, see tools/extract_axes.py)" if not behavior_pool else ""))

    pool = axis_pool(pack)
    assert pool, f"the {cfg['lang']} language pack declares no usable format axes"
    print("Format axes: " + ", ".join(f"{k}({len(v)})" for k, v in sorted(pool.items())))

    # The core card, with its own formatting sentences removed. A full card says
    # things like "third person" and "show the line of thought"; those would
    # contradict the <format> block, and two conflicting instructions about
    # format teach the model to ignore the system block — the exact failure this
    # step is meant to cure.
    card_template = cfg["format"]["system_card_core"] or pack.get("cards.core")
    if not card_template:
        card_template = cfg["format"]["system_card"] or pack.require("cards.full")
        print("\n[!] No core card found, falling back to the full one. If it contains "
              "instructions about person, tense or showing thoughts, they will argue "
              "with the <format> block. Add cards.core to the language pack.\n")

    llm = LLM(cfg)
    workers = cfg["llm"]["workers"]
    used, stats, rejected = Counter(), Counter(), Counter()
    accepted, retentions = [], []

    # Build the job list first, so axis usage is balanced across the whole run
    # rather than drifting as it goes.
    jobs = []
    for row in rows:
        rng = stable_rng(settings["seed"], row["id"])
        scene = scenes.get(row.get("source_id"))
        # Synthetic rows have no source scene; they are already in the source's
        # voice, so they serve as their own style reference.
        scene_text = scene["text"] if scene else render_turns(row["turns"])
        for _ in range(settings["variants_per_scene"]):
            combo = pick_combo(pool, rng, settings["max_axes_per_variant"], used)
            for axis, value in combo.items():
                used[f"{axis}={value}"] += 1
            jobs.append((row, scene_text, combo))

    def work(job):
        row, scene_text, combo = job
        directives = [directive(pack, axis, value) for axis, value in sorted(combo.items())]
        raw = llm.chat(
            pack.require("augment.system"),
            render(pack.require("augment.user"),
                   scene=scene_text[:9000],
                   transcript=render_turns(row["turns"]),
                   directives="\n".join(f"- {d}" for d in directives)),
            temperature=settings["temperature"], json_mode=True)
        data = extract_json(raw)
        raw_turns = data if isinstance(data, list) else (data.get("turns", [])
                                                         if isinstance(data, dict) else [])
        turns = [t for t in raw_turns
                 if isinstance(t, dict) and t.get("role") in ("char", "user")
                 and str(t.get("text", "")).strip()]
        return turns, combo, directives

    print(f"\nJobs: {len(jobs)}   workers: {workers}\n")
    reject_path = work_path(cfg, "aug_rejected.jsonl")
    processed = 0
    for job, result, error in parallel_map(work, jobs, workers):
        processed += 1
        row, scene_text, _ = job
        if error:
            rejected["llm error"] += 1
        else:
            turns, combo, directives = result
            ok, reason, score = check_variant(turns, row["turns"], scene_text, combo,
                                              df, cutoff, settings, stem_length, is_refusal)
            if not ok:
                rejected[reason.split(" ")[0]] += 1
                append_jsonl(reject_path, {"id": row["id"], "combo": combo, "reason": reason})
            else:
                rng = stable_rng(settings["seed"], row["id"], json.dumps(combo, sort_keys=True))
                behavior_on = rng.random() > settings["behavior_drop_rate"]
                user_name = row.get("user") or fallback_user
                card = build_system_card(render(card_template, char=row["char"]),
                                         directives, behavior_pool, behavior_on,
                                         row["char"], user_name, rng)
                opener = render(openers[row["id"] % len(openers)],
                                char=row["char"], user=user_name)
                session = to_sharegpt(turns, card, opener=opener)
                if session:
                    accepted.append(session)
                    retentions.append(score)
                    for axis, value in combo.items():
                        stats[f"{axis}={value}"] += 1
                    stats["_behavior_on" if behavior_on else "_behavior_off"] += 1
        if processed % 25 == 0 or processed == len(jobs):
            print(f"  {processed}/{len(jobs)}   accepted {len(accepted)}", flush=True)

    if settings["keep_original"]:
        card_full = cfg["format"]["system_card"] or pack.require("cards.full")
        for row in rows:
            user_name = row.get("user") or fallback_user
            opener = render(openers[row["id"] % len(openers)],
                            char=row["char"], user=user_name)
            session = to_sharegpt(row["turns"], render(card_full, char=row["char"]),
                                  opener=opener)
            if session:
                accepted.append(session)
                stats["original"] += 1

    random.Random(settings["seed"]).shuffle(accepted)
    write_jsonl(settings["output"], accepted)

    # -- report ------------------------------------------------------------
    on = stats.pop("_behavior_on", 0)
    off = stats.pop("_behavior_off", 0)
    ratio = len(accepted) / max(len(rows), 1)
    print(f"\nDone: {len(rows)} transcripts -> {len(accepted)} sessions "
          f"(x{ratio:.2f}) -> {settings['output']}")
    if on + off:
        print(f"  behavioural directive omitted in {off}/{on + off} "
              f"({off / (on + off):.0%}, target {settings['behavior_drop_rate']:.0%})")
    if retentions:
        retentions.sort()
        print(f"  distinctive-word retention among accepted: min {retentions[0]:.2f}, "
              f"median {retentions[len(retentions) // 2]:.2f} "
              f"(threshold {settings['min_distinctive']:.2f})")
        print("  if the minimum sits right on the threshold, raise it and re-read the "
              "rejects; if it is far above, the threshold can be tightened")

    print("\nAxis coverage:")
    for axis, values in sorted(pool.items()):
        for value in values:
            count = stats.get(f"{axis}={value}", 0)
            print(f"  {axis}={value:<14} {count:>6}" + ("   <- never used" if not count else ""))
    if stats.get("original"):
        print(f"  {'original':<22} {stats['original']:>6}")
    if rejected:
        print(f"\nRejected (details in {reject_path}):")
        for reason, count in sorted(rejected.items(), key=lambda kv: -kv[1]):
            print(f"  {reason:<22} {count:>6}")

    print("\n  * Keep the training step count as it was: this is the same data in "
          "different wrappers, not more data.")
    print(f"  * Inspect it: python tools/inspect_dataset.py --sample 10 {settings['output']}")


# ---------------------------------------------------------------------------
# --diluent
# ---------------------------------------------------------------------------

def diluent(cfg, source, destination):
    """Regex-only format variation over third-party ShareGPT data."""
    settings = cfg["augment"]
    pack = load_pack(cfg["lang"])
    raw_rows = read_jsonl(source)
    assert raw_rows, f"{source} is empty"
    # Normalise first: third-party files use several role conventions, and a
    # file read raw would silently lose every session that uses another one.
    rows = [{"conversations": conv} for conv in
            (normalize_entry(entry) for entry in raw_rows) if conv]
    if len(rows) < len(raw_rows):
        print(f"{len(raw_rows) - len(rows)} of {len(raw_rows)} sessions are unusable "
              f"(no assistant turn) and were dropped")

    pool = tier1_pool(pack)
    assert pool, "no axes available for regex transforms in this language pack"
    print(f"Diluent: {len(rows)} sessions, axes: "
          + ", ".join(f"{k}({len(v)})" for k, v in sorted(pool.items())))

    out, stats, used = [], Counter(), Counter()
    for index, row in enumerate(rows):
        conversation = row.get("conversations") or []
        if not conversation:
            continue
        out.append({"conversations": conversation})
        stats["original"] += 1

        rng = stable_rng(settings["seed"], "diluent", index)
        gpt_indices = [i for i, m in enumerate(conversation) if m["from"] == "gpt"]

        for _ in range(settings["variants_per_scene"]):
            combo = pick_combo(pool, rng, settings["max_axes_per_variant"], used)
            texts, applied = {}, {}
            for axis, value in combo.items():
                transform, _supported = TIER1[axis]
                staged, statuses, blocked = {}, [], False
                for i in gpt_indices:
                    new_text, status = transform(texts.get(i, conversation[i]["value"]), value)
                    if status == NA:
                        blocked = True
                        break
                    staged[i] = new_text
                    statuses.append(status)
                # An axis counts only if it actually changed something somewhere
                # and contradicted nothing. Otherwise the system block would
                # promise a format the answers do not follow.
                if blocked or CHANGED not in statuses:
                    continue
                texts.update(staged)
                applied[axis] = value
                used[f"{axis}={value}"] += 1

            if not applied:
                continue
            directives = [directive(pack, axis, value) for axis, value in sorted(applied.items())]
            block = format_block(directives)
            new_conversation = []
            for i, message in enumerate(conversation):
                if message["from"] == "system":
                    new_conversation.append({"from": "system",
                                             "value": message["value"].rstrip() + "\n\n" + block})
                elif i in texts:
                    new_conversation.append({"from": "gpt", "value": texts[i]})
                else:
                    new_conversation.append(dict(message))
            if not any(m["from"] == "system" for m in new_conversation):
                new_conversation.insert(0, {"from": "system", "value": block})
            out.append({"conversations": new_conversation})
            for axis, value in applied.items():
                stats[f"{axis}={value}"] += 1

    random.Random(settings["seed"]).shuffle(out)
    write_jsonl(destination, out)
    print(f"\nDone: {len(rows)} -> {len(out)} (x{len(out) / max(len(rows), 1):.2f}) "
          f"-> {destination}")
    for key, count in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"  {key:<22} {count:>6}")


# ---------------------------------------------------------------------------
# --selftest
# ---------------------------------------------------------------------------

SELFTEST_CASES = [
    ("dash dialogue, two paragraphs",
     "Viktor turned. His fingers shook.\n— You are late, — he said quietly. *Let it be.* "
     "The lamps went out.\n\nViktor moved to the table. His fingers shook.\n"
     "— That is not why I came, — he said quietly. *He waited.* It smelled of damp."),
    ("guillemets, two lines",
     "He froze. «You are late», — he said. A pause. «And you know why», — he added."),
    ("straight quotes", 'He froze. "You are late," he said. The candle guttered.'),
    ("no direct speech",
     "Viktor stood by the window. Rain drummed on the tin. Nothing happened."),
    ("both emphasis kinds",
     "He came in. *The door creaked.* _Her again._ The room was dark."),
]


def selftest():
    """Exercise the tier-1 regex transforms on the cases that have broken before.

    History, because it explains the shape of the code: the dash transform broke
    twice. First a greedy match swallowed the attribution tag along with the
    prose after it. Then a two-pass substitution paired the closing marker of one
    line with the opening marker of the next and ate everything between them.
    Both are covered here.
    """
    failures = 0
    for name, source in SELFTEST_CASES:
        print("=" * 72)
        print(name)
        for axis, (transform, values) in TIER1.items():
            for value in values:
                result, status = transform(source, value)
                broken = (
                    "\x00" in result
                    or "\n— ," in result
                    or "— , —" in result
                    or (status == CHANGED and result.strip() == source.strip())
                    or (status == "ok" and result.strip() != source.strip())
                )
                failures += bool(broken)
                marker = "BROKEN" if broken else "ok    "
                print(f"  {axis}={value:<11} {status:<8} {marker}  {result[:60]!r}")
    print("=" * 72)
    print(f"FAILED: {failures}" if failures else "ALL CHECKS PASSED")
    return 1 if failures else 0


def main():
    def extra(parser):
        parser.add_argument("--rewrite", action="store_true",
                            help="LLM rewrite of your own data (the default)")
        parser.add_argument("--diluent", nargs=2, metavar=("SRC", "DST"),
                            help="regex-only pass over a third-party ShareGPT file")
        parser.add_argument("--selftest", action="store_true",
                            help="check the regex transforms, no config or model needed")

    args = parse_args(__doc__.splitlines()[0], extra)

    if args.selftest:
        sys.exit(selftest())
    cfg = load_config(args.config)
    if args.diluent:
        diluent(cfg, *args.diluent)
        return
    if not cfg["augment"]["enabled"]:
        print("augment.enabled is false — skipping. Step 08 will use the step 06 output.")
        return
    rewrite(cfg)


if __name__ == "__main__":
    main()
