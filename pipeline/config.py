# -*- coding: utf-8 -*-
"""Configuration loading and command line parsing.

Every numbered script starts the same way:

    args = parse_args("what this step does")
    cfg = load_config(args.config)

`load_config` merges the user's YAML on top of DEFAULTS, so a config file only
has to mention the values it actually changes. Missing sections are not an
error; missing *required* values (the input path, the LLM address) are, and are
reported with a message that says what to fix.
"""
import argparse
import copy
import os
import sys

import yaml

# Defaults for every tunable in the pipeline. The user's config.yaml is merged
# on top of this, key by key, so an old config keeps working when a new option
# is added here.
DEFAULTS = {
    # Language of THIS run. Selects prompts/<lang>.yaml. One run = one language.
    "lang": "en",

    "input": {
        # Plain UTF-8 text file: the novel, the log archive, whatever the source is.
        "path": "input/source.txt",
        # Tried in order until one decodes cleanly. utf-8 first, always.
        "encodings": ["utf-8-sig", "utf-8", "cp1251", "cp1252", "koi8-r", "latin-1"],
    },

    # All intermediate JSONL files of this run land here. One workdir per language.
    "workdir": "work/en",

    "llm": {
        # Any OpenAI-compatible server: llama-server, koboldcpp, vLLM, TabbyAPI,
        # or a hosted API. Only /v1/chat/completions is used.
        "base_url": "http://127.0.0.1:5001",
        "model": "local",
        "api_key": "none",
        "temperature": 0.3,
        "max_tokens": 4096,
        "retries": 3,
        "timeout": 600,
        # 1 for a local single-slot server. 6-8 for a hosted API with real
        # concurrency. More than the server can take only produces timeouts.
        "workers": 1,
    },

    "segment": {
        # A structural block longer than this is handed to the LLM for scene
        # boundary detection. Shorter blocks are kept as they are, for free.
        "max_block_chars": 14000,
        # Scenes shorter than this are glued to their neighbour.
        "min_scene_chars": 700,
        # Scenes longer than this are split on paragraph borders.
        "max_scene_chars": 11000,
        # Sliding window budget for the LLM pass, in characters.
        "llm_window_chars": 9000,
        # Window overlap, in paragraphs, so a boundary at a window edge is not lost.
        "llm_window_overlap": 6,
    },

    "attribute": {
        "temperature": 0.0,
    },

    "convert": {
        # Who becomes {{char}}: "auto" = the most talkative speaker of the scene,
        # or an exact character name to keep only scenes where that name speaks.
        "char": "auto",
        # Transcripts with fewer turns than this are dropped.
        "min_turns": 2,
        # Scenes with less direct speech than this are not worth converting.
        "min_dialogue_share": 0.03,
        "temperature": 0.4,
    },

    "validate": {
        # Share of the transcript's content words that also occur in the source
        # scene. Low value = the converter wrote its own prose instead of
        # carrying the source over.
        "lexical_overlap_min": 0.55,
        # The LLM judge pass. Turn it off to keep only the cheap lexical filter.
        "llm_check": True,
        "temperature": 0.1,
    },

    "synth": {
        # Synthetic sessions "in the spirit of the source" that are not in the
        # source. Without them the adapter quotes instead of generalising.
        "enabled": True,
        "count": 200,
        "turns_min": 3,
        "turns_max": 6,
        # Real scenes shown to the model per generation, as thinking anchors.
        "anchors_per_gen": 2,
        "anchor_max_chars": 2500,
        # Above this overlap with the anchors the model is rewriting the source.
        "max_anchor_overlap": 0.5,
        # 4-gram Jaccard against already accepted rows.
        "dedupe_jaccard": 0.35,
        "temperature": 0.9,
        # Explicit character rotation. Empty = the most talkative characters found
        # in step 02.
        "chars": [],
        "chars_limit": 8,
    },

    "format": {
        # Leave empty to take the cards from the language pack (prompts/<lang>.yaml).
        # {char} is substituted with the character name.
        "system_card": "",
        # Same card with all format instructions removed. Step 07 needs it: its
        # <format> block would otherwise contradict the card.
        "system_card_core": "",
        "output": "dataset_base.jsonl",
    },

    "augment": {
        # Rewrite each transcript several times under different format
        # directives, so the model learns to read the system block.
        "enabled": True,
        "variants_per_scene": 3,
        # Also keep the plain, directive-free version of every transcript.
        "keep_original": True,
        # Share of variants that get no behavioural directive attached.
        "behavior_drop_rate": 0.33,
        # Same meaning and same value as validate.lexical_overlap_min.
        "min_overlap": 0.55,
        # Retention of the source's distinctive words. Catches softening.
        "min_distinctive": 0.70,
        # Protection against a variant that silently truncates the scene.
        "min_len_ratio": 0.35,
        "temperature": 0.35,
        "max_axes_per_variant": 3,
        # Optional behavioural directives harvested by tools/extract_axes.py.
        "axes_file": "axes.yaml",
        # Axis groups from axes_file that are never trained on, so that
        # generalisation to unseen instructions can be measured.
        "holdout_axes": ["html"],
        "output": "dataset_aug.jsonl",
        "seed": 42,
        # 0 = every transcript. N = a random sample of N, for a trial run.
        "limit": 0,
    },

    "assemble": {
        # "auto" picks the augmented dataset if step 07 produced one, otherwise
        # the plain one from step 06. Or name a file explicitly.
        "source_input": "auto",
        "diluent": {
            # Third-party roleplay data mixed in so the adapter stays usable
            # outside the source's rails. Off by default: the pipeline works
            # without it, and the files are yours to choose.
            "enabled": False,
            # Local ShareGPT .jsonl or .parquet files.
            "files": [],
            # Total number of sessions to sample across all files.
            "count": 800,
            # Translate the diluent into the run language with the config LLM.
            "translate": False,
            # Drop sessions that trip the content filter (see pipeline/safety.py).
            "safety_filter": True,
        },
        # Session budget in characters, trimmed on turn borders.
        "max_session_chars": 14000,
        "seed": 42,
        "output": "dataset_final.jsonl",
    },

    "publish": {
        # "user/dataset-name" on the Hugging Face Hub.
        "repo_id": "",
        "private": True,
        # Share of sessions held out as an eval split. 0 = no split.
        "eval_share": 0.05,
        # Written next to the dataset and uploaded as the dataset card.
        "card_output": "DATASET_CARD.md",
        "license": "other",
        "seed": 42,
    },
}


def deep_merge(base, override):
    """Recursive dict merge. `override` wins on scalars, dicts are merged."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def fail(message):
    """User-facing error: print the fix, stop. Not for internal invariants."""
    sys.exit(f"\n[config] {message}\n")


def load_config(path="config.yaml"):
    """Read the YAML config, merge it over DEFAULTS, sanity check the result."""
    if not os.path.exists(path):
        fail(f"config file not found: {path}\n"
             f"Copy config.yaml from the project root and edit it, or pass --config PATH.")
    with open(path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    if not isinstance(user_cfg, dict):
        fail(f"{path} must contain a YAML mapping at the top level.")

    cfg = deep_merge(DEFAULTS, user_cfg)

    # Guards on values a wrong setting would silently corrupt the run with.
    assert isinstance(cfg["lang"], str) and cfg["lang"], "lang must be a non-empty string"
    assert cfg["llm"]["workers"] >= 1, "llm.workers must be at least 1"
    assert cfg["segment"]["min_scene_chars"] < cfg["segment"]["max_scene_chars"], \
        "segment.min_scene_chars must be smaller than segment.max_scene_chars"
    assert 0.0 <= cfg["validate"]["lexical_overlap_min"] <= 1.0, \
        "validate.lexical_overlap_min is a share, it must be between 0 and 1"
    assert 0.0 <= cfg["augment"]["min_distinctive"] <= 1.0, \
        "augment.min_distinctive is a share, it must be between 0 and 1"

    cfg["_config_path"] = path
    return cfg


def parse_args(description, extra=None):
    """Standard CLI for a pipeline step.

    `extra` is an optional callable that receives the parser and can add
    step-specific arguments before parsing.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="config.yaml",
                        help="path to the config file (default: config.yaml)")
    if extra is not None:
        extra(parser)
    return parser.parse_args()


def work_path(cfg, name):
    """Path of an intermediate file inside this run's workdir."""
    return os.path.join(cfg["workdir"], name)


def out_path(cfg, section, key):
    """Path of a step's final output file, relative to the working directory."""
    return cfg[section][key]
