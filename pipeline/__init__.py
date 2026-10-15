# -*- coding: utf-8 -*-
"""Shared library for the roleplay dataset pipeline.

The numbered scripts in the project root are thin drivers. Everything that is
used by more than one step lives here, so that a fix lands in exactly one place.

Modules:
    config      config file loading, CLI argument parsing, workdir paths
    files       text reading with encoding detection, JSONL read/write/append
    llm         OpenAI-compatible chat client, retries, refusal detection
    lang        language pack loading (prompts/<lang>.yaml) and placeholder rendering
    textstats   language-neutral text statistics used by the quality gates
    axes        format axes: directives, regex transforms, combination picking
    sharegpt    conversion of internal turn lists into the ShareGPT format
    safety      content filters applied to third-party data before it is mixed in
"""

__version__ = "1.0.0"
