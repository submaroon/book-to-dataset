# -*- coding: utf-8 -*-
"""The LLM client and everything around a model call.

The pipeline only ever uses one endpoint, /v1/chat/completions, so it runs
against llama-server, koboldcpp, vLLM, TabbyAPI, or any hosted OpenAI-compatible
API without changes.

Three things here are less obvious than the HTTP call itself:

  * `extract_json` — models wrap JSON in markdown fences, preambles and
    reasoning blocks. Parsing has to survive that.
  * `looks_like_refusal` — a refusal is a valid HTTP 200 with a polite
    paragraph in it. Steps treat refusals as "not done yet" rather than as
    results, so a rerun with a different model picks them up.
  * `parallel_map` — one worker means a plain loop, which keeps local
    single-slot servers happy and makes tracebacks readable.
"""
import json
import re
import sys
import time

import requests

# Refusal markers that are model-language independent enough to always check.
# Language packs add their own under `refusal_markers`.
BASE_REFUSAL_MARKERS = (
    "i cannot", "i can't", "i'm unable", "i am unable", "cannot assist",
    "can't assist", "cannot help with", "can't help with", "i won't be able",
    "i will not be able", "i'm not able to", "i am not able to", "cannot fulfill",
    "can't fulfill", "cannot comply", "against my guidelines", "content policy",
    "as an ai", "as a language model",
)


class LLM:
    """Thin OpenAI-compatible chat client with retries."""

    def __init__(self, cfg):
        llm_cfg = cfg["llm"]
        self.base_url = llm_cfg["base_url"].rstrip("/")
        self.model = llm_cfg.get("model", "local")
        self.api_key = llm_cfg.get("api_key", "none")
        self.temperature = llm_cfg.get("temperature", 0.3)
        self.max_tokens = llm_cfg.get("max_tokens", 4096)
        self.retries = llm_cfg.get("retries", 3)
        self.timeout = llm_cfg.get("timeout", 600)

    def chat(self, system, user, temperature=None, json_mode=False, max_tokens=None):
        """One call. Returns the message content as a string.

        Raises after `retries` failures, which the calling step treats as "retry
        this item on the next run" rather than as a reason to stop.
        """
        payload = {
            "model": self.model,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            # Servers that do not know this field ignore it; servers that do
            # honour it save a whole class of parsing failures.
            payload["response_format"] = {"type": "json_object"}

        last_error = None
        for attempt in range(self.retries):
            try:
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except Exception as exc:  # noqa: BLE001 - any failure is retryable here
                last_error = exc
                wait = 2 ** attempt
                print(f"  [retry {attempt + 1}/{self.retries}] {exc}; sleeping {wait}s",
                      file=sys.stderr)
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed after {self.retries} retries: {last_error}")

    def ping(self):
        """Check the server answers at all. Used by the smoke test."""
        reply = self.chat("You are a test fixture.", "Reply with the single word OK.",
                          temperature=0.0, max_tokens=16)
        return reply.strip()


def make_refusal_checker(extra_markers=()):
    """Build a refusal detector for a given language pack."""
    markers = tuple(BASE_REFUSAL_MARKERS) + tuple(m.lower() for m in extra_markers)

    def looks_like_refusal(text):
        # Only the head is checked: a refusal announces itself immediately,
        # while a legitimate answer may mention these phrases in passing later.
        head = (text or "").strip().lower()[:500]
        return any(marker in head for marker in markers)

    return looks_like_refusal


_JSON_RE = re.compile(r"\{.*\}|\[.*\]", re.S)
_THINK_CLOSED = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.S | re.I)
_THINK_OPEN = re.compile(r"^<think(?:ing)?>", re.I)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)


def extract_json(text):
    """Pull a JSON value out of a model reply.

    Handles, in order: closed reasoning blocks, a reasoning block that got cut
    off by max_tokens, markdown fences, and finally a preamble before the JSON.
    """
    text = (text or "").strip()
    text = _THINK_CLOSED.sub("", text).strip()
    if _THINK_OPEN.match(text) and "</think" not in text.lower():
        # Unterminated reasoning block: the answer, if any, is after the last
        # opening brace.
        brace = text.rfind("{")
        text = text[brace:] if brace != -1 else text
    text = _FENCE.sub("", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_RE.search(text)
        if match:
            return json.loads(match.group(0))
        raise


def parallel_map(fn, items, workers):
    """Run fn over items, yielding (item, result, error) as results arrive.

    workers == 1 runs a plain sequential loop. That is not just an optimisation:
    a local single-slot server queues concurrent requests anyway, and sequential
    execution keeps the console log in a readable order.
    """
    items = list(items)
    if workers <= 1:
        for item in items:
            try:
                yield item, fn(item), None
            except Exception as exc:  # noqa: BLE001 - reported per item, never fatal
                yield item, None, exc
        return

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                yield item, future.result(), None
            except Exception as exc:  # noqa: BLE001
                yield item, None, exc
