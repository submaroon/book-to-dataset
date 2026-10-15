# -*- coding: utf-8 -*-
"""Language packs.

Every prompt, every directive and every canned phrase lives in
`prompts/<lang>.yaml`, not in the scripts. Adding a language means copying
`prompts/en.yaml`, translating the strings, and setting `lang:` in the config.
No Python is touched.

Placeholders use double square brackets — `[[scene]]`, `[[char]]` — instead of
Python's `{}` formatting. The prompts are full of literal JSON braces, and with
`str.format` every one of them would have to be escaped by whoever translates
the pack. Square brackets need no escaping, and `render` asserts that nothing
was left unsubstituted, so a typo in a placeholder name fails loudly instead of
sending `[[chr]]` to the model.
"""
import os
import re

import yaml

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")

_PLACEHOLDER = re.compile(r"\[\[([a-z_]+)\]\]")

# Fallback chapter words, used when a pack does not declare its own. Covers the
# languages the shipped packs cover plus a few near neighbours; a pack for any
# other language should list its own under meta.chapter_words.
DEFAULT_CHAPTER_WORDS = (
    "chapter", "part", "prologue", "epilogue", "interlude", "book", "section",
    "глава", "часть", "пролог", "эпилог", "интерлюдия", "книга", "раздел",
)


def chapter_regex(pack=None, full_line=False):
    """Regex matching a chapter heading line.

    Built from the language pack rather than hard-coded, so adding a language
    does not mean editing the segmentation code. Bare numbers and Roman
    numerals alone on a line are recognised for every language.
    """
    words = list(pack.get("meta.chapter_words", []) if pack else []) or list(DEFAULT_CHAPTER_WORDS)
    alternation = "|".join(re.escape(w) for w in words)
    tail = r"\b.*$" if full_line else r"\b"
    return re.compile(
        rf"^\s*(?:{alternation}){tail}"
        r"|^\s*[IVXLC]{1,7}\s*$"      # a Roman numeral alone on its line
        r"|^\s*\d{1,3}\s*$",          # a bare number alone on its line
        re.I | re.UNICODE,
    )


class LanguagePack:
    """Read-only view over prompts/<lang>.yaml with helpful failures."""

    def __init__(self, lang, data, path):
        self.lang = lang
        self.path = path
        self._data = data

    def get(self, dotted, default=None):
        """Fetch `section.key` from the pack, returning `default` if absent."""
        node = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted):
        """Fetch `section.key`, or stop with a message naming the missing key."""
        value = self.get(dotted, None)
        assert value is not None, (
            f"language pack {self.path} has no '{dotted}'. "
            f"Copy that key from prompts/en.yaml and translate it.")
        return value

    # -- convenience accessors used by more than one step ------------------

    @property
    def unknown_speaker(self):
        """The word the model is told to use when it cannot identify a speaker."""
        return self.require("meta.unknown_speaker")

    @property
    def generic_speakers(self):
        """Words that are never a character name: pronouns, roles, "the voice".

        Attribution returns these regularly, and treating one as a character
        produces transcripts where {{char}} is literally "he".
        """
        return set(w.lower() for w in self.get("meta.generic_speakers", []))

    @property
    def stem_length(self):
        """Prefix length used for crude stemming in the quality gates.

        Inflected languages need this: a legitimate tense change turns "knocked"
        into "knocks" and exact word matching would call that a rewrite. 5 works
        for Russian; set it higher for languages with long roots, or very high
        to effectively disable stemming.
        """
        return int(self.get("meta.stem_length", 5))

    @property
    def refusal_markers(self):
        return list(self.get("meta.refusal_markers", []))

    @property
    def fallback_user(self):
        return self.require("meta.fallback_user")

    @property
    def fallback_char(self):
        return self.require("meta.fallback_char")


# Keys whose value must be a list of plain strings.
_STRING_LISTS = (
    "synth.situations",
    "synth.behavior_mods",
    "openers",
    "diluent_openers",
    "meta.generic_speakers",
    "meta.chapter_words",
    "meta.refusal_markers",
)


def _validate(pack):
    """Catch pack authoring mistakes at load time rather than mid-run.

    The one that actually happens: an unquoted YAML list item containing a
    colon and a space is parsed as a mapping, not a string. The pipeline then
    runs for an hour and fails on whichever row happens to draw that entry.
    Checking here turns that into an immediate, named error.
    """
    for dotted in _STRING_LISTS:
        items = pack.get(dotted)
        if items is None:
            continue
        assert isinstance(items, list), \
            f"{pack.path}: '{dotted}' must be a list, got {type(items).__name__}"
        for index, item in enumerate(items):
            assert isinstance(item, str), (
                f"{pack.path}: '{dotted}' item {index} is a {type(item).__name__}, "
                f"not a string: {item!r}\n"
                f"A YAML list item containing ': ' must be quoted.")

    for axis, values in (pack.get("axes", {}) or {}).items():
        assert isinstance(values, dict), \
            f"{pack.path}: 'axes.{axis}' must be a mapping of value name to directive"
        for name, text in values.items():
            assert isinstance(text, str), (
                f"{pack.path}: 'axes.{axis}.{name}' is a {type(text).__name__}, "
                f"not a string. Quote it if it contains ': '.")

    for dotted in ("cards.full", "cards.core"):
        card = pack.get(dotted)
        if card is not None:
            assert isinstance(card, str), f"{pack.path}: '{dotted}' must be a string"
            assert "[[char]]" in card, \
                f"{pack.path}: '{dotted}' has no [[char]] placeholder — every session " \
                f"would get the same card with no character name in it."
    return pack


def load_pack(lang):
    """Load and validate prompts/<lang>.yaml."""
    path = os.path.join(PROMPTS_DIR, f"{lang}.yaml")
    if not os.path.exists(path):
        available = sorted(f[:-5] for f in os.listdir(PROMPTS_DIR) if f.endswith(".yaml"))
        raise SystemExit(
            f"\nNo language pack for lang: {lang}\n"
            f"Available: {', '.join(available)}\n"
            f"To add one: cp prompts/en.yaml prompts/{lang}.yaml and translate the strings.\n")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _validate(LanguagePack(lang, data, path))


def render(template, **values):
    """Substitute [[placeholders]] in a prompt template.

    Every placeholder present in the template must be supplied, and every
    supplied value must be used — both directions are asserted, because a silent
    mismatch here means the model receives an incomplete prompt and the failure
    only shows up as bad data much later.
    """
    wanted = set(_PLACEHOLDER.findall(template))
    given = set(values)
    missing = wanted - given
    assert not missing, f"prompt needs values for: {', '.join(sorted(missing))}"

    result = template
    for key in wanted:
        result = result.replace(f"[[{key}]]", str(values[key]))

    leftover = _PLACEHOLDER.findall(result)
    assert not leftover, f"unsubstituted placeholders remain: {leftover}"
    return result
