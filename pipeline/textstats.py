# -*- coding: utf-8 -*-
"""Text statistics used by the quality gates.

These are deliberately cheap and language-neutral. They do not need a tokenizer,
a model, or a word list, which means they run on the full corpus for free and
work the same on any alphabet.

Three measures, three different failure modes:

  overlap    How much of the output's vocabulary comes from the source. Low
             overlap means the model wrote its own prose instead of carrying
             the source over.
  retention  How many of the source's *rare* words survived. Catches softening:
             a retelling can keep a high overlap while replacing every concrete
             word with a vague one, because vague words are common words.
  jaccard    Similarity between two generated items. Catches a generator that
             produces the same scene over and over.
"""
import re
from collections import Counter

# Any run of 4+ letters, in any script. \W excludes punctuation, and the
# explicit \d and _ exclusions keep numbers and identifiers out.
WORD_RE = re.compile(r"[^\W\d_]{4,}", re.UNICODE)


def words(text):
    """Lowercased content words of a text."""
    return [w.lower() for w in WORD_RE.findall(text or "")]


def overlap(candidate, reference):
    """Share of the candidate's content words that also occur in the reference.

    Returns 0.0 for an empty candidate, which correctly reads as "rejected".
    """
    candidate_words = set(words(candidate))
    if not candidate_words:
        return 0.0
    return len(candidate_words & set(words(reference))) / len(candidate_words)


def ngrams(text, n=4):
    """Set of word n-grams, for near-duplicate detection."""
    tokens = words(text)
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(left, right):
    """Similarity of two n-gram sets, 0.0 to 1.0."""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def build_df(texts):
    """Document frequency of every word across the corpus.

    "Document" here is one scene. This is what makes rarity measurable without
    any external word list: a word is distinctive for *this* text if it occurs
    in few of its scenes.
    """
    df = Counter()
    for text in texts:
        df.update(set(words(text)))
    return df


def rarity_cutoff(n_documents, share=0.02, floor=2):
    """Document frequency below which a word counts as distinctive.

    The median df on a novel collapses to 1 — more than half the vocabulary
    occurs in exactly one scene — so a median-based threshold would call almost
    everything rare and the retention ratio would be computed over a handful of
    words. "Occurs in at most 2% of scenes" keeps the set statistically usable.
    """
    return max(floor, round(share * n_documents)) if n_documents else floor


def stems(word_list, length):
    """Crude prefix stemming, so inflection does not look like word replacement."""
    return set(w[:length] if len(w) > length else w for w in word_list)


def retention(source, result, df, cutoff, stem_length, min_rare=8):
    """Share of the source's distinctive words still present in the result.

    Falls back to the full vocabulary when the fragment has too few rare words
    to make a stable ratio — otherwise losing 2 words out of 6 would read as a
    30% drop and reject a perfectly good rewrite.
    """
    source_words = words(source)
    rare = [w for w in source_words if df.get(w, 0) <= cutoff]
    if len(rare) < min_rare:
        rare = source_words
    if not rare:
        return 1.0
    before = stems(rare, stem_length)
    after = stems(words(result), stem_length)
    return len(before & after) / len(before)


def percentile(sorted_values, share):
    """Value at a given share of a pre-sorted list. Empty list returns 0."""
    if not sorted_values:
        return 0
    index = min(len(sorted_values) - 1, int(len(sorted_values) * share))
    return sorted_values[index]
