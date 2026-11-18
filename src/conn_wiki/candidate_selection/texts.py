"""Text preparation for the judge prompt: one snippet per mention.

Ported from the uploaded decide_common.py (squash/snippet/snippets); the only
change is that the four constants (MIN_SENTENCE_CHARS, MAX_CHARS,
MAX_SNIPPETS) are now config.Config fields, so experiments can tune them from
the CLI without touching code.  The marking logic is unchanged:

  1. mentions WITHOUT a sentence are ignored;
  2. a sentence shorter than min_sentence_chars is replaced by the longer
     context;
  3. the mention is marked [[like this]] at the occurrence closest to its
     document offset;
  4. every whitespace run becomes one space, and snippets longer than
     max_chars are cut around the [[...]] mark.
"""
import re


def squash(text: str) -> str:
    """Every whitespace run (newlines included) becomes one space."""
    return re.sub(r"\s+", " ", text or "").strip()


def snippet(m: dict, cfg) -> str | None:
    """Text of one mention with [[mention]] marked, or None if the mention has no sentence."""
    sentence = squash(m.get("sentence"))
    if not sentence:
        return None                                              # no sentence: ignore the mention
    text, base = sentence, m.get("sentence_start")
    context = squash(m.get("context"))
    if len(sentence) < cfg.min_sentence_chars and len(context) > len(sentence):
        text, base = context, m.get("context_start")             # short sentence: use the context

    name = squash(m.get("text")).lower()                         # mark the occurrence closest to the offset
    hits = [i for i in range(len(text)) if name and text.lower().startswith(name, i)]
    if hits:
        expected = m["start"] - base if base is not None and m.get("start") is not None else 0
        i = min(hits, key=lambda p: abs(p - expected))
        text = f"{text[:i]}[[{text[i:i + len(name)]}]]{text[i + len(name):]}"
    text = squash(text)                                          # no newline or double space after marking

    if len(text) > cfg.max_chars:                                # cut around the mention
        a = max(0, max(text.find("[["), 0) - cfg.max_chars // 2)
        text = text[a:a + cfg.max_chars]
    return text


def snippets(cluster: dict, cfg) -> list[str]:
    """Snippets of the mentions of one cluster, document order, deduplicated, capped."""
    out = []
    for m in sorted(cluster["mentions"], key=lambda m: m.get("start") or 0):
        s = snippet(m, cfg)
        if s and s not in out:
            out.append(s)
    return out[:cfg.max_snippets]
