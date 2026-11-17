# Entity resolution for the 1939 radio bulletins

Two-level mention clustering with per-type strategies, token-aware string
similarity, and paragraph-level context. Package: `entity_resolution/`.

## Run

```bash
# full mode (RECOMMENDED for wikidata linking): the bulletin text comes from the
# IE records' own `text` field.  Run fix_ie_offsets.py first to replace that
# field with the preprocessed bulletin text (paragraphs separated by \n\n) and
# correct all entity/relation offsets — no separate text file is needed.
python fix_ie_offsets.py --ie ner_re.jsonl --processed bulletins_processed.jsonl \
        --output ner_re_fixed.jsonl
python entity_resolution/run_clustering.py --ie ner_re_fixed.jsonl --output data/interim/

# optional override: take the text from an external file instead (the mention
# offsets must have been computed against exactly that variant)
python entity_resolution/run_clustering.py --ie extracted.jsonl \
        --ocr bulletins_processed.jsonl --output data/interim/

# simple mode (only when the records carry no text at all: context falls back
# to the IE output's `context` field)
python entity_resolution/run_clustering.py --ie extracted.jsonl --output data/interim/

# resolve Organization like Person as well
python entity_resolution/run_clustering.py --ie ... \
        --string-similarity-types Person,Organization,Place,LegalDocument,Vehicle
```

Outputs (in `--output`): `global_clusters.json` (corpus clusters — the input the
wikidata linker reads), `document_clusters.jsonl` (per bulletin; the `clusters`
field is the wikidata-facing one, name-less OCR-garbage clusters are tracked
separately in `empty_name_clusters`), and `clustering_report.json` (diagnostics).

## Strategies

| Type | Strategy |
| --- | --- |
| Person, Place, LegalDocument, Vehicle | two-level string-similarity clustering |
| Event | exact `event_id` matching **within** a bulletin only (never cross-corpus) |
| Organization (default) | document-local singletons; add it to `--string-similarity-types` to resolve it like Person |
| anything else | document-local singletons |

Stage 1 clusters mentions within one bulletin (fuzzy + containment attachment,
cannot-link on numbers/Roman numerals and person nationality). Stage 2 merges
document clusters across the corpus with constrained single linkage at
`--corpus-threshold` (default 0.9). IE `subtype` fields are never used.

## Asymmetric strictness between the two levels

- **Stage 1 (within one bulletin): relaxed** — plain character-level similarity
  (`fuzz.ratio`), the original behaviour. Inside one bulletin two different
  entities with similar names are rare, and recall matters more here. Caveat:
  if two names sharing a generic tail DO occur in the same bulletin, this
  stage-1 merge cannot be undone at stage 2 — pass `--stage1-token-guard` if
  you ever observe that.
- **Stage 2 (across the corpus): always strict** — the token-aware score
  vetoes merges that would only happen through a shared generic tail.

## Token-aware similarity (why "spaansche" and "fransche" stay apart)

Plain character similarity overrates multi-word names that share a generic tail:

```
fuzz.ratio("fransche communistische partij", "spaansche communistische partij")
   = 0.918  -> merged at threshold 0.9            (WRONG)
name_similarity(..., ...)  = 0.806 -> kept apart  (RIGHT)
name_similarity("fransche communistische partij", "franse ...") = 0.95 -> merges
name_similarity("chamber lain", "chamberlain") = 1.0 -> OCR word split merges
```

The merge score is `min(fuzz.ratio, worst per-token Jaro-Winkler)`: every token of
the shorter name must find a similar partner token in the longer name
(`cluster.name_similarity`, cached and exported). Vetoes are logged in
`clustering_report.json` under `token_guard_blocks` / `token_guard_examples`.

## Paragraph handling

Paragraphs are `\n\n`-separated blocks. Paragraphs with at most
`short_para_max_words` words (default 4; `--short-para-words`, 0 disables) —
datelines and kickers such as a standalone `"Soesterberg."` — are merged into
the paragraph they introduce; a few-word paragraph at the very end merges into
the previous one. Consequences:

- a mention's `context` is never a bare title, so the wikidata linker always
  sees real evidence text;
- `paragraph_id` does not inflate on title lines;
- sentence segmentation, the repaired `sentence`, and the person-cue windows
  all work on the merged paragraphs (the existing sentence repair already
  glues a short dateline sentence to the following sentence).

## Context fields (what the wikidata linker gets per mention)

| field | full mode (bulletin text from the IE `text` field or `--ocr`) | simple mode |
| --- | --- | --- |
| `text` / `raw_text` | cleaned slice / raw slice of the bulletin text | IE text |
| `sentence` | repaired sentence, clamped to the paragraph | — |
| `context` | **the whole paragraph containing the mention** (`\n\n`-separated) | the IE output's `context` field (often empty) |
| `paragraph_id` | index of that paragraph in the bulletin | — |
| `cues_local` | person cues from transient local windows | person cues from the IE context field |

`context_before` / `context_after` are no longer stored on mentions — in simple
mode with an empty IE context field they were always empty; the paragraph
supersedes them as linker evidence (appositions, titles, nationalities stay
within the paragraph). Person-cue extraction still uses the local windows
internally; cues never cross a `\n\n` boundary.

## Tests

```bash
python entity_resolution/test_entity_resolution.py   # 77 checks
python entity_resolution/example_usage.py            # end-to-end demo (writes example_output/)
```
