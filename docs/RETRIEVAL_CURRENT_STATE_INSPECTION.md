# Retrieval Current-State Inspection

This note records the retrieval implementation and artifacts inspected before
the HarMeme-train-only remediation. It is intentionally descriptive: it does
not treat the legacy corpus as paper eligible.

## Active legacy artifacts

- Configured corpus: `dataset/source/wiki_common/cache/corpus_texts.jsonl`
- Source snapshot: `dataset/source/wiki_common/wiki_corpus.jsonl`
- Manifest: `dataset/source/wiki_common/wiki_manifest.json`
- Dense artifact: `dataset/source/wiki_common/cache/corpus_emb.npy`
- Per-query caches:
  - `dataset/source/covid_img+text/rag/cache_all/covid_rag_all_top5.jsonl`
  - `dataset/source/political_img+text/rag/cache_all/political_rag_all_top5.jsonl`
  - equivalent Facebook and Memotion files, which are forbidden for the paper corpus

## Current schemas

The full legacy source row is:

```json
{
  "kid": "pageid:revid:sentence_index",
  "pageid": 11600,
  "revid": 1335656402,
  "rev_timestamp": "2026-01-30T12:36:09Z",
  "source": "wikipedia",
  "text": "...",
  "title": "...",
  "url": "https://en.wikipedia.org/wiki/..."
}
```

The configured `corpus_texts.jsonl` projection contains only `kid` and
`text`. It has 64,818 rows.

Each local per-query cache row contains:

```json
{
  "id": "source sample ID",
  "labels": 0,
  "query": "combined query text",
  "split": "legacy split label",
  "topk": [{"kid": "...", "score": 0.0, "text": "..."}]
}
```

The label and legacy split fields are not needed for replay and must not be
read by the canonical builder. Dataset identity is represented by the cache
file location and its `meta.json`, rather than by a field in each query row.

The legacy manifest records aggregate builder settings and these dataset
provenance labels: `facebook`, `covid`, `political`, and `memotion`. It also
records counts, a legacy absolute dataset root, Wikipedia page/sentence
limits, a user agent, and a sleep interval. It does not bind construction to
the immutable HarMeme split or identify source samples at corpus-row level.

## How the legacy rows were created

No corpus-builder or cache-builder source implementation is present in the
current repository. The artifacts show that a Wikipedia sentence corpus was
selected using terms from all four datasets and that per-sample queries were
then ranked against a cached dense representation. The manifest's user-agent,
page limits, revision metadata, and sleep interval show that the original
source collection required network access. The exact sparse/dense builder and
embedding model cannot be recovered from source code currently in this tree.

At framework runtime, `module/backbone/retrieval.py` does not consume the
legacy `corpus_emb.npy`. It loads configured text rows, computes BM25-like and
lexical scores, computes deterministic 256-dimensional hashed vectors in
memory, and combines sparse/dense ranks with reciprocal-rank fusion.

## Provenance and replay decision

The aggregate corpus is not safely filterable by dataset name and will not be
used as the unit of provenance. Corpus rows contain no originating sample IDs.
The per-query caches do contain real source sample IDs and the exact top-k
document IDs independently retrieved for each query.

The immutable source manifest and local caches align exactly:

- 5,611/5,611 HarMeme train sample keys have one query-cache row.
- 1,402/1,402 HarMeme validation sample keys are separately identifiable.
- No source cache IDs are unknown to the immutable train/validation manifest.
- All cached top-k `kid` values resolve to the local full Wikipedia snapshot.
- HarMeme-train rows contain 28,055 retrieval occurrences and 4,791 unique
  external documents.

Therefore a deterministic offline replay is possible by selecting query rows
with exact sample keys from the immutable `train` partition, joining their
`kid` values to the full local Wikipedia snapshot, and rebuilding all indexes.
This does not filter the aggregate corpus by dataset. Facebook, Memotion, and
HarMeme-validation query caches are never inputs to the canonical replay.

The canonical query ID will be a documented deterministic identifier derived
from the real dataset name, real sample ID, and exact cached query hash. It is
not a fabricated sample ID. Only legitimate HarMeme-train origins will survive
document deduplication.

## Pre-change verification

The mandated pre-change command was run in `meme_cikm`:

```bash
conda run -n meme_cikm python -m pytest tests -q
```

It stopped during collection with five import errors caused by a pre-existing
Python 3.10 syntax error at `experiments/paper_export.py:311` (a backslash in a
nested f-string expression). No retrieval tests ran in that baseline attempt.
