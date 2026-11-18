# Fixed-version Flat, Generic, and Domain comparison

This command compares three ways to select source chunks for the same fixed detector. Build both GraphRAG 3.3 indexes from the exact same corpus and install the [3.3 environment](../graph/README.md) first.

| Arm | Ranking used by the detector |
|---|---|
| Flat | Cosine similarity between the Generic query embedding and the native source-chunk embeddings shared by both builds |
| Generic | Source-unit order observed inside native local search with the Generic extraction prompt |
| Domain | Source-unit order observed inside native local search with the Domain extraction prompt |

Every arm sends at most four complete chunks and 6,000 Unicode characters. The detector sees verbatim source chunks labeled with local citation IDs. It uses the same protected prompt, caller-supplied strategy, Qwen Plus settings, and strict four-field output contract in all arms. The graph query runs at community level zero with a 6,000-token native context budget and produces no generative answer. The comparison writes the native context and the source-chunk selection separately so the recorded result shows what the detector actually saw.

## Run

```bash
python -m comparison.paired --generic runs/generic --domain runs/domain --cases cases.json --strategy comparison/strategy.txt --out runs/paired
```

Use a fresh output directory. It retains the comparison inputs, observable provider I/O, per-case retrieval and detection, and `report.json` on completion. The command stops on a changed graph, different source corpus or source embeddings, incompatible native settings, a failed request, or an invalid detector response. One case uses exactly two embedding calls and three detection calls. Up to twelve cases are accepted per invocation.

[`strategy.txt`](strategy.txt) contains the exact fixed comparison strategy, shared unchanged across arms. The software test uses the invented instruction `Use the provided source records.` The public command builds fresh graphs from caller-supplied inputs. Historical execution caches remain private.

`cases.json` is an array. Each case needs a unique `sample_id` and a `sentence` for the indexed patient. Optionally attach a full-source reference grade and exact source quotations. The detector never receives the reference.

```json
[
  {
    "sample_id": "SYNTHETIC-Q1",
    "sentence": "The red token is assigned to the first drawer.",
    "reference": {
      "source_grade": "E1",
      "source_quotes": [{"evidence_id": "Note-A", "quote": "red token"}]
    }
  }
]
```

## Interpret the output

The default build ceilings of 8,192 and 32,768 differ. Treat the default pair as a comparison of two graph configurations, with extraction prompt and output ceiling both varying. For a comparison at the same output ceiling, pass the same `--completion-ceiling` to both build commands. The report retains the actual ceilings and does not assert a causal effect.

When the caller supplies references, the report counts agreement with the full-source reference grades and reports how many reference quotes occur in the retrieved chunks with the correct source ID. This can help locate ranking misses. The evidence that supports a grade in one retrieved subset would require its own review. Clinical accuracy and independent clinical validation are outside the scope of this report.

The original comparison logic lived in the paper supplement’s `paired_graph.py`. The public implementation retains its per-arm retrieval, budget, source binding, detector contract, and quote-coverage method in an independently runnable command. Clinical source records, historic build caches, labels, and recorded model responses are absent from the public code.
