# Graph construction and retrieval

CuraView has two GraphRAG workflows. Run them in separate Python environments.

| GraphRAG version | Workflow | Index format |
|---|---|---|
| 2.7.0 | Build patient-specific indexes for the [local detector](../detection/) | Native 2.7 Parquet and LanceDB |
| 3.3.0 | Build Generic and Domain semantic indexes, retrieve native source order, and run the [fixed-version comparison](../comparison/) | Native 3.3 Parquet and LanceDB |

## Input

Supply one authorized corpus per patient. The top-level `patient_id` is a caller-controlled identifier. Each document needs a unique `id`, a `title`, and the verbatim `text` you want indexed. Omit protected or target-only material from the input. Both construction paths use the following JSON shape.

```json
{
  "patient_id": "SYNTHETIC-P1",
  "documents": [
    {"id": "Note-A", "title": "Invented note", "text": "The red token is assigned to the first drawer."}
  ]
}
```

The [offline tools](tools/README.md) can also convert caller-supplied EHR JSONL or CSV files into this format. The fixtures used by the software checks are invented.

The model provider receives source content during indexing and receives queries or selected text during retrieval and comparison. Use an authorized endpoint for the data you supply. Keep corpora, graphs, caches, and request/response records in local private run directories.

## GraphRAG 2.7 for the detector

Install with Python 3.12 in a dedicated environment.

```bash
python -m pip install -r graph/requirements-v27.txt
python -m graph.prepare_v27 --corpus corpus.json --project runs/graph-v27
graphrag index --root runs/graph-v27/output/index_output/SYNTHETIC-P1
graphrag query --root runs/graph-v27/output/index_output/SYNTHETIC-P1 --method local --query "Where is the red token?"
```

Set the `GRAPHRAG_API_KEY` environment variable before calling GraphRAG. The prepared YAML uses the DashScope OpenAI-compatible endpoint, Qwen Plus, and `text-embedding-v3`. The detector's `graphrag.project_root` points to the project directory, here `runs/graph-v27`. Its patient index is located beneath `output/index_output/<patient_id>`. The CLI also accepts the native 2.7 global, basic, and DRIFT query modes.

The 2.7 settings retain the original 1,200-token chunks and 100-token overlap. The input filename, title mapping, and prompt paths were made portable. The public prompt copies omit source-derived example blocks from extraction, graph and text community reporting, and local and basic search. These copies preserve the task instructions and output templates. Consequently they are public adaptations of the older templates.

## GraphRAG 3.3 for the fixed comparison

Install with Python 3.12 in a second environment.

```bash
python -m pip install -r graph/requirements-v33.txt
python -m graph.native build --corpus corpus.json --arm generic --out runs/generic
python -m graph.native build --corpus corpus.json --arm domain --out runs/domain
python -m graph.native query --graph runs/generic --out runs/query --text "Where is the red token?"
```

Set `GRAPHRAG_API_KEY` before running. The default `CURAVIEW_API_BASE` is `https://dashscope.aliyuncs.com/compatible-mode/v1`. An override must support the fixed Qwen request shape seen by the tests and both models. Requests use Qwen Plus with temperature zero and thinking disabled, and `text-embedding-v3` with 1,024 floating-point dimensions. The native build uses 1,200-token chunks, a 100-token overlap, a maximum community cluster size of 50, the shared 15 entity types, zero extra extraction passes, and shared description and community prompts. `--completion-ceiling` allows 4,096, 8,192 or 32,768 tokens. The defaults implement the paper supplement’s Generic and Domain settings at 8,192 and 32,768 respectively. Index provider requests use a 600-second timeout.

The public Domain build loads the [English Domain extraction prompt](v33/prompts/domain.extract.txt) directly. Its UTF-8 SHA-256 is `67902699f8c2efd37da2eda0f3483cae62abda109319c5b7e14c93847da0730b`. Generic uses the resolved Microsoft GraphRAG template. The arms share the description and community templates. See the [table guide](../supplement/Table_Guide.md) for the scope of the reported experiments.

The build writes the corpus, prompt copies, resolved file paths, native tables, LanceDB vector store, workflow outcomes and a SHA-256 manifest to a fresh graph directory. Keep the directory at its build path. Its settings refer to that path. Query verifies the manifest before loading the graph. `context.json` includes the observed native source-unit order, query embedding, vector candidates, source mappings, and selected native context. The command calls only the embedding endpoint for query retrieval. Each provider attempt writes its observable request and response into the local run. Authentication headers and server internals are outside these records.

## Verification

```bash
python -m unittest graph.test_tools -v
python -m unittest graph.test_native -v
```

The test uses GraphRAG 3.3.0 and a loopback provider fixture. It exercises two native builds, context-only retrieval, all three comparison arms and rejection of changed graph inputs. It checks execution and provenance handling on invented documents. Provider model quality and the paper’s clinical results require separate data and evidence.

Run `graph.test_tools` in the 2.7 environment to check native input loading and chunking without provider calls.
