"""Build and query one caller-supplied corpus using fixed GraphRAG 3.3 settings."""
import argparse
import asyncio
import hashlib
from importlib import metadata
import json
from pathlib import Path
from unittest.mock import patch

from .io import load_corpus, read_json, write_json
from .provider import CHAT_MODEL, EMBEDDING_MODEL, DIMENSIONS, PROVIDER, Transport, register_transport

VERSION = "3.3.0"
BUILD_TIMEOUT = 600
ARMS = {"generic": 8192, "domain": 32768}
PROMPTS = Path(__file__).parent / "v33" / "prompts"
WORKFLOWS = [
    "create_base_text_units", "create_final_documents", "extract_graph",
    "finalize_graph", "create_communities", "create_final_text_units",
    "create_community_reports", "generate_text_embeddings",
]
ENTITY_TYPES = [
    "PATIENT", "CLINICAL_EVENT", "EXAMINATION", "IMAGING_FINDING", "ANATOMICAL_SITE",
    "DISEASE", "SYMPTOM", "MEASUREMENT", "CLINICAL_IMPRESSION", "TIMEPOINT",
    "DEMOGRAPHIC_ATTRIBUTE", "LAB_TEST", "LAB_RESULT", "TREATMENT", "PROCEDURE",
]
EMBEDDINGS = ["entity_description", "community_full_content", "text_unit_text"]
TABLES = ["documents", "text_units", "entities", "relationships", "communities",
          "community_reports", "raw_entities", "raw_relationships",
          *(f"embeddings.{name}" for name in EMBEDDINGS)]


def settings(root, arm, max_tokens):
    model = {"type": PROVIDER, "model_provider": "openai", "retry": None, "metrics": None}
    file_store = lambda name: {"type": "file", "base_dir": str(root / name)}
    return {
        "completion_models": {"default_completion_model": {
            **model, "model": CHAT_MODEL, "call_args": {"max_tokens": max_tokens, "temperature": 0},
        }},
        "embedding_models": {"default_embedding_model": {**model, "model": EMBEDDING_MODEL}},
        "concurrent_requests": 1,
        "async_mode": "asyncio",
        "input_storage": file_store("input"),
        "output_storage": file_store("output"),
        "update_output_storage": file_store("update_output"),
        "cache": {"type": "json", "storage": file_store("cache")},
        "reporting": file_store("logs"),
        "vector_store": {"type": "lancedb", "db_uri": str(root / "vectors"), "index_schema": {
            name: {"index_name": name, "vector_size": DIMENSIONS} for name in EMBEDDINGS
        }},
        "workflows": WORKFLOWS,
        "chunking": {"size": 1200, "overlap": 100, "encoding_model": "cl100k_base"},
        "cluster_graph": {"max_cluster_size": 50},
        "extract_graph": {"max_gleanings": 0, "entity_types": ENTITY_TYPES,
                          "prompt": str(root / "prompts" / f"{arm}.extract.txt")},
        "summarize_descriptions": {"prompt": str(root / "prompts" / "describe.txt")},
        "community_reports": {"graph_prompt": str(root / "prompts" / "community.graph.txt"),
                              "text_prompt": str(root / "prompts" / "community.text.txt")},
        "snapshots": {"raw_graph": True, "embeddings": True},
        "embed_text": {"batch_size": 10, "batch_max_tokens": 8000, "names": EMBEDDINGS},
        "local_search": {"max_context_tokens": 6000, "top_k_entities": 10, "top_k_relationships": 10},
    }


def require_runtime():
    for name in ("graphrag", "graphrag-llm", "graphrag-cache", "graphrag-vectors"):
        if metadata.version(name) != VERSION:
            raise ValueError(f"use the isolated GraphRAG {VERSION} environment")


def manifest(root):
    files = [root / "corpus.json", root / "settings.json", root / "build.json"]
    for dirname in ("prompts", "output", "vectors"):
        files.extend(path for path in (root / dirname).rglob("*") if path.is_file())
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)}


def load_graph(root):
    import pandas as pd
    root = Path(root).resolve()
    require_runtime()
    if json.loads(read_json(root / "manifest.json")) != manifest(root):
        raise ValueError("graph files differ from the build manifest")
    build = json.loads(read_json(root / "build.json"))
    if build["version"] != VERSION or build["arm"] not in ARMS:
        raise ValueError("incompatible graph build")
    config = settings(root, build["arm"], build["completion_ceiling"])
    if json.loads(read_json(root / "settings.json")) != config:
        raise ValueError("graph settings differ from fixed configuration")
    names = (f"{build['arm']}.extract.txt", "describe.txt", "community.graph.txt", "community.text.txt")
    for name in names:
        if (root / "prompts" / name).read_bytes() != (PROMPTS / name).read_bytes():
            raise ValueError(f"graph prompt differs from the fixed template: {name}")
    tables = {name: pd.read_parquet(root / "output" / f"{name}.parquet") for name in TABLES}
    corpus = load_corpus(root / "corpus.json")
    sources = {doc["id"]: doc["text"] for doc in corpus["documents"]}
    units = tables["text_units"]
    vectors = tables["embeddings.text_unit_text"]
    if (any(table.empty for table in tables.values())
            or dict(zip(tables["documents"]["id"], tables["documents"]["text"])) != sources
            or set(units["document_id"]) != set(sources) or not units["id"].is_unique
            or not vectors["id"].is_unique or set(vectors["id"]) != set(units["id"])
            or any(row.text not in sources[row.document_id] for row in units.itertuples())):
        raise ValueError("native graph lacks a complete and consistent source mapping")
    return build, corpus, tables


async def build_graph(corpus_path, root, arm, max_tokens=None):
    require_runtime()
    import pandas as pd
    from graphrag.api import build_index
    from graphrag.config.models.graph_rag_config import GraphRagConfig

    corpus = load_corpus(corpus_path)
    max_tokens = ARMS[arm] if max_tokens is None else max_tokens
    if type(max_tokens) is not int or max_tokens not in (4096, 8192, 32768):
        raise ValueError("completion ceiling must be 4096, 8192 or 32768")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    config = settings(root, arm, max_tokens)
    (root / "input").mkdir()
    (root / "prompts").mkdir()
    for name in (f"{arm}.extract.txt", "describe.txt", "community.graph.txt", "community.text.txt"):
        (root / "prompts" / name).write_bytes((PROMPTS / name).read_bytes())
    write_json(root / "settings.json", config)
    write_json(root / "corpus.json", corpus)
    transport = Transport(root / "requests", timeout=BUILD_TIMEOUT)
    register_transport(transport)
    results = await build_index(GraphRagConfig(**config), input_documents=pd.DataFrame(corpus["documents"]))
    workflow = [{"name": row.workflow, "error_type": type(row.error).__name__ if row.error else None}
                for row in results]
    write_json(root / "workflows.json", workflow)
    if transport.failed or [row.workflow for row in results] != WORKFLOWS or any(
        row.error is not None for row in results
    ):
        raise RuntimeError("native indexing failed. Inspect workflows.json and requests/ in the run directory.")
    write_json(root / "build.json", {"version": VERSION, "arm": arm, "completion_ceiling": max_tokens,
                                   "provider_base_url": transport.endpoint, "provider_calls": transport.calls,
                                   "request_timeout_seconds": BUILD_TIMEOUT})
    write_json(root / "manifest.json", manifest(root))
    load_graph(root)


def native_context(root, build, tables, query, transport, max_tokens=6000):
    from graphrag.config.models.graph_rag_config import GraphRagConfig
    from graphrag.query.factory import get_local_search_engine
    from graphrag.query.indexer_adapters import (
        read_indexer_entities, read_indexer_relationships, read_indexer_reports, read_indexer_text_units,
    )
    from graphrag.query.structured_search.local_search import mixed_context
    from graphrag.utils.api import get_embedding_store

    if not query.strip() or type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("nonempty query and positive context token ceiling are required")
    register_transport(transport)
    config = GraphRagConfig(**settings(Path(root).resolve(), build["arm"], build["completion_ceiling"]))
    vectors = get_embedding_store(config.vector_store, "entity_description")
    query_vectors, vector_candidates, ranking = [], [], []

    class ObservedVectors:
        def __getattr__(self, name):
            return getattr(vectors, name)

        def similarity_search_by_text(self, **kwargs):
            original_embedder = kwargs["text_embedder"]

            def observe_embedding(text):
                vector = original_embedder(text)
                query_vectors.append(list(vector))
                return vector

            candidates = vectors.similarity_search_by_text(**{**kwargs, "text_embedder": observe_embedding})
            vector_candidates.extend({"id": str(row.document.id), "score": float(row.score)}
                                     for row in candidates)
            return candidates

    engine = get_local_search_engine(
        config=config,
        reports=read_indexer_reports(tables["community_reports"], tables["communities"], 0),
        entities=read_indexer_entities(tables["entities"], tables["communities"], 0),
        text_units=read_indexer_text_units(tables["text_units"]),
        relationships=read_indexer_relationships(tables["relationships"]),
        covariates={"claims": []}, response_type="multiple paragraphs",
        description_embedding_store=ObservedVectors(),
    )
    original_context = mixed_context.build_text_unit_context

    def observe_units(*args, **kwargs):
        ranking.extend(unit.id for unit in kwargs["text_units"])
        return original_context(*args, **kwargs)

    params = {**engine.context_builder_params, "return_candidate_context": True,
              "max_context_tokens": max_tokens, "use_community_summary": True}
    previous = transport.calls
    with patch.object(mixed_context, "build_text_unit_context", observe_units):
        context = engine.context_builder.build_context(query=query, **params)
    if transport.calls - previous != 1 or len(query_vectors) != 1:
        raise RuntimeError("expected exactly one observed native query embedding")
    return {
        "query": query, "text_unit_ranking": ranking, "query_embedding": query_vectors[0],
        "native_vector_candidates": vector_candidates,
        "candidate_context": {name: json.loads(frame.to_json(orient="records"))
                              for name, frame in context.context_records.items()},
        "context_text": context.context_chunks, "answer_generation": "Not Run",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--corpus", required=True, type=Path)
    build.add_argument("--out", required=True, type=Path)
    build.add_argument("--arm", required=True, choices=ARMS)
    build.add_argument("--completion-ceiling", type=int, choices=(4096, 8192, 32768))
    query = commands.add_parser("query")
    query.add_argument("--graph", required=True, type=Path)
    query.add_argument("--out", required=True, type=Path)
    query.add_argument("--text", required=True)
    args = parser.parse_args()
    if args.command == "build":
        asyncio.run(build_graph(args.corpus, args.out, args.arm, args.completion_ceiling))
    else:
        build_info, corpus, tables = load_graph(args.graph)
        args.out.mkdir(parents=True, exist_ok=False)
        transport = Transport(args.out / "requests")
        if transport.endpoint != build_info["provider_base_url"]:
            raise ValueError("build and query require the same provider endpoint")
        result = native_context(args.graph, build_info, tables, args.text, transport)
        write_json(args.out / "context.json", {
            **result, "patient_id": corpus["patient_id"],
            "source_mappings": tables["text_units"][["id", "document_id", "text"]].to_dict("records"),
        })


if __name__ == "__main__":
    main()

