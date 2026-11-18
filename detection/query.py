"""Query the detector's configured GraphRAG 2.7 index."""
import argparse
import json
import sys

from curaview._02_graph.retrieval.graphrag_query_tool import query_graphrag_for_sentence
from curaview._03_detection.validation.detection_errors import GraphRAGQueryError, require_query_evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-id", required=True)
    args = parser.parse_args()
    query = sys.stdin.read().strip()
    if not query:
        parser.error("Pass a query on stdin")
    try:
        output = require_query_evidence(query_graphrag_for_sentence.invoke(
            {"sentence": query, "patient_id": args.patient_id}
        ))
    except GraphRAGQueryError:
        raise SystemExit("GraphRAG query failed. Check the configured index and provider.")
    print(json.dumps(json.loads(output), ensure_ascii=False))


if __name__ == "__main__":
    main()
