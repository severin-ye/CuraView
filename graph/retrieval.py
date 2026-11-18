"""Patient-scoped retrieval comparisons; no labels enter runtime routing.

Dense vectors and retained GraphRAG rankings must come from actual upstream
artifacts. Missing backends fail explicitly, never become synthetic evidence.
"""
from __future__ import annotations

from collections import Counter
import math
import re

MODES = frozenset(('bm25', 'dense', 'hybrid', 'graph', 'dynamic'))


def anchor_query(sentence):
    """Mask a disputed dose/value while retaining names and calendar anchors.

    A lexical heuristic, not medical entity alignment or a learned router.
    """
    return re.sub(r'(?i)\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|ug|g|ml|mmol/l|mmhg|cm|mm)\b',
                  ' ', sentence).strip()


def _terms(text):
    return re.findall(r'[a-z0-9]+|[\u4e00-\u9fff]', text.lower())


def _bm25(query, docs):
    tokens = [Counter(_terms(d['text'])) for d in docs]
    lengths = [sum(t.values()) for t in tokens]
    avg = sum(lengths) / len(lengths) if lengths else 1.
    df = Counter(term for t in tokens for term in t)
    result = {}
    for doc, terms, size in zip(docs, tokens, lengths):
        score = 0.
        for word in set(_terms(query)):
            n = terms[word]
            if n:
                idf = math.log(1 + (len(docs) - df[word] + .5) / (df[word] + .5))
                score += idf * n * 2.5 / (n + 1.5 * (.25 + .75 * size / max(avg, 1.)))
        result[doc['evidence_id']] = score
    return result


def _cosine(a, b):
    if not a or len(a) != len(b) or any(type(v) not in (int, float) or not math.isfinite(v) for v in a + b):
        raise ValueError('invalid or incompatible dense vectors')
    na, nb = sum(v * v for v in a), sum(v * v for v in b)
    if not na or not nb:
        raise ValueError('zero dense vector')
    return sum(x * y for x, y in zip(a, b)) / math.sqrt(na * nb)


def retrieve(query, evidence, *, patient_id, mode='bm25', top_k=10, max_chars=12000,
             vectors=None, query_vector=None, graph_ranking=None):
    if mode not in MODES or type(top_k) is not int or top_k < 1 or type(max_chars) is not int or max_chars < 1:
        raise ValueError('invalid retrieval configuration')
    if not isinstance(query, str) or not query.strip():
        raise ValueError('query must be nonempty')
    docs = [dict(d) for d in evidence if str(d.get('patient_id', patient_id)) == str(patient_id)]
    ids = [d['evidence_id'] for d in docs]
    if len(set(ids)) != len(ids) or any(not isinstance(d.get('text'), str) or not d['text'].strip() for d in docs):
        raise ValueError('invalid or duplicate evidence')
    actual = mode
    route_reason = 'fixed_comparison_arm'
    q = anchor_query(query) or query
    if mode == 'dynamic':
        if re.search(r'(?i)\d\s*(?:mg|mcg|ml|mmhg|cm|mm)\b', query):
            actual, route_reason = 'bm25', 'predeclared_numeric_anchor_rule'
        elif vectors is not None and query_vector is not None:
            actual, route_reason = 'hybrid', 'predeclared_semantic_fusion_rule'
        else:
            actual, route_reason = 'bm25', 'dense_unavailable_explicit_lexical_fallback'
    lexical = _bm25(q, docs)
    dense = {}
    if actual in ('dense', 'hybrid'):
        if vectors is None or query_vector is None or any(i not in vectors for i in ids):
            raise ValueError('actual dense vectors required for all patient evidence')
        dense = {i: _cosine(list(query_vector), list(vectors[i])) for i in ids}
    if actual == 'bm25':
        scores = lexical
    elif actual == 'dense':
        scores = dense
    elif actual == 'hybrid':
        scores = dict.fromkeys(ids, 0.)
        for channel in (lexical, dense):
            for rank, i in enumerate(sorted(ids, key=lambda i: (-channel[i], i)), 1):
                scores[i] += 1 / (60 + rank)
    else:
        if graph_ranking is None or len(set(graph_ranking)) != len(graph_ranking) or any(i not in ids for i in graph_ranking):
            raise ValueError('retained patient-scoped GraphRAG evidence ranking required')
        scores = {i: 1 / (rank + 1) for rank, i in enumerate(graph_ranking)}
    ranked = sorted((d for d in docs if d['evidence_id'] in scores), key=lambda d: (-scores[d['evidence_id']], d['evidence_id']))
    candidates, selected, used = [], [], 0
    for rank, d in enumerate(ranked, 1):
        reason = None
        if rank > top_k:
            reason = 'outside_top_k'
        elif used + len(d['text']) > max_chars:
            reason = 'complete_record_exceeds_remaining_budget'
        row = {**d, 'rank': rank, 'score': None if actual=='graph' else scores[d['evidence_id']],
               'derived_rank_weight':scores[d['evidence_id']] if actual=='graph' else None,
               'lexical_score': lexical[d['evidence_id']], 'dense_score': dense.get(d['evidence_id']),
               'selected': reason is None, 'exclusion_reason': reason}
        candidates.append(row)
        if reason is None:
            selected.append(dict(d))
            used += len(d['text'])
    return {'schema': 'curaview.retrieval-comparison.v1', 'requested_mode': mode,
            'actual_mode': actual, 'route_reason': route_reason, 'patient_id': str(patient_id),
            'query': query, 'anchor_query': q, 'candidates': candidates, 'selected': selected,
            'budget': {'unit': 'unicode_characters', 'limit': max_chars, 'used': used, 'complete_records_only': True},
            'top_k': top_k, 'status': 'FOUND_PARTIAL' if selected else 'UNKNOWN',
            'native_internal_ranking_observed': False,
            'ranking_origin':'supplied_graph_order_not_independently_verified' if actual=='graph' else actual,
            'clinical_gold': False}


def diagnose(result, reference=None):
    """Offline reference-only attribution; never feed these states to a router."""
    out = {'status': result['status'], 'recall': None, 'sufficiency': None,
           'reference_available': False, 'clinical_gold': False, 'diagnostic_only': True}
    if not reference or reference.get('verified') is not True:
        return out
    selected = {d['evidence_id'] for d in result['selected']}
    relevant = set(reference.get('relevant_ids', []))
    groups = [set(g) for g in reference.get('sufficient_groups', []) if g]
    out['reference_available'] = True
    if relevant:
        out['recall'] = len(relevant & selected) / len(relevant)
    if groups:
        out['sufficiency'] = int(any(g <= selected for g in groups))
    if out['sufficiency'] == 1:
        out['status'] = 'FOUND_SUFFICIENT'
    elif relevant - selected:
        out['status'] = 'KNOWN_MISS'
    elif not relevant and reference.get('corpus_search_completed') is True and reference.get('search_scope'):
        out['status'] = 'NOT_FOUND_IN_CORPUS'
    return out
