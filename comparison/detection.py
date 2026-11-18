"""Fixed-version comparison detection rendering and validation."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).parent
PROTECTED = (ROOT / "detection_prompt.md").read_text(encoding="utf-8")
FRAGMENTS = json.loads((ROOT / "detection_fragments.json").read_text(encoding="utf-8"))

def render_detection_case(row, citation_aliases=False):
    """Render the existing S/D citation view without I/O or reference changes."""
    runtime=row;mapping={}
    if citation_aliases:
        aliases={'S'+str(i):e['evidence_id'] for i,e in enumerate(row['provided_evidence'],1)}
        documents={e['evidence_id']:'D'+str(i) for i,e in enumerate(row['source_evidence'],1)}
        evidence=[]
        for alias,e in zip(aliases,row['provided_evidence']):
            source=e.get('source_evidence_id',e['evidence_id'])
            evidence.append({'evidence_id':alias,'text':e['text'],
                             'source_document_id':documents.get(source)})
        sentence=row['sentence']
        for original,alias in sorted(documents.items(),key=lambda p:-len(p[0])):
            sentence=sentence.replace(original,alias)
        runtime={**row,'sentence':sentence,'provided_evidence':evidence}
        mapping={'sample_id':row['sample_id'],'citation_to_original':aliases,
                 'document_to_original':{v:k for k,v in documents.items()},
                 'original_sentence':row['sentence'],'rendered_sentence':sentence,
                 'source_text_changed':False,'clinical_reference_changed':False}
    return runtime,mapping


def detection_messages(case, strategy):
    if not isinstance(strategy, str) or not strategy.strip():
        raise ValueError('strategy must be nonempty text')
    visible_ids = [e['evidence_id'] for e in case['provided_evidence']]
    return [{'role': 'system', 'content': PROTECTED +
             FRAGMENTS[0] + json.dumps({'strategy': strategy}, ensure_ascii=False) +
             FRAGMENTS[1] + json.dumps(visible_ids, ensure_ascii=False) +
             FRAGMENTS[2]},
            {'role': 'user', 'content': json.dumps({'sentence': case['sentence'],
             'provided_evidence': case['provided_evidence']}, ensure_ascii=False)}]


def citation_diagnostic(reasoning, evidence_ids, grade):
    ids=set(evidence_ids)
    brackets=set(re.findall(r'\[([^\[\]]+)\]', reasoning))
    # IDs use ASCII letters, digits, underscores and hyphens. Unicode prose
    # (including Chinese) can touch an ID without becoming part of that ID.
    id_start=r'(?<![A-Za-z0-9_-])'; id_end=r'(?![A-Za-z0-9_-])'
    mentioned={e for e in ids if re.search(id_start+re.escape(e)+id_end, reasoning)}
    # The frozen cycle used ev-* hashes. Generic bracket prose and quote ellipses
    # are not claimed source IDs; an arbitrary unbracketed token is not an ID.
    claimed=set(re.findall(id_start+r'ev-[A-Za-z0-9_-]+'+id_end, reasoning))
    claimed.update(re.findall(id_start+r'(?:[0-9a-fA-F]{32,128}|[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})'+id_end,reasoning))
    claimed.update(re.findall(id_start+r'S[0-9]+'+id_end, reasoning))
    # E1-E4 are fixed protocol grade markers, also used in reasoning prose.
    # A genuinely provided ID with that spelling remains matched via `ids`.
    grade_markers={'E1','E2','E3','E4'}
    claimed.update(b for b in brackets if b not in grade_markers and re.fullmatch(r'(?:ev-[A-Za-z0-9_-]+|[A-Z]+[0-9]+)', b))
    unknown=claimed-ids
    issues=[]
    if grade in ('E1','E2','E4'):
        if not mentioned: issues.append('missing_source_id')
        elif not mentioned & brackets: issues.append('missing_bracket_format')
    if unknown: issues.append('unknown_source_id')
    missing=grade in ('E1','E2','E4') and not mentioned
    formatting=bool(mentioned and not mentioned & brackets)
    return {'citation_failed':bool(missing or unknown),'missing_required_id':bool(missing),
            'unknown_source_ids':sorted(unknown),'formatting_only':bool(formatting and not missing and not unknown),
            'noncitation_brackets':sorted(brackets-ids-claimed),'mentioned_source_ids':sorted(mentioned),
            'issues':issues,'mentioned_ids':sorted(mentioned),'unknown_ids':sorted(unknown),
            'non_id_brackets':sorted(brackets-ids-claimed),
            'source_existence_checked':True,'semantic_entailment_checked':False}


def quote_proxy(result,reference):
    quotes=reference.get('source_quotes',[]) if reference.get('source_grade') in ('E1','E2','E4') else []
    hits=[any(q['quote'] in u['text'] and q['evidence_id'] in (
        u.get('source_evidence_ids') or [u.get('source_evidence_id',u.get('document_id'))]) for u in result['selected']) for q in quotes]
    return {'quoted_fact_coverage':sum(hits)/len(hits) if hits else None,
            'quoted_facts':len(hits),'covered_quoted_facts':sum(hits),
            'authority':'limited_review_quote_proxy_not_exhaustive_retrieval_gold',
            'source_grade_prediction_agreement_is_clinical_accuracy':False}
