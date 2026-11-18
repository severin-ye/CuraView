"""Strict future-output contract and tolerant historical diagnostics.

The diagnostic extractor does not validate or relabel historical predictions.
The strict parser accepts only the frozen prompt's single four-field JSON object.
"""
from __future__ import annotations
import json
import re
from typing import Any

GRADES = frozenset(('E1', 'E2', 'E3', 'E4'))
TYPES = frozenset(('diagnosis_error', 'medication_error', 'exam_result_error',
                   'time_error', 'value_error', 'negation_error', 'invented_fact'))
KEYS = frozenset(('reasoning', 'is_hallucination', 'hallucination_type', 'evidence_grade'))

def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate_json_key')
        result[key] = value
    return result

def parse_strict_detection(text: Any) -> dict:
    """Validate the complete output without extraction, repair or coercion."""
    result = dict(status='invalid', y_pred=None, evidence_grade=None,
                  hallucination_type=None, reasoning=None, error=None)
    if not isinstance(text, str):
        result['error'] = 'output_not_string'
        return result
    try:
        payload = json.loads(text, object_pairs_hook=_unique_pairs,
                             parse_constant=lambda value: (_ for _ in ()).throw(ValueError('non_json_constant')))
    except (ValueError, TypeError):
        result['error'] = 'not_single_json_object_or_duplicate_keys'
        return result
    if not isinstance(payload, dict) or set(payload) != KEYS:
        result['error'] = 'requires_exact_four_keys'
        return result
    grade = payload['evidence_grade']
    if not isinstance(grade, str) or grade not in GRADES:
        result['error'] = 'invalid_evidence_grade'
        return result
    result['evidence_grade'] = grade
    if type(payload['is_hallucination']) is not bool:
        result['error'] = 'label_not_boolean'
        return result
    if not isinstance(payload['reasoning'], str) or not payload['reasoning'].strip():
        result['error'] = 'reasoning_not_nonempty_string'
        return result
    positive = grade in ('E3', 'E4')
    if payload['is_hallucination'] != positive:
        result['error'] = 'label_grade_conflict'
        return result
    typ = payload['hallucination_type']
    if (not positive and typ is not None) or (positive and (not isinstance(typ, str) or typ not in TYPES)):
        result['error'] = 'invalid_hallucination_type'
        return result
    return dict(status='valid', y_pred=int(positive), evidence_grade=grade,
                hallucination_type=typ, reasoning=payload['reasoning'], error=None)

def extract_final_output(rawtext: Any) -> dict:
    """Recognize legacy final blocks, preserving explicit label absence.

    Mirrors legacy data_contract's last JSON with evidence_grade, otherwise its
    last 1200 characters' Chinese 判断结果 block. E0 remains recognizable only
    for diagnostics; it is never accepted by parse_strict_detection.
    """
    result = dict(payload={}, format=None, explicit_label=None, recognized=False,
                  strict_valid=False)
    if not isinstance(rawtext, str):
        return result
    result['strict_valid'] = parse_strict_detection(rawtext)['status'] == 'valid'
    for match in reversed(list(re.finditer(r'\{', rawtext))):
        try:
            payload, _ = json.JSONDecoder().raw_decode(rawtext[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and 'evidence_grade' in payload:
            result.update(payload=payload, format='json', recognized=True)
            explicit = payload.get('is_hallucination')
            if type(explicit) is bool:
                result['explicit_label'] = explicit
            # Legacy parser falls back to localized flag when JSON omitted it.
            if explicit is None:
                result['explicit_label'] = _localized_label(rawtext)
            return result
    tail = rawtext[-1200:]
    marker = tail.rfind('判断结果')
    if marker < 0:
        return result
    tail = tail[marker:]
    grades = list(re.finditer(r'证据等级\s*[:：]\s*(E[0-4])\b', tail, re.I))
    if grades:
        explicit = _localized_label(tail)
        payload = {'evidence_grade': grades[-1].group(1).upper()}
        if explicit is not None:
            payload['is_hallucination'] = explicit
        result.update(payload=payload, format='legacy_chinese', explicit_label=explicit, recognized=True)
    return result

def _localized_label(text):
    final_text = text[text.rfind('判断结果'):] if '判断结果' in text else text
    match = re.search(r'是否为幻觉\s*[:：]\s*(是|否|true|false|yes|no)(?![\w])', final_text, re.I)
    return match.group(1).lower() in ('是','true','yes') if match else None
