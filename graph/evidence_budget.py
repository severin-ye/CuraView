"""Keep complete evidence records within the unchanged tokenizer budget."""
import hashlib
import json


def serialize(value):
    return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)


def select_evidence(tokenizer, evidence, budget=12000):
    def size(text):
        return len(tokenizer.encode(text, add_special_tokens=False))
    try:
        context = json.loads(evidence)
    except (ValueError, TypeError) as error:
        raise ValueError('evidence must be complete JSON') from error
    if not isinstance(context, dict):
        raise ValueError('evidence JSON must be an object of evidence tables')
    if budget < size('{}'):
        raise ValueError('evidence budget cannot hold a JSON object')
    original_tokens = size(evidence)
    if original_tokens <= budget:
        return evidence, {'policy':'complete_records_in_original_table_order_v1',
                          'dropped_records':[], 'selected_records':
                          {k:len(v) for k,v in context.items() if isinstance(v,list)}}
    selected, dropped = {}, []
    for key in sorted(context):
        value = context[key]
        if isinstance(value, list):
            trial = {**selected, key:[]}
            if size(serialize(trial)) <= budget:
                selected = trial
            for index, record in enumerate(value):
                trial = {**selected, key:[*selected.get(key,[]), record]}
                if size(serialize(trial)) <= budget:
                    selected = trial
                else:
                    dropped.append({'path':[key,index], 'reason':'whole_record_exceeds_remaining_budget',
                                    'sha256':hashlib.sha256(serialize(record).encode()).hexdigest()})
        else:
            trial = {**selected, key:value}
            if size(serialize(trial)) <= budget:
                selected = trial
            else:
                dropped.append({'path':[key], 'reason':'whole_value_exceeds_remaining_budget',
                                'sha256':hashlib.sha256(serialize(value).encode()).hexdigest()})
    used = serialize(selected)
    return used, {'policy':'complete_records_in_original_table_order_v1',
                  'dropped_records':dropped, 'selected_records':
                  {k:len(v) for k,v in selected.items() if isinstance(v,list)}}
