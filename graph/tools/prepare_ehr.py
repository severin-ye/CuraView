"""Prepare graph inputs from caller-provided, single-patient source records."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from graph.io import write_json

FORBIDDEN = {'target', 'target_text', 'discharge_target', 'discharge_summary',
             'reference_answer', 'gold', 'gold_label', 'label', 'hallucination'}

def flatten_ehr_record(record, include_discharge_target=True):
    """
    展平 EHR 记录中的嵌套结构
    
    Args:
        record: 原始 EHR 记录
        include_discharge_target: 是否包含 discharge_target，默认 True
        
    Returns:
        展平后的记录，包含必要的 text 和 id 字段
    """
    # 提取主要文本内容
    text_parts = []
    
    # 从 discharge 中提取文本
    if 'discharge' in record and isinstance(record['discharge'], dict):
        if 'text' in record['discharge']:
            text_parts.append(record['discharge']['text'])
    
    # 从 discharge_target 中提取文本（如果允许且存在）
    if include_discharge_target and 'discharge_target' in record and isinstance(record['discharge_target'], dict):
        if 'text' in record['discharge_target']:
            text_parts.append(f"\n[Discharge Target]\n{record['discharge_target']['text']}")
    
    # 从 radiology 中提取文本（如果存在）
    if 'radiology' in record and isinstance(record['radiology'], dict):
        if 'text' in record['radiology']:
            text_parts.append(f"\n[Radiology Report]\n{record['radiology']['text']}")
    
    # 合并文本
    text = '\n'.join(text_parts) if text_parts else ''
    
    # 生成唯一 ID
    record_id = None
    if 'universal' in record and isinstance(record['universal'], dict):
        if 'subject_id' in record['universal']:
            record_id = str(record['universal']['subject_id'])
    
    if not record_id:
        record_id = record.get('name', 'unknown')
    
    # 返回展平的记录
    return {
        'id': record_id,
        'text': text,
        'name': record.get('name', 'Unknown'),
        'diagnosis': record.get('diagnosis', {}),
        'edstays': record.get('edstays', {}),
    }


def build_text_from_items(
    record: Dict[str, Any],
    items: List[str]
) -> str:
    """
    从指定的 items 构建文本内容用于图谱生成
    
    Args:
        record: 包含指定字段的患者记录
        items: 要处理的字段列表
        
    Returns:
        构建的文本内容
    """
    text_parts = []
    
    # 处理每个 item
    for item in items:
        if item not in record:
            continue
            
        item_data = record[item]
        
        if item == 'triage':
            # 构建 triage 信息
            if isinstance(item_data, dict):
                triage_info = []
                if 'temperature' in item_data:
                    triage_info.append(f"体温: {item_data['temperature']}°F")
                if 'heartrate' in item_data:
                    triage_info.append(f"心率: {item_data['heartrate']} bpm")
                if 'resprate' in item_data:
                    triage_info.append(f"呼吸率: {item_data['resprate']} /min")
                if 'o2sat' in item_data:
                    triage_info.append(f"血氧饱和度: {item_data['o2sat']}%")
                if 'sbp' in item_data:
                    triage_info.append(f"收缩压: {item_data['sbp']} mmHg")
                if 'dbp' in item_data:
                    triage_info.append(f"舒张压: {item_data['dbp']} mmHg")
                if 'pain' in item_data:
                    triage_info.append(f"疼痛评分: {item_data['pain']}")
                if 'acuity' in item_data:
                    triage_info.append(f"紧急程度: {item_data['acuity']}")
                if 'chiefcomplaint' in item_data:
                    triage_info.append(f"主诉: {item_data['chiefcomplaint']}")
                
                if triage_info:
                    text_parts.append(f"[分诊信息 - 患者当前状态]\n" + "\n".join(triage_info))
        
        elif item == 'edstays':
            # 构建 edstays 信息
            if isinstance(item_data, dict):
                edstays_info = []
                if 'intime' in item_data:
                    edstays_info.append(f"入院时间: {item_data['intime']}")
                if 'outtime' in item_data:
                    edstays_info.append(f"出院时间: {item_data['outtime']}")
                if 'gender' in item_data:
                    edstays_info.append(f"性别: {item_data['gender']}")
                if 'race' in item_data:
                    edstays_info.append(f"种族: {item_data['race']}")
                if 'arrival_transport' in item_data:
                    edstays_info.append(f"到达方式: {item_data['arrival_transport']}")
                if 'disposition' in item_data:
                    edstays_info.append(f"处置结果: {item_data['disposition']}")
                
                if edstays_info:
                    text_parts.append(f"[急诊就诊记录 - 行政流程]\n" + "\n".join(edstays_info))
        
        elif item == 'diagnosis':
            # 构建 diagnosis 信息
            if isinstance(item_data, dict):
                diagnosis_info = []
                if 'icd_code' in item_data:
                    diagnosis_info.append(f"ICD编码: {item_data['icd_code']}")
                if 'icd_title' in item_data:
                    diagnosis_info.append(f"诊断名称: {item_data['icd_title']}")
                if 'icd_version' in item_data:
                    diagnosis_info.append(f"ICD版本: {item_data['icd_version']}")
                if 'seq_num' in item_data:
                    diagnosis_info.append(f"诊断序号: {item_data['seq_num']}")
                
                if diagnosis_info:
                    text_parts.append(f"[诊断信息 - 结构化诊断]\n" + "\n".join(diagnosis_info))
        
        elif item == 'radiology':
            # 构建 radiology 信息
            if isinstance(item_data, dict):
                radiology_info = []
                if 'note_type' in item_data:
                    radiology_info.append(f"报告类型: {item_data['note_type']}")
                if 'charttime' in item_data:
                    radiology_info.append(f"检查时间: {item_data['charttime']}")
                if 'text' in item_data:
                    radiology_info.append(f"\n检查报告内容:\n{item_data['text']}")
                
                if radiology_info:
                    text_parts.append(f"[影像学检查 - 专科检查结果]\n" + "\n".join(radiology_info))
    
    return '\n\n'.join(text_parts) if text_parts else ''


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _json_hash(value):
    return _digest(json.dumps(value, ensure_ascii=False, sort_keys=True,
                              separators=(',', ':')).encode('utf-8'))


def _file_record(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'size': path.stat().st_size}


def _subject(value):
    value = str(value).strip()
    if not value or not value.isascii() or not value.isdigit():
        raise ValueError('subject_id must be a nonempty decimal identifier')
    return str(int(value))


class _Lines:
    def __init__(self, stream):
        self.stream = stream
        self.consumed = []

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self.stream)
        self.consumed.append(line)
        return line


def _scan(record, selected=None):
    """Parse CSV transiently; construct evidence only for selected patients.

    A whole-file byte scan and CSV parsing are necessary to find all selected
    rows. Unselected rows are not assembled into structured evidence text or
    retained as evidence objects. selected=None is the legacy pool-count pass.
    """
    path = Path(record['path'])
    if 'discharge_target' in path.name.lower():
        raise ValueError('discharge_target is a target, not an evidence source')
    # Allow long radiology records without any silent clipping.
    csv.field_size_limit(100_000_000)
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        lines = _Lines(stream)
        reader = csv.reader(lines, strict=True)
        try:
            headers = next(reader)
        except StopIteration:
            raise ValueError('empty evidence CSV') from None
        if len(set(headers)) != len(headers) or 'subject_id' not in headers:
            raise ValueError('CSV requires unique headers including subject_id')
        if FORBIDDEN.intersection(h.strip().lower() for h in headers):
            raise ValueError('target/label columns are forbidden as evidence')
        if len(headers) < 2:
            raise ValueError('CSV has no evidence fields')
        while True:
            lines.consumed.clear()
            start = reader.line_num + 1
            try:
                values = next(reader)
            except StopIteration:
                break
            if len(values) != len(headers):
                raise ValueError(f'malformed CSV record at line {start}')
            row = dict(zip(headers, values))
            patient = _subject(row['subject_id'])
            if selected is not None and patient not in selected:
                yield patient, 0, None
                continue
            text = row.get('text')
            if text is None:
                text = '\n'.join(f'{key}: {value}' for key, value in row.items()
                                 if key != 'subject_id' and value.strip())
            if not text.strip():
                raise ValueError(f'empty evidence record at line {start}')
            raw_hash = _digest(''.join(lines.consumed).encode('utf-8'))
            evidence = None
            if selected is not None and patient in selected:
                evidence = {'evidence_id': 'ev-' + _json_hash([record['sha256'], start, raw_hash])[:24],
                            'text': text, 'source_path': record['path'],
                            'source_sha256': record['sha256'], 'row_number': start,
                            'row_end_number': reader.line_num, 'row_sha256': raw_hash,
                            'row_hash_encoding': 'decoded-utf8-preserving-newlines',
                            'text_sha256': _digest(text.encode('utf-8'))}
            yield patient, len(text), evidence


def csv_documents(paths, patient):
    patient = _subject(patient)
    records = [_file_record(path) for path in paths]
    if len({record["path"] for record in records}) != len(records):
        raise ValueError("CSV sources must be distinct files")
    documents = []
    for record in records:
        for _, _, item in _scan(record, {patient}):
            if item is not None:
                documents.append({"id": item["evidence_id"],
                                  "title": f"Source row {item['row_number']}", "text": item["text"]})
        if _file_record(record["path"]) != record:
            raise ValueError("source changed while it was being read")
    return documents


def jsonl_documents(path, patient, mode):
    record = _file_record(path)
    documents = []
    with Path(record["path"]).open(encoding="utf-8-sig") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {number} must be an object")
            identity = row.get("universal", {})
            sid = identity.get("subject_id") if isinstance(identity, dict) else None
            sid = sid if sid is not None else row.get("patient_id", row.get("id", row.get("name")))
            if str(sid) != patient:
                continue
            if mode == "structured":
                text = build_text_from_items(row, ["triage", "edstays", "diagnosis", "radiology"])
            elif mode == "discharge":
                discharge = row.get("discharge", {})
                text = discharge.get("text", "") if isinstance(discharge, dict) else ""
            else:
                text = flatten_ehr_record(row, include_discharge_target=False)["text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"matching JSONL record at line {number} has no usable evidence text")
            evidence_id = "ev-" + _json_hash([record["sha256"], number, _digest(line.encode())])[:24]
            documents.append({"id": evidence_id, "title": f"Source record {number}", "text": text})
    if _file_record(record["path"]) != record:
        raise ValueError("source changed while it was being read")
    return documents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", nargs="+", type=Path)
    source.add_argument("--jsonl", type=Path)
    parser.add_argument("--patient", required=True)
    parser.add_argument("--mode", choices=("discharge-radiology", "structured", "discharge"),
                        default="discharge-radiology", help="format of nested JSONL input")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    patient = args.patient.strip()
    if not patient:
        raise ValueError("patient ID is required")
    documents = (csv_documents(args.csv, patient) if args.csv
                 else jsonl_documents(args.jsonl, patient, args.mode))
    if not documents:
        raise ValueError("no matching source records found")
    write_json(args.out, {"patient_id": _subject(patient) if args.csv else patient,
                          "documents": documents})


if __name__ == "__main__":
    main()
