# 医学幻觉检测结果修正参考模板

本文件维护修正任务的判定口径。当前 `curaview/_03_detection/validation/rewriter.py` 通过 `detect_func` 重新执行完整检测，并不加载本模板；实际检测提示由检测入口加载。

## 任务

根据待检测句子、患者检索证据和原判断重新判定。先检查原证据等级是否正确，再修正标签；不能为了保持原等级而保留错误分类。不得编造患者事实或把医学常识当作患者证据。

## 证据规则

- **E1：直接明确支持。** 患者结构化字段或非结构化文本直接支持陈述；entities、relationships、reports等字段形态不决定等级。
- **E2：间接推断支持。** 需要从患者证据作间接推断，不能仅凭常识推测。
- **E3：无支持且无矛盾。** 本次检索证据既不支持也不反驳陈述；适用于整句，不限新增片段，不等于证明现实世界中必然错误。
- **E4：直接矛盾。** 陈述与患者事实直接冲突，不是危害严重程度。

优先级：E4 > E1 > E2 > E3。E1/E2必须为 `hallucination_status=false`、`hallucination_type=null`；E3/E4必须为 `hallucination_status=true`，并选择合法类型，无非医学句子例外。重写不得通过reasoning关键词绕过一致性规则。

## 输出

```json
{
  "reasoning": "说明原判断的问题、引用实际患者证据并解释修正依据",
  "hallucination_status": false,
  "hallucination_type": null,
  "evidence_grade": "E1"
}
```

有幻觉时类型只能是 diagnosis_error、medication_error、exam_result_error、time_error、value_error、negation_error、invented_fact。无幻觉时类型必须为null。


## English counterpart

This maintained reference template defines correction criteria. The current `curaview/_03_detection/validation/rewriter.py` reruns full detection through `detect_func`; it does not load this template. Detection entrypoints load the actual detection prompt.

Reassess the sentence against supplied patient evidence and the original judgment. Verify the grade before repairing the status; do not preserve a wrong grade merely to keep the original classification. Never invent patient facts or substitute general medical knowledge for patient evidence.

- E1: explicit direct support from structured or unstructured patient evidence. The entities, relationships, or reports field does not determine the grade.
- E2: indirect inference supported by patient evidence; general knowledge alone is insufficient.
- E3: retrieved evidence neither supports nor contradicts the statement. This applies to whole sentences, beyond added fragments, and does not establish real-world falsehood.
- E4: direct contradiction of patient facts, not a harm-severity grade.

Priority: E4 > E1 > E2 > E3. E1/E2 require false and null type. E3/E4 require true and a valid type, with no nonmedical exception. Reasoning keywords cannot bypass consistency validation.

Return the JSON fields shown above: reasoning must explain the original problem, cite supplied patient evidence, and justify the correction; hallucination_status is Boolean; hallucination_type is null for false or one of diagnosis_error, medication_error, exam_result_error, time_error, value_error, negation_error, invented_fact for true; evidence_grade is E1–E4. The displayed E1/false/null JSON illustrates the supported case.

If evidence is missing, state that limitation and do not invent quotations.
