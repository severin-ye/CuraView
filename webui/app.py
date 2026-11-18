"""Local CuraView research interface over the public detector and evaluator."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from collections import deque
import subprocess
import sys
import tempfile
from threading import Lock
from uuid import uuid4

from flask import Flask, abort, jsonify, render_template, request

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection.subprocess_runner import DETECTOR, run_detection, valid_patient_id
from evaluation.formal_metrics import join_pairs, patient_macro_metrics, sentence_micro_metrics

app = Flask(__name__)
app.config.update(MAX_CONTENT_LENGTH=2_000_000, TRUSTED_HOSTS=["127.0.0.1", "localhost"])
work_dir = Path(os.environ.get("CURAVIEW_WEBUI_RUN_ROOT") or Path.cwd() / "runs" / "webui").resolve()
tasks = {}
lock = Lock()
executor = ThreadPoolExecutor(max_workers=1)
terminal = {"completed", "partial", "failed"}

pages = {
    "detection": "检测",
    "evaluation": "评估",
    "graph": "图谱查询",
    "logs": "运行日志",
}


@app.before_request
def check_origin():
    if request.method == "POST":
        origin = request.headers.get("Origin")
        if origin and origin.rstrip("/") != request.host_url.rstrip("/"):
            abort(403)


@app.after_request
def protect_response(response):
    response.headers.update({
        "Cache-Control": "no-store",
        "Content-Security-Policy": "default-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'",
        "X-Content-Type-Options": "nosniff",
    })
    return response


@app.get("/")
@app.get("/<page>")
def dashboard(page="detection"):
    if page not in pages:
        abort(404)
    return render_template("dashboard.html", pages=pages, page=page)


def payload():
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def run_task(task_id, text, patient_id):
    with lock:
        tasks[task_id]["status"] = "running"
    try:
        report = run_detection(text, patient_id, work_dir / task_id)
        result = {"status": report["processing_status"], "report": report}
    except subprocess.TimeoutExpired:
        result = {"status": "failed", "error": "检测超时。可查看该任务的日志。"}
    except (OSError, RuntimeError, ValueError):
        result = {"status": "failed", "error": "检测未生成有效报告。可查看该任务的日志。"}
    with lock:
        tasks[task_id].update(result)


@app.post("/detection/api/run")
def detection_run():
    data = payload()
    text, patient_id = data.get("text"), data.get("patient_id")
    if not isinstance(text, str) or not text.strip() or not valid_patient_id(patient_id):
        return jsonify(error="请填写待检测文本及 patient_加数字格式的索引 ID。"), 400
    task_id = uuid4().hex
    with lock:
        tasks[task_id] = {"task_id": task_id, "patient_id": patient_id, "status": "pending"}
    executor.submit(run_task, task_id, text, patient_id)
    return jsonify(task_id=task_id, status="pending"), 202


@app.get("/detection/api/status/<task_id>")
def detection_status(task_id):
    with lock:
        task = tasks.get(task_id)
        if task is None:
            return jsonify(error="任务不存在。"), 404
        return jsonify(**{key: value for key, value in task.items() if key != "report"})


@app.get("/detection/api/result/<task_id>")
def detection_result(task_id):
    with lock:
        task = tasks.get(task_id)
        if task is None or task["status"] not in terminal:
            return jsonify(error="任务尚无结果。"), 404
        return jsonify(**task)


@app.get("/detection/api/list")
def detection_list():
    with lock:
        return jsonify([dict(task_id=task["task_id"], patient_id=task["patient_id"], status=task["status"])
                        for task in list(tasks.values())[-30:][::-1]])


@app.post("/evaluation/api/compare")
def evaluation_run():
    data = payload()
    manifest, predictions = data.get("manifest"), data.get("predictions")
    if not isinstance(manifest, list) or not manifest or not isinstance(predictions, list):
        return jsonify(error="请提供参考标签和预测结果的 JSON 数组。"), 400
    if not all(isinstance(row, dict) for row in manifest + predictions):
        return jsonify(error="数组中的每一项都必须是对象。"), 400
    try:
        rows = join_pairs(manifest, predictions, "patient_id", "sentence_id", "y_true", "y_pred")
        return jsonify(sentence_micro=sentence_micro_metrics(rows), patient_macro=patient_macro_metrics(rows))
    except ValueError as error:
        return jsonify(error=str(error)), 400


@app.post("/graph/api/search")
def graph_search():
    data = payload()
    query, patient_id = data.get("query"), data.get("patient_id")
    if not isinstance(query, str) or not query.strip() or not valid_patient_id(patient_id):
        return jsonify(error="请填写查询内容及 patient_加数字格式的索引 ID。"), 400
    work_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "PYTHONUTF8": "1",
        "CURAVIEW_LEGACY_TRACE_ROOT": str(work_dir / ("graph-" + uuid4().hex)),
    }
    if config := env.get("CURAVIEW_DETECTION_CONFIG"):
        env["CURAVIEW_DETECTION_CONFIG"] = str(Path(config).resolve())
    with tempfile.TemporaryDirectory(dir=work_dir) as temporary:
        env["CURAVIEW_DETECTION_LOG_ROOT"] = temporary
        try:
            result = subprocess.run(
                [sys.executable, str(DETECTOR.with_name("query.py")), "--patient-id", patient_id],
                input=query, capture_output=True, encoding="utf-8", timeout=120, cwd=temporary,
                env=env, check=True,
            )
            return jsonify(json.loads(result.stdout))
        except (OSError, subprocess.SubprocessError, ValueError):
            return jsonify(error="图谱查询失败。请检查索引和模型配置。"), 502


@app.get("/logs/api/tail")
def logs_tail():
    task_id = request.args.get("task_id")
    with lock:
        if task_id not in tasks:
            return jsonify(error="任务不存在。"), 404
    path = work_dir / task_id / "detector.log"
    if not path.is_file():
        return jsonify(lines=[])
    with path.open(encoding="utf-8", errors="replace") as stream:
        return jsonify(lines=list(deque(stream, maxlen=100)))


if __name__ == "__main__":
    work_dir.mkdir(parents=True, exist_ok=True)
    app.run(host="127.0.0.1", port=int(os.environ.get("CURAVIEW_PORT", "5000")))
