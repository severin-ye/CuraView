const form = document.querySelector("form[data-action]");
const status = document.querySelector("#status");
const result = document.querySelector("#result");
const recent = document.querySelector("#recent");
const states = {pending: "等待", running: "运行中", completed: "完成", partial: "部分完成", failed: "失败"};

async function request(url, payload) {
  const response = await fetch(url, payload === undefined ? {} : {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "请求失败");
  return data;
}

function display(value) {
  result.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  result.hidden = false;
}

async function refresh() {
  if (!recent) return;
  const tasks = await request("/detection/api/list");
  const nodes = tasks.map((task) => {
    const item = document.createElement("li");
    item.textContent = `${task.patient_id} · ${states[task.status]} · ${task.task_id}`;
    return item;
  });
  recent.replaceChildren(...nodes);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = form.querySelector("button");
  button.disabled = true;
  result.hidden = true;
  status.textContent = "正在处理…";
  const data = Object.fromEntries(new FormData(form));
  const action = form.dataset.action;
  try {
    if (action === "detect") {
      const {task_id} = await request("/detection/api/run", data);
      status.textContent = `检测任务 ${task_id} 正在处理…`;
      let task;
      do {
        await new Promise((resolve) => setTimeout(resolve, 750));
        task = await request(`/detection/api/status/${task_id}`);
      } while (task.status === "pending" || task.status === "running");
      const finished = await request(`/detection/api/result/${task_id}`);
      display(finished.report || finished.error);
      status.textContent = `检测${states[finished.status]} · ${task_id}`;
      await refresh();
      return;
    }
    if (action === "evaluate") {
      display(await request("/evaluation/api/compare", {
        manifest: JSON.parse(data.manifest),
        predictions: JSON.parse(data.predictions),
      }));
    } else if (action === "query") {
      display(await request("/graph/api/search", data));
    } else {
      const {lines} = await request(`/logs/api/tail?task_id=${encodeURIComponent(data.task_id)}`);
      display(lines.join("") || "该检测任务尚无日志。");
    }
    status.textContent = "完成";
  } catch (error) {
    status.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

refresh().catch(() => { status.textContent = "未能读取最近检测。"; });
