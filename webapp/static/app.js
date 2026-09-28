const state = {
  mode: "new",
  configs: [],
  runs: [],
  currentJobId: null,
  pollTimer: null,
  cudaAvailable: null,
};

const $ = (id) => document.getElementById(id);
const nullableNumber = (id) => {
  const value = $(id).value.trim();
  return value === "" ? null : Number(value);
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`);
  return body;
}

function selectedConfig() {
  return state.configs.find((item) => item.name === $("config-name").value);
}

function setMode(mode) {
  state.mode = mode;
  const isContinue = mode === "continue";
  $("mode-new").classList.toggle("active", !isContinue);
  $("mode-new").setAttribute("aria-selected", String(!isContinue));
  $("mode-continue").classList.toggle("active", isContinue);
  $("mode-continue").setAttribute("aria-selected", String(isContinue));
  document.querySelectorAll(".new-only").forEach((element) => element.classList.toggle("hidden", isContinue));
  document.querySelectorAll(".continue-only").forEach((element) => element.classList.toggle("hidden", !isContinue));
  $("launch-label").textContent = isContinue ? "開始接續訓練" : "開始全新訓練";
  updateOutputPreview();
  if (isContinue) loadRuns();
}

function applyDefaults(config) {
  if (!config) return;
  const defaults = config.defaults;
  $("config-meta").textContent = `${config.label} · 下一個輸出編號 ${config.next_run_number}`;
  $("device").value = defaults.device;
  applyCudaAvailability();
  $("batch-size").value = defaults.batch_size;
  $("num-workers").value = defaults.num_workers;
  $("dataset-root").value = defaults.dataset_root;
  $("epochs").value = defaults.epochs;
  $("learning-rate").value = defaults.learning_rate;
  $("seed").value = defaults.seed;
  $("max-steps").value = defaults.max_steps_per_epoch ?? "";
  $("max-items").value = defaults.max_items ?? "";
  $("checkpoint-every").value = defaults.checkpoint_every;
  $("sample-every").value = defaults.sample_every;
  $("next-run").textContent = `下一次：#${config.next_run_number}`;
  updateOutputPreview();
  if (state.mode === "continue") loadRuns();
}

function applyCudaAvailability() {
  const select = $("device");
  const cudaOption = select.querySelector('option[value="cuda"]');
  if (!cudaOption || state.cudaAvailable === null) return;
  cudaOption.disabled = !state.cudaAvailable;
  cudaOption.textContent = state.cudaAvailable ? "CUDA GPU" : "CUDA GPU（容器未偵測到）";
  if (!state.cudaAvailable && select.value === "cuda") select.value = "auto";
}

function updateOutputPreview() {
  const config = selectedConfig();
  if (!config) return;
  $("output-preview").textContent = `runs/${config.category}/${config.next_run_number}`;
}

async function loadConfigs() {
  try {
    state.configs = await api("/api/configs");
    const select = $("config-name");
    select.innerHTML = state.configs.map((config) => `<option value="${config.name}">${config.name}</option>`).join("");
    if (!state.configs.length) {
      select.innerHTML = '<option value="">找不到 YAML 設定檔</option>';
      $("launch-button").disabled = true;
      return;
    }
    applyDefaults(state.configs[0]);
  } catch (error) {
    showError(error.message);
  }
}

async function loadRuns() {
  const config = selectedConfig();
  if (!config) return;
  const select = $("source-run");
  select.innerHTML = '<option value="">正在讀取…</option>';
  try {
    state.runs = await api(`/api/configs/${encodeURIComponent(config.name)}/runs`);
    const validRuns = state.runs.filter((run) => !run.error);
    select.innerHTML = validRuns.length
      ? validRuns.map((run) => `<option value="${run.run_number}">第 ${run.run_number} 次 · epoch ${run.epoch} · ${run.checkpoint}</option>`).join("")
      : '<option value="">沒有可接續的訓練</option>';
    select.disabled = !validRuns.length;
    updateSourceMeta();
  } catch (error) {
    select.innerHTML = '<option value="">讀取失敗</option>';
    showError(error.message);
  }
}

function updateSourceMeta() {
  const run = state.runs.find((item) => String(item.run_number) === $("source-run").value);
  $("source-meta").textContent = run
    ? `已完成 epoch ${run.epoch}，global step ${run.global_step}，checkpoint LR ${run.learning_rate}`
    : "請先選擇有 checkpoint 的訓練。";
  if (run) $("continue-learning-rate").value = run.learning_rate;
}

function buildPayload() {
  const common = {
    config_name: $("config-name").value,
    dataset_root: $("dataset-root").value.trim(),
    device: $("device").value,
    batch_size: Number($("batch-size").value),
    seed: Number($("seed").value),
    max_steps_per_epoch: nullableNumber("max-steps"),
    checkpoint_every: Number($("checkpoint-every").value),
    sample_every: Number($("sample-every").value),
    max_items: nullableNumber("max-items"),
    num_workers: Number($("num-workers").value),
  };
  if (state.mode === "new") {
    return {
      ...common,
      epochs: Number($("epochs").value),
      learning_rate: Number($("learning-rate").value),
    };
  }
  return {
    ...common,
    source_run: Number($("source-run").value),
    additional_epochs: Number($("additional-epochs").value),
    learning_rate: $("reset-learning-rate").checked ? Number($("continue-learning-rate").value) : null,
  };
}

async function submitTraining(event) {
  event.preventDefault();
  hideError();
  if (state.mode === "continue" && !$("source-run").value) {
    showError("請先選擇可接續的訓練編號。");
    return;
  }
  const button = $("launch-button");
  button.disabled = true;
  $("launch-label").textContent = "正在建立工作…";
  try {
    const endpoint = state.mode === "new" ? "/api/train/new" : "/api/train/continue";
    const job = await api(endpoint, { method: "POST", body: JSON.stringify(buildPayload()) });
    state.currentJobId = job.id;
    renderJob(job);
    startPolling();
    await loadConfigs();
  } catch (error) {
    showError(error.message);
  } finally {
    button.disabled = false;
    $("launch-label").textContent = state.mode === "new" ? "開始全新訓練" : "開始接續訓練";
  }
}

function renderJob(job) {
  $("job-status").textContent = job.status.toUpperCase();
  $("job-status").className = `status-badge ${job.status}`;
  $("job-summary").innerHTML = `
    <div><small>模式</small><strong>${job.mode === "continue" ? "再次訓練" : "全新訓練"}</strong></div>
    <div><small>輸出編號</small><strong>${job.category} / ${job.run_number}</strong></div>`;
  const terminal = $("terminal");
  terminal.textContent = job.log || "$ 工作已建立，等待第一筆輸出…";
  terminal.scrollTop = terminal.scrollHeight;
  const active = ["starting", "running", "stopping"].includes(job.status);
  $("stop-job").classList.toggle("hidden", !active);
  $("stop-job").disabled = job.status === "stopping";
}

function renderRecentJobs(items) {
  const container = $("recent-jobs-list");
  if (!items.length) {
    container.innerHTML = '<p class="empty-state">目前沒有工作紀錄。</p>';
    return;
  }
  container.innerHTML = items.slice(0, 6).map((job) => `
    <div class="recent-item" data-job-id="${job.id}">
      <strong>${job.category} / #${job.run_number}</strong><span>${job.status}</span>
      <small>${job.mode === "continue" ? "再次訓練" : "全新訓練"} · ${new Date(job.created_at).toLocaleString()}</small>
    </div>`).join("");
  container.querySelectorAll(".recent-item").forEach((item) => {
    item.addEventListener("click", async () => {
      state.currentJobId = item.dataset.jobId;
      renderJob(await api(`/api/jobs/${state.currentJobId}`));
      startPolling();
    });
  });
}

async function refreshJobs() {
  try {
    const items = await api("/api/jobs");
    renderRecentJobs(items);
    if (!state.currentJobId && items.length) {
      state.currentJobId = items[0].id;
      renderJob(items[0]);
    }
  } catch (error) {
    showError(error.message);
  }
}

function startPolling() {
  clearInterval(state.pollTimer);
  state.pollTimer = setInterval(async () => {
    if (!state.currentJobId) return;
    try {
      const job = await api(`/api/jobs/${state.currentJobId}`);
      renderJob(job);
      await refreshJobs();
      if (["completed", "failed", "stopped"].includes(job.status)) {
        clearInterval(state.pollTimer);
        await loadConfigs();
        if (state.mode === "continue") await loadRuns();
      }
    } catch (error) {
      clearInterval(state.pollTimer);
      showError(error.message);
    }
  }, 1500);
}

async function stopCurrentJob() {
  if (!state.currentJobId || !confirm("確定要停止目前的訓練嗎？已完成的 checkpoint 會保留。")) return;
  try {
    const job = await api(`/api/jobs/${state.currentJobId}/stop`, { method: "POST" });
    renderJob(job);
  } catch (error) {
    showError(error.message);
  }
}

function showError(message) {
  $("form-error").textContent = message;
  $("form-error").classList.remove("hidden");
}
function hideError() { $("form-error").classList.add("hidden"); }

async function checkHealth() {
  try {
    const health = await api("/api/health");
    state.cudaAvailable = health.cuda_available;
    applyCudaAvailability();
    $("health-pill").classList.add("online");
    $("health-text").textContent = health.cuda_available
      ? `CUDA READY · ${health.cuda_device_name}`
      : "API ONLINE · CUDA UNAVAILABLE";
  } catch (_) {
    $("health-pill").classList.add("offline");
    $("health-text").textContent = "API OFFLINE";
  }
}

$("mode-new").addEventListener("click", () => setMode("new"));
$("mode-continue").addEventListener("click", () => setMode("continue"));
$("config-name").addEventListener("change", () => applyDefaults(selectedConfig()));
$("source-run").addEventListener("change", updateSourceMeta);
$("reset-learning-rate").addEventListener("change", (event) => {
  $("continue-learning-rate").disabled = !event.target.checked;
});
$("training-form").addEventListener("submit", submitTraining);
$("refresh-jobs").addEventListener("click", refreshJobs);
$("stop-job").addEventListener("click", stopCurrentJob);

checkHealth();
loadConfigs();
refreshJobs();
