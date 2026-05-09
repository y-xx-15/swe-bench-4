const form = document.querySelector(".repair-panel");
const repairButton = document.querySelector("#repair-button");
const runMessage = document.querySelector("#run-message");
const views = Array.from(document.querySelectorAll(".view"));
const navLinks = Array.from(document.querySelectorAll(".nav a"));
const nav = document.querySelector(".nav");
const viewIds = new Set(views.map((view) => view.id));
const API_BASE = window.location.protocol === "file:" ? "http://127.0.0.1:8001" : "";
let repairCompleted = false;
let instanceLoadTimer;

const fields = {
  generatedTestCount: document.querySelector("#generated-test-count"),
  keptTestCount: document.querySelector("#kept-test-count"),
  testName: document.querySelector("#test-name"),
  coveredTests: document.querySelector("#covered-tests"),
  qualityScore: document.querySelector("#quality-score"),
  keptReason: document.querySelector("#kept-reason"),
  testFile: document.querySelector("#test-file"),
  testCode: document.querySelector("#test-code"),
  patchStatus: document.querySelector("#patch-status"),
  patchCode: document.querySelector("#patch-code"),
  beforeCode: document.querySelector("#before-code"),
  afterCode: document.querySelector("#after-code"),
  resolvedState: document.querySelector("#resolved-state"),
  passedCount: document.querySelector("#passed-count"),
  failedCount: document.querySelector("#failed-count"),
  applyMode: document.querySelector("#apply-mode"),
  analysisText: document.querySelector("#analysis-text"),
};

function navigateTo(viewId) {
  const targetId = viewIds.has(viewId) ? viewId : "home";
  if (targetId !== "home" && !repairCompleted) {
    setMessage("请先在主界面点击开始修复，完成后才能查看增强测试用例和修复结果。", "error");
    if (window.location.hash !== "#home") {
      window.location.hash = "home";
    }
    return;
  }
  views.forEach((view) => {
    view.classList.toggle("active", view.id === targetId);
  });
  navLinks.forEach((link) => {
    const isActive = link.getAttribute("href") === `#${targetId}`;
    link.classList.toggle("active", isActive);
    if (isActive) {
      link.setAttribute("aria-current", "page");
    } else {
      link.removeAttribute("aria-current");
    }
  });
  if (window.location.hash !== `#${targetId}`) {
    window.location.hash = targetId;
  }
}

function setFlowLocked(locked) {
  repairCompleted = !locked;
  nav.classList.toggle("locked", locked);
}

function syncViewFromHash() {
  navigateTo(window.location.hash.replace("#", "") || "home");
}

function setMessage(text, type = "") {
  runMessage.textContent = text;
  runMessage.className = `run-message ${type}`.trim();
}

function formatList(items) {
  if (!Array.isArray(items) || items.length === 0) {
    return "暂无";
  }
  return items.join(", ");
}

function extractAddedTestCode(patch) {
  if (!patch) {
    return "暂无增强测试用例";
  }
  const lines = patch.split("\n");
  const added = lines
    .filter((line) => line.startsWith("+") && !line.startsWith("+++"))
    .map((line) => line.slice(1));
  return added.join("\n").trim() || patch;
}

function extractPatchFile(patch, fallback) {
  const match = patch?.match(/\+\+\+ b\/([^\n]+)/);
  return match ? match[1].trim() : fallback || "增强测试文件";
}

function buildAfterCode(beforeCode, patch) {
  if (!beforeCode || !patch) {
    return beforeCode || "暂无修复后代码";
  }
  if (patch.includes("__slots__ = ()") && !beforeCode.includes("__slots__ = ()")) {
    const lines = beforeCode.split("\n");
    const docEndIndex = lines.findIndex((line, index) => index > 0 && line.trim() === '"""');
    if (docEndIndex !== -1) {
      const nextLines = [...lines];
      nextLines.splice(docEndIndex + 1, 0, "", "    __slots__ = ()");
      return nextLines.join("\n");
    }
  }
  return beforeCode;
}

function renderBugInput(result) {
  document.querySelector("#bug-code").value = result.bug_code || "";
  document.querySelector("#bug-desc").value = result.bug_description || "";
}

function renderResult(result) {
  const test = result.enhanced_tests?.[0] || {};
  const summary = result.summary || {};
  const analysis = summary.patch_analysis || {};

  fields.generatedTestCount.textContent = String(summary.enhanced_candidates_total ?? 0);
  fields.keptTestCount.textContent = String(summary.kept_enhanced_candidates ?? result.enhanced_tests?.length ?? 0);
  fields.testName.textContent = formatList(test.identifiers || summary.kept_enhanced_test_identifiers);
  fields.coveredTests.textContent = formatList(test.covered_original_tests || summary.original_test_identifiers);
  fields.qualityScore.textContent =
    typeof test.quality_score === "number" ? test.quality_score.toFixed(2) : "暂无";
  fields.keptReason.textContent = test.reason || "暂无筛选说明";
  fields.testFile.textContent = extractPatchFile(test.patch, "sympy/core/tests/test_basic.py");
  fields.testCode.textContent = extractAddedTestCode(test.patch);

  fields.patchStatus.textContent = summary.generated_patch_accepted ? "accepted" : "generated";
  fields.patchCode.textContent = result.patch || "暂无补丁内容";
  fields.beforeCode.textContent = result.bug_code || document.querySelector("#bug-code").value || "暂无修复前代码";
  fields.afterCode.textContent = buildAfterCode(fields.beforeCode.textContent, result.patch);
  fields.resolvedState.textContent = summary.final_resolved || summary.resolved ? "resolved" : "unresolved";
  fields.passedCount.textContent = String(summary.final_passed_count ?? 0);
  fields.failedCount.textContent = String(summary.final_failed_count ?? 0);
  fields.applyMode.textContent = summary.final_patch_cleanly_applied
    ? "clean"
    : summary.final_patch_apply_mode || "unknown";
  fields.analysisText.textContent =
    analysis.root_cause ||
    "后端已返回修复结果，可在增强测试用例和最终补丁区域查看详细内容。";
}

async function loadInstanceInputs() {
  const instanceId = document.querySelector("#instance-id").value.trim();
  if (!instanceId) {
    return;
  }
  setFlowLocked(true);
  navigateTo("home");
  setMessage("正在根据实例 ID 加载缺陷代码和缺陷描述...");

  try {
    const response = await fetch(
      `${API_BASE}/api/results?instance_id=${encodeURIComponent(instanceId)}`,
    );
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "实例数据加载失败");
    }
    renderBugInput(result);
    setMessage(`已载入 ${result.instance_id} 的缺陷输入，请点击开始修复。`, "success");
  } catch (error) {
    setMessage(`${error.message}。请确认该实例已有运行结果。`, "error");
  }
}

function scheduleInstanceInputLoad() {
  window.clearTimeout(instanceLoadTimer);
  instanceLoadTimer = window.setTimeout(loadInstanceInputs, 420);
}

async function requestRepair(event) {
  event.preventDefault();

  const payload = {
    instance_id: document.querySelector("#instance-id").value.trim(),
    bug_code: document.querySelector("#bug-code").value,
    bug_description: document.querySelector("#bug-desc").value,
  };

  repairButton.disabled = true;
  setMessage("正在调用本地项目接口，读取增强测试和补丁结果...");

  try {
    const response = await fetch(`${API_BASE}/api/repair`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "修复接口调用失败");
    }
    renderResult(result);
    setFlowLocked(false);
    setMessage("修复已完成，可查看修复结果。", "success");
    navigateTo("tests");
  } catch (error) {
    setMessage(`${error.message}。请确认已通过 python3 ui/server.py 启动本地服务。`, "error");
  } finally {
    repairButton.disabled = false;
  }
}

navLinks.forEach((link) => {
  link.addEventListener("click", (event) => {
    const targetId = link.getAttribute("href")?.replace("#", "") || "home";
    if (targetId !== "home" && !repairCompleted) {
      event.preventDefault();
      navigateTo(targetId);
    }
  });
});

form.addEventListener("submit", requestRepair);
document.querySelector("#instance-id").addEventListener("input", scheduleInstanceInputLoad);
window.addEventListener("hashchange", syncViewFromHash);
setFlowLocked(true);
syncViewFromHash();
