from __future__ import annotations

import json
import re
import hashlib
import subprocess
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


UI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UI_DIR.parent
OUTPUT_ROOTS = [
    PROJECT_ROOT / "outputs" / "enhanced_patch_pipeline",
    PROJECT_ROOT / "outputs" / "enhanced_patch_pipeline_ui",
    PROJECT_ROOT / "outputs" / "enhanced_patch_pipeline_enhanced",
    PROJECT_ROOT / "outputs" / "enhanced_patch_pipeline_baseline",
]
KEY_INSTANCE_ID = "instance_id"
DEFAULT_INSTANCE_ID = "sympy__sympy-20590"
DATASET_CANDIDATES = [
    "SWE-bench/SWE-bench_Lite",
    "SWE-bench/SWE-bench",
]
RUNNING_JOBS: dict[str, dict] = {}
PREFERRED_RESULT_DIRS = {
    DEFAULT_INSTANCE_ID: PROJECT_ROOT
    / "outputs"
    / "enhanced_patch_pipeline"
    / "3"
    / "8"
    / DEFAULT_INSTANCE_ID
}


def load_dataset_instances(name: str, split: str, instance_ids: list[str]) -> list[dict]:
    from datasets import load_dataset, load_from_disk

    if name.lower() in {"swe-bench", "swebench", "swe_bench"}:
        name = "SWE-bench/SWE-bench"
    elif name.lower() in {
        "swe-bench-lite",
        "swebench-lite",
        "swe_bench_lite",
        "swe-bench_lite",
        "lite",
    }:
        name = "SWE-bench/SWE-bench_Lite"

    name_path = Path(name)
    parquet_path = name_path / f"{split}.parquet"
    if name.endswith(".json"):
        dataset = json.loads(Path(name).read_text(encoding="utf-8"))
    elif name.endswith(".jsonl"):
        dataset = [
            json.loads(line)
            for line in Path(name).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif name.endswith(".parquet"):
        dataset = load_dataset("parquet", data_files=name, split="train")
    elif parquet_path.exists():
        dataset = load_dataset("parquet", data_files=str(parquet_path), split="train")
    elif (name_path / split / "dataset_info.json").exists():
        dataset = load_from_disk(name_path / split)
    else:
        dataset = load_dataset(name, split=split)

    instance_id_set = set(instance_ids)
    return [
        dict(instance)
        for instance in dataset
        if instance.get(KEY_INSTANCE_ID) in instance_id_set
    ]


def read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path, default: str = "") -> str:
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8")


def candidate_run_score(instance_dir: Path) -> tuple[int, int, int, int, float]:
    summary_path = instance_dir / "summary.json"
    summary = read_json(summary_path, {})
    resolved_bonus = 1 if summary.get("final_resolved") or summary.get("resolved") else 0
    normal_run_bonus = 0 if "ablation" in instance_dir.parts else 1
    kept_tests = int(summary.get("kept_enhanced_candidates") or 0)
    has_filtered_tests = 1 if read_json(instance_dir / "enhanced_candidates_filtered.json", []) else 0
    mtime = summary_path.stat().st_mtime if summary_path.exists() else 0.0
    return resolved_bonus, normal_run_bonus, has_filtered_tests, kept_tests, mtime


def find_instance_dir(instance_id: str) -> Path | None:
    preferred_dir = PREFERRED_RESULT_DIRS.get(instance_id)
    if preferred_dir and preferred_dir.exists():
        return preferred_dir

    candidates: list[Path] = []
    for output_root in OUTPUT_ROOTS:
        if output_root.exists():
            candidates.extend(path for path in output_root.rglob(instance_id) if path.is_dir())
    if not candidates:
        return None
    return max(candidates, key=candidate_run_score)


def first_source_context(code_context: dict, summary: dict) -> str:
    analysis = summary.get("patch_analysis") or {}
    source_components = [
        item
        for item in analysis.get("affected_components", [])
        if item.get("file") and "test" not in item.get("file", "").lower()
    ]
    for component in source_components:
        content = code_context.get(component.get("file"))
        symbol = component.get("symbol")
        if content and symbol:
            symbol_context = extract_symbol_context(content, symbol)
            if symbol_context:
                return symbol_context

    target_files = [item.get("file") for item in source_components]
    target_files.extend(
        path
        for path in analysis.get("suggested_repair_scope", [])
        if "test" not in path.lower()
    )
    target_files.extend(path for path in code_context if not path.startswith("__tail__"))

    for target_file in target_files:
        content = code_context.get(target_file)
        if content:
            return trim_context(content)
    return ""


def extract_symbol_context(content: str, symbol: str) -> str:
    pattern = re.compile(rf"^(class|def)\s+{re.escape(symbol)}\b.*$", re.MULTILINE)
    match = pattern.search(content)
    if not match:
        return ""
    start = match.start()
    next_block = re.search(r"^(class|def)\s+\w+\b", content[match.end() :], re.MULTILINE)
    end = match.end() + next_block.start() if next_block else len(content)
    return trim_context(content[start:end])


def trim_context(content: str, max_lines: int = 80) -> str:
    lines = content.strip().splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[:max_lines] + ["..."])


def build_bug_description(summary: dict, failure_focus: dict) -> str:
    analysis = summary.get("patch_analysis") or {}
    parts = [
        analysis.get("root_cause"),
        analysis.get("failing_signal"),
        analysis.get("repair_constraint"),
    ]
    if not any(parts):
        failing_tests = failure_focus.get("failing_tests_sample") or []
        dominant_errors = failure_focus.get("dominant_errors") or []
        parts = [
            f"失败测试：{', '.join(failing_tests)}" if failing_tests else "",
            f"主要错误：{', '.join(dominant_errors)}" if dominant_errors else "",
        ]
    return "\n\n".join(part for part in parts if part)


def extract_patch_comparison(patch: str) -> tuple[str, str]:
    before_blocks: list[str] = []
    after_blocks: list[str] = []
    current_file = ""
    in_hunk = False

    for line in patch.splitlines():
        if line.startswith("--- a/"):
            current_file = line[6:]
            continue
        if line.startswith("diff --git "):
            current_file = line.split(" b/")[-1] if " b/" in line else current_file
            continue
        if line.startswith("@@"):
            in_hunk = True
            header = f"# {current_file}\n{line}" if current_file else line
            before_blocks.append(header)
            after_blocks.append(header)
            continue
        if not in_hunk:
            continue
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("-"):
            before_blocks.append(line[1:])
        elif line.startswith("+"):
            after_blocks.append(line[1:])
        elif line.startswith(" "):
            before_blocks.append(line[1:])
            after_blocks.append(line[1:])
        elif line.startswith("\\"):
            continue
        else:
            before_blocks.append(line)
            after_blocks.append(line)

    return "\n".join(before_blocks).strip(), "\n".join(after_blocks).strip()


def normalize_patch_result(raw_result: dict, instance_dir: Path, index: int) -> dict | None:
    patch = raw_result.get("patch") or raw_result.get("diff") or raw_result.get("model_patch") or ""
    if not patch.strip():
        return None
    summary = raw_result.get("summary") or read_json(instance_dir / "summary.json", {})
    before_code = raw_result.get("before_code")
    after_code = raw_result.get("after_code")
    if not before_code or not after_code:
        before_code, after_code = extract_patch_comparison(patch)
    patch_hash = hashlib.sha1(patch.encode("utf-8")).hexdigest()[:10]
    return {
        "id": raw_result.get("id") or patch_hash,
        "label": raw_result.get("label")
        or f"{'已通过' if summary.get('final_resolved') else '未通过'} · 补丁 {index + 1}",
        "source": str(instance_dir.relative_to(PROJECT_ROOT)),
        "patch": patch,
        "before_code": before_code,
        "after_code": after_code,
        "summary": summary,
    }


def collect_patch_results(instance_dir: Path) -> list[dict]:
    results: list[dict] = []
    seen_hashes: set[str] = set()
    for filename in ("patch_results.json", "patch_candidates.json"):
        raw_results = read_json(instance_dir / filename, [])
        if not isinstance(raw_results, list):
            continue
        for raw_result in raw_results:
            if not isinstance(raw_result, dict):
                continue
            normalized = normalize_patch_result(raw_result, instance_dir, len(results))
            if not normalized:
                continue
            patch_hash = hashlib.sha1(normalized["patch"].encode("utf-8")).hexdigest()[:10]
            if patch_hash in seen_hashes:
                continue
            seen_hashes.add(patch_hash)
            results.append(normalized)

    patch = read_text(instance_dir / "model_patch.diff")
    if patch.strip():
        normalized = normalize_patch_result({"patch": patch, "label": "当前最终补丁"}, instance_dir, len(results))
        patch_hash = hashlib.sha1(patch.encode("utf-8")).hexdigest()[:10]
        if normalized and patch_hash not in seen_hashes:
            results.append(normalized)
    return results


def candidate_key(candidate: dict) -> tuple:
    identifiers = candidate.get("identifiers") or candidate.get("enhanced_identifiers") or []
    return tuple(identifiers), candidate.get("patch", "")


def collect_enhanced_tests(instance_dir: Path) -> tuple[list[dict], list[dict]]:
    filtered = read_json(instance_dir / "enhanced_candidates_filtered.json", [])
    raw = read_json(instance_dir / "enhanced_candidates_raw.json", [])
    rejected = read_json(instance_dir / "enhanced_candidates_rejected.json", [])

    selected: list[dict] = []
    seen: set[tuple] = set()
    for source in (filtered, raw):
        if not isinstance(source, list):
            continue
        for candidate in source:
            if not isinstance(candidate, dict):
                continue
            if source is raw and not candidate.get("kept", False):
                continue
            key = candidate_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            selected.append(candidate)

    return selected, rejected if isinstance(rejected, list) else []


def build_result(instance_id: str, bug_code: str = "", bug_description: str = "") -> dict:
    instance_dir = find_instance_dir(instance_id)
    if instance_dir is None:
        raise FileNotFoundError(f"未找到实例 {instance_id} 的运行结果")

    summary = read_json(instance_dir / "summary.json", {})
    code_context = read_json(instance_dir / "code_context.json", {})
    failure_focus = read_json(instance_dir / "failure_focus.json", {})
    enhanced_tests, rejected_tests = collect_enhanced_tests(instance_dir)
    default_bug_code = first_source_context(code_context, summary)
    default_bug_description = build_bug_description(summary, failure_focus)

    return {
        "instance_id": instance_id,
        "source": str(instance_dir.relative_to(PROJECT_ROOT)),
        "bug_code": bug_code or default_bug_code,
        "bug_description": bug_description or default_bug_description,
        "summary": summary,
        "enhanced_tests": enhanced_tests,
        "rejected_tests": rejected_tests,
        "patch": read_text(instance_dir / "model_patch.diff"),
        "patch_results": collect_patch_results(instance_dir),
        "patch_response": read_text(instance_dir / "patch_response.txt"),
        "original_failure": read_json(instance_dir / "original_failure.json", {}),
        "failure_focus": failure_focus,
    }


def find_dataset_instance(instance_id: str) -> tuple[dict, str]:
    last_error: Exception | None = None
    for dataset_name in DATASET_CANDIDATES:
        try:
            dataset = load_dataset_instances(dataset_name, "test", [instance_id])
            if dataset:
                return dict(dataset[0]), dataset_name
        except Exception as exc:
            last_error = exc
    detail = f"；查询数据集失败：{last_error}" if last_error else ""
    raise FileNotFoundError(f"数据集中未找到实例 {instance_id}{detail}")


def build_dataset_preview(instance: dict, dataset_name: str) -> dict:
    problem_statement = instance.get("problem_statement") or ""
    repo = instance.get("repo") or ""
    version = instance.get("version") or ""
    return {
        "instance_id": instance.get(KEY_INSTANCE_ID),
        "source": dataset_name,
        "bug_code": (
            f"# {repo} {version}\n"
            "# 本地尚无该实例的代码上下文。点击“开始修复”后，系统会调用项目流水线构建上下文并生成补丁。"
        ).strip(),
        "bug_description": problem_statement.strip(),
        "summary": {},
        "enhanced_tests": [],
        "rejected_tests": [],
        "patch": "",
        "patch_response": "",
        "original_failure": {},
        "failure_focus": {},
        "status": "dataset_only",
    }


def run_real_pipeline(instance_id: str, dataset_name: str) -> subprocess.Popen:
    command = [
        sys.executable,
        "-m",
        "swebench.experiments.enhanced_patch_pipeline",
        "--dataset_name",
        dataset_name,
        "--instance_ids",
        instance_id,
        "--output_dir",
        str(PROJECT_ROOT / "outputs" / "enhanced_patch_pipeline_ui"),
    ]
    return subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def start_repair_job(instance_id: str) -> dict:
    existing_job = RUNNING_JOBS.get(instance_id)
    if existing_job and existing_job.get("process").poll() is None:
        try:
            instance, dataset_name = find_dataset_instance(instance_id)
            preview = build_dataset_preview(instance, dataset_name)
        except FileNotFoundError:
            preview = {}
        return {
            **preview,
            "instance_id": instance_id,
            "pid": existing_job["process"].pid,
            "dataset_name": existing_job["dataset_name"],
            "message": "该实例的自动修复任务已在运行，请稍后查看结果。",
            "status": "running",
        }

    instance, dataset_name = find_dataset_instance(instance_id)
    process = run_real_pipeline(instance_id, dataset_name)
    RUNNING_JOBS[instance_id] = {
        "process": process,
        "dataset_name": dataset_name,
    }
    preview = build_dataset_preview(instance, dataset_name)
    return {
        **preview,
        "pid": process.pid,
        "dataset_name": dataset_name,
        "message": "本地尚无该实例结果，已从数据集中找到实例并启动自动修复任务。",
        "status": "started",
    }


class RepairRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def write_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_request_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw_body = self.rfile.read(length).decode("utf-8")
        return json.loads(raw_body or "{}")

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/results":
            query = parse_qs(parsed.query)
            instance_id = query.get("instance_id", [DEFAULT_INSTANCE_ID])[0] or DEFAULT_INSTANCE_ID
            try:
                self.write_json(build_result(instance_id))
            except FileNotFoundError:
                try:
                    self.write_json(start_repair_job(instance_id), HTTPStatus.ACCEPTED)
                except FileNotFoundError as dataset_exc:
                    self.write_json({"error": str(dataset_exc)}, HTTPStatus.NOT_FOUND)
            return
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/repair":
            self.write_json({"error": "未知接口"}, HTTPStatus.NOT_FOUND)
            return

        try:
            payload = self.read_request_json()
            instance_id = payload.get("instance_id") or DEFAULT_INSTANCE_ID
            if payload.get("run_real_pipeline"):
                self.write_json(start_repair_job(instance_id), HTTPStatus.ACCEPTED)
                return
            try:
                self.write_json(
                    build_result(
                        instance_id=instance_id,
                        bug_code=payload.get("bug_code", ""),
                        bug_description=payload.get("bug_description", ""),
                    )
                )
            except FileNotFoundError:
                self.write_json(start_repair_job(instance_id), HTTPStatus.ACCEPTED)
        except FileNotFoundError as exc:
            self.write_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        except json.JSONDecodeError:
            self.write_json({"error": "请求体不是合法 JSON"}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self.write_json({"error": f"服务端处理失败：{exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> None:
    host = "127.0.0.1"
    port = 8001
    server = ThreadingHTTPServer((host, port), RepairRequestHandler)
    print(f"缺陷修复界面已启动：http://{host}:{port}")
    print("按 Ctrl+C 停止服务")
    server.serve_forever()


if __name__ == "__main__":
    main()
